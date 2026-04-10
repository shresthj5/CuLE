#!/usr/bin/env python3
"""Validate CuLE's currently supported ALE v5 subset against ale-py.

This script intentionally separates checks that should pass today from strict
black-box ALE trajectory parity, which is reported as a research-validity gate.
"""

import argparse
import hashlib
import importlib.resources as resources
import json
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch

import torchcule_atari
from tests.test_v5_supported_games import SUPPORTED_V5_ENVS
from torchcule.atari import Env, Rom


ACTION_NAMES = (
    "NOOP",
    "FIRE",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
    "UPFIRE",
    "RIGHTFIRE",
    "LEFTFIRE",
    "DOWNFIRE",
    "UPRIGHTFIRE",
    "UPLEFTFIRE",
    "DOWNRIGHTFIRE",
    "DOWNLEFTFIRE",
)

CULE_ACTION_NAME_BY_VALUE = {
    int(getattr(torchcule_atari, action_name)): action_name
    for action_name in ACTION_NAMES
}

TRANSIENT_STATE_FIELDS = frozenset({
    # CuLE derives reward from the previous score, but that previous-score
    # baseline is not represented in the public AtariState bridge.
    "reward",
})


def _jsonify(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    return value


def _action_indices(step, num_envs, action_count):
    return [(step + env_index) % action_count for env_index in range(num_envs)]


def _ale_action_index(step, action_count):
    return (step * 7 + 3) % action_count


def _public_state_dict(atari_state):
    result = {}
    for name in dir(atari_state):
        if name.startswith("_"):
            continue
        if name in TRANSIENT_STATE_FIELDS:
            continue
        value = getattr(atari_state, name)
        if isinstance(value, list):
            result[name] = list(value)
        elif isinstance(value, (bool, int, float, str)):
            result[name] = value
    return result


def _first_state_diff(left, right):
    for key in sorted(left):
        if key not in right:
            return {"field": key, "left": left[key], "right": "<missing>"}
        if left[key] != right[key]:
            if isinstance(left[key], list) and isinstance(right[key], list):
                differing = sum(1 for a, b in zip(left[key], right[key]) if a != b)
                return {
                    "field": key,
                    "differing_entries": differing,
                    "left_head": left[key][:8],
                    "right_head": right[key][:8],
                }
            return {"field": key, "left": left[key], "right": right[key]}
    return None


def _cule_action_names(env):
    return [CULE_ACTION_NAME_BY_VALUE[int(value)] for value in env.action_set.cpu().tolist()]


def _ale_action_names(env):
    return [action.name for action in env.unwrapped.ale.getMinimalActionSet()]


def validate_metadata():
    gym.register_envs(ale_py)
    failures = []

    for env_name in SUPPORTED_V5_ENVS:
        spec = gym.spec(env_name)
        game_name = spec.kwargs["game"]
        rom_path = Path(resources.files("ale_py.roms") / f"{game_name}.bin")
        expected_md5 = hashlib.md5(rom_path.read_bytes()).hexdigest()

        rom = Rom(env_name)
        if rom.md5() != expected_md5:
            failures.append({
                "env": env_name,
                "check": "rom_md5",
                "cule": rom.md5(),
                "ale_py": expected_md5,
            })

        ale_env = gym.make(
            env_name,
            obs_type="grayscale",
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=False,
        )
        cule_env = Env(
            env_name,
            1,
            color_mode="gray",
            device="cpu",
            rescale=False,
            frameskip=1,
            repeat_prob=0.0,
        )

        ale_actions = _ale_action_names(ale_env)
        cule_actions = _cule_action_names(cule_env)
        if ale_actions != cule_actions:
            failures.append({
                "env": env_name,
                "check": "minimal_action_meanings",
                "cule": cule_actions,
                "ale_py": ale_actions,
            })

        v5_default_env = Env(env_name, 1, color_mode="gray", device="cpu", rescale=False)
        if v5_default_env.frameskip != 4 or v5_default_env.repeat_prob != 0.25:
            failures.append({
                "env": env_name,
                "check": "v5_default_semantics",
                "frameskip": v5_default_env.frameskip,
                "repeat_prob": v5_default_env.repeat_prob,
            })

        ale_env.close()

    return failures


def validate_state_roundtrip(devices):
    failures = []

    for device_name in devices:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            continue

        for env_name in SUPPORTED_V5_ENVS:
            env = Env(
                env_name,
                1,
                color_mode="gray",
                device=device,
                rescale=False,
                frameskip=1,
                repeat_prob=0.0,
            )
            seed = torch.tensor([123], dtype=torch.int32, device=device)
            env.reset(seeds=seed, initial_steps=0)

            before = torchcule_atari.AtariEnv.get_states(env, [0])[0]
            torchcule_atari.AtariEnv.set_states(env, [0], [before])
            after = torchcule_atari.AtariEnv.get_states(env, [0])[0]

            diff = _first_state_diff(_public_state_dict(before), _public_state_dict(after))
            if diff is not None:
                failures.append({"env": env_name, "device": device_name, "diff": diff})

    return failures


def validate_restored_cpu_cuda(steps, num_envs):
    if not torch.cuda.is_available():
        return {"transition_failures": [], "pixel_failures": [], "skipped": "CUDA is not available"}

    transition_failures = []
    pixel_failures = []

    for env_name in SUPPORTED_V5_ENVS:
        cpu_env = Env(
            env_name,
            num_envs,
            color_mode="gray",
            device="cpu",
            rescale=False,
            frameskip=1,
            repeat_prob=0.0,
        )
        cuda_env = Env(
            env_name,
            num_envs,
            color_mode="gray",
            device="cuda:0",
            rescale=False,
            frameskip=1,
            repeat_prob=0.0,
        )
        seeds = torch.arange(123, 123 + num_envs, dtype=torch.int32)
        indices = list(range(num_envs))

        cpu_env.reset(seeds=seeds, initial_steps=0)
        cuda_env.reset(seeds=seeds.to("cuda:0"), initial_steps=0)

        states = torchcule_atari.AtariEnv.get_states(cpu_env, indices)
        torchcule_atari.AtariEnv.set_states(cpu_env, indices, states)
        torchcule_atari.AtariEnv.set_states(cuda_env, indices, states)

        for step in range(steps):
            actions = torch.tensor(
                _action_indices(step, num_envs, cpu_env.action_space.n),
                dtype=torch.uint8,
            )
            cpu_obs, cpu_rewards, cpu_done, cpu_info = cpu_env.step(actions)
            cuda_obs, cuda_rewards, cuda_done, cuda_info = cuda_env.step(actions.to("cuda:0"))
            torch.cuda.synchronize()

            rewards_match = torch.equal(cpu_rewards, cuda_rewards.cpu())
            done_match = torch.equal(cpu_done, cuda_done.cpu())
            lives_match = torch.equal(cpu_info["ale.lives"], cuda_info["ale.lives"].cpu())
            pixels_match = torch.equal(cpu_obs, cuda_obs.cpu())

            if not (rewards_match and done_match and lives_match):
                transition_failures.append({
                    "env": env_name,
                    "step": step,
                    "rewards_match": rewards_match,
                    "done_match": done_match,
                    "lives_match": lives_match,
                    "cpu_rewards": cpu_rewards.tolist(),
                    "cuda_rewards": cuda_rewards.cpu().tolist(),
                    "cpu_done": cpu_done.tolist(),
                    "cuda_done": cuda_done.cpu().tolist(),
                    "cpu_lives": cpu_info["ale.lives"].tolist(),
                    "cuda_lives": cuda_info["ale.lives"].cpu().tolist(),
                })
                break

            if not pixels_match:
                diff = (cpu_obs.cpu().to(torch.int16) - cuda_obs.cpu().to(torch.int16)).abs()
                pixel_failures.append({
                    "env": env_name,
                    "step": step,
                    "differing_pixels": int((cpu_obs.cpu() != cuda_obs.cpu()).sum().item()),
                    "max_abs_diff": int(diff.max().item()),
                    "sum_abs_diff": int(diff.sum().item()),
                })
                break

    return {
        "transition_failures": transition_failures,
        "pixel_failures": pixel_failures,
    }


def validate_ale_blackbox(steps):
    gym.register_envs(ale_py)
    failures = []

    for env_name in SUPPORTED_V5_ENVS:
        ale_env = gym.make(
            env_name,
            obs_type="grayscale",
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=False,
        )
        cule_env = Env(
            env_name,
            1,
            color_mode="gray",
            device="cpu",
            rescale=False,
            frameskip=1,
            repeat_prob=0.0,
        )

        ale_env.reset(seed=123)
        cule_env.reset(seeds=torch.tensor([123], dtype=torch.int32), initial_steps=0)

        for step in range(steps):
            action = _ale_action_index(step, cule_env.action_space.n)
            ale_obs, ale_reward, ale_terminated, ale_truncated, ale_info = ale_env.step(action)
            cule_obs, cule_reward, cule_done, cule_info = cule_env.step(torch.tensor([action], dtype=torch.uint8))
            cule_obs = cule_obs.numpy()[0, :, :, 0]

            mismatches = []
            if tuple(ale_obs.shape) != tuple(cule_obs.shape):
                mismatches.append({
                    "kind": "shape",
                    "ale_py": tuple(ale_obs.shape),
                    "cule": tuple(cule_obs.shape),
                })
            if float(ale_reward) != float(cule_reward.item()):
                mismatches.append({
                    "kind": "reward",
                    "ale_py": float(ale_reward),
                    "cule": float(cule_reward.item()),
                })
            if bool(ale_terminated or ale_truncated) != bool(cule_done.item()):
                mismatches.append({
                    "kind": "done",
                    "ale_py": bool(ale_terminated or ale_truncated),
                    "cule": bool(cule_done.item()),
                })
            if int(ale_info["lives"]) != int(cule_info["ale.lives"].item()):
                mismatches.append({
                    "kind": "lives",
                    "ale_py": int(ale_info["lives"]),
                    "cule": int(cule_info["ale.lives"].item()),
                })
            if tuple(ale_obs.shape) == tuple(cule_obs.shape) and not np.array_equal(ale_obs, cule_obs):
                diff = np.abs(ale_obs.astype(np.int16) - cule_obs.astype(np.int16))
                mismatches.append({
                    "kind": "pixels",
                    "differing_pixels": int((ale_obs != cule_obs).sum()),
                    "max_abs_diff": int(diff.max()),
                    "sum_abs_diff": int(diff.sum()),
                })

            if mismatches:
                failures.append({
                    "env": env_name,
                    "step": step,
                    "action_index": int(action),
                    "mismatches": mismatches,
                })
                break

        ale_env.close()

    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--strict-ale-parity", action="store_true")
    args = parser.parse_args()

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")

    metadata_failures = validate_metadata()
    state_roundtrip_failures = validate_state_roundtrip(devices)
    restored_cpu_cuda = validate_restored_cpu_cuda(args.steps, args.num_envs)
    ale_blackbox_failures = validate_ale_blackbox(args.steps)

    result = {
        "supported_v5_env_count": len(SUPPORTED_V5_ENVS),
        "state_roundtrip_excluded_fields": sorted(TRANSIENT_STATE_FIELDS),
        "metadata_failures": metadata_failures,
        "state_roundtrip_failures": state_roundtrip_failures,
        "restored_cpu_cuda": restored_cpu_cuda,
        "ale_blackbox_failures": ale_blackbox_failures,
        "summary": {
            "metadata_failure_count": len(metadata_failures),
            "state_roundtrip_failure_count": len(state_roundtrip_failures),
            "restored_cpu_cuda_transition_failure_count": len(restored_cpu_cuda.get("transition_failures", [])),
            "restored_cpu_cuda_pixel_failure_count": len(restored_cpu_cuda.get("pixel_failures", [])),
            "ale_blackbox_failure_count": len(ale_blackbox_failures),
        },
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(_jsonify(result), indent=2, sort_keys=True) + "\n")

    print(json.dumps(_jsonify(result["summary"]), indent=2, sort_keys=True))

    should_fail = (
        metadata_failures
        or state_roundtrip_failures
        or restored_cpu_cuda.get("transition_failures")
        or (args.strict_ale_parity and ale_blackbox_failures)
    )
    return 1 if should_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
