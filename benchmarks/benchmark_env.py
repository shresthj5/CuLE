import argparse
import json
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torchcule_atari
from torchcule.atari import Env


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CuLE Atari environment throughput")
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="ALE environment id")
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device")
    parser.add_argument("--color-mode", type=str, default="gray", choices=["gray", "rgb"], help="Observation color mode")
    parser.add_argument("--rescale", action="store_true", help="Resize observations to 84x84")
    parser.add_argument("--steps", type=int, default=1000, help="Measured benchmark iterations")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup iterations excluded from timing")
    parser.add_argument("--action-mode", type=str, default="random", choices=["random", "noop"], help="Action generation mode")
    parser.add_argument("--frameskip", type=int, default=None, help="Override ALE/CuLE frameskip")
    parser.add_argument("--repeat-prob", type=float, default=None, help="Override ALE/CuLE repeat action probability")
    parser.add_argument("--seed", type=int, default=123, help="Base seed used to initialize environments")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(args):
    device = torch.device(args.device)
    env = Env(
        args.env_name,
        args.num_envs,
        color_mode=args.color_mode,
        device=device,
        rescale=args.rescale,
        frameskip=args.frameskip,
        repeat_prob=args.repeat_prob,
    )

    seeds = torch.arange(args.num_envs, device=device, dtype=torch.int32) + args.seed
    env.reset(seeds=seeds, initial_steps=0)

    if args.action_mode == "noop":
        actions = torch.full((args.num_envs,), env.noop_action_index, device=device, dtype=torch.uint8)

        def sample_actions():
            return actions
    else:
        def sample_actions():
            return env.sample_random_actions()

    for _ in range(args.warmup_steps):
        env.step(sample_actions())

    synchronize(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        env.step(sample_actions())
    synchronize(device)
    elapsed = time.perf_counter() - start

    env_steps = args.steps * args.num_envs
    steps_per_second = env_steps / elapsed
    steps_per_hour = steps_per_second * 3600.0
    emulator_frames = env_steps * env.frameskip
    emulator_frames_per_second = steps_per_second * env.frameskip
    emulator_frames_per_hour = steps_per_hour * env.frameskip

    return {
        "env_name": args.env_name,
        "device": str(device),
        "atari_env_block_size": int(torchcule_atari.ATARI_ENV_BLOCK_SIZE),
        "atari_process_block_size": int(torchcule_atari.ATARI_PROCESS_BLOCK_SIZE),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "action_mode": args.action_mode,
        "frameskip": env.frameskip,
        "repeat_prob": env.repeat_prob,
        "elapsed_seconds": elapsed,
        "env_steps": env_steps,
        "env_steps_per_second": steps_per_second,
        "env_steps_per_hour": steps_per_hour,
        "emulator_frames": emulator_frames,
        "emulator_frames_per_second": emulator_frames_per_second,
        "emulator_frames_per_hour": emulator_frames_per_hour,
    }


def main():
    args = parse_args()
    result = benchmark(args)

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("env_name           :", result["env_name"])
        print("device             :", result["device"])
        print("atari_env_block_sz :", result["atari_env_block_size"])
        print("atari_proc_block_sz:", result["atari_process_block_size"])
        print("num_envs           :", result["num_envs"])
        print("action_mode        :", result["action_mode"])
        print("frameskip          :", result["frameskip"])
        print("repeat_prob        :", result["repeat_prob"])
        print("elapsed_seconds    :", round(result["elapsed_seconds"], 6))
        print("env_steps          :", result["env_steps"])
        print("env_steps_per_sec  :", round(result["env_steps_per_second"], 2))
        print("env_steps_per_hour :", round(result["env_steps_per_hour"], 2))
        print("emu_frames         :", result["emulator_frames"])
        print("emu_frames_per_sec :", round(result["emulator_frames_per_second"], 2))
        print("emu_frames_per_hour:", round(result["emulator_frames_per_hour"], 2))


if __name__ == "__main__":
    main()
