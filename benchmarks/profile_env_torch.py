import argparse
from pathlib import Path
import sys

import torch
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchcule.atari import Env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile CuLE Atari env.step with the PyTorch profiler"
    )
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="ALE environment id")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device")
    parser.add_argument(
        "--color-mode",
        type=str,
        default="gray",
        choices=["gray", "rgb"],
        help="Observation color mode",
    )
    parser.add_argument("--rescale", action="store_true", help="Resize observations to 84x84")
    parser.add_argument("--steps", type=int, default=5, help="Profiled iterations")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup iterations excluded from the table")
    parser.add_argument(
        "--action-mode",
        type=str,
        default="random",
        choices=["random", "noop"],
        help="Action generation mode",
    )
    parser.add_argument("--frameskip", type=int, default=None, help="Override ALE/CuLE frameskip")
    parser.add_argument(
        "--repeat-prob", type=float, default=None, help="Override ALE/CuLE repeat action probability"
    )
    parser.add_argument("--seed", type=int, default=123, help="Base seed used to initialize environments")
    parser.add_argument("--row-limit", type=int, default=20, help="Number of rows shown in the profiler table")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="self_cuda_time_total",
        help="torch.profiler table sort column",
    )
    parser.add_argument(
        "--asyn",
        action="store_true",
        help="Pass asyn=True to env.step and synchronize only after the loop",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Optional directory where a Chrome trace is written",
    )
    return parser.parse_args()


def build_env(args):
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
    return device, env


def build_action_sampler(args, env):
    if args.action_mode == "noop":
        actions = torch.full(
            (args.num_envs,),
            env.noop_action_index,
            device=env.device,
            dtype=torch.uint8,
        )

        def sample_actions():
            return actions

        return sample_actions

    def sample_actions():
        return env.sample_random_actions()

    return sample_actions


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    device, env = build_env(args)
    sample_actions = build_action_sampler(args, env)

    for _ in range(args.warmup_steps):
        env.step(sample_actions(), asyn=args.asyn)
    synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(args.steps):
            env.step(sample_actions(), asyn=args.asyn)
        synchronize(device)

    print(prof.key_averages().table(sort_by=args.sort_by, row_limit=args.row_limit))

    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = args.trace_dir / "torch-profiler-trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"Chrome trace written to {trace_path}")


if __name__ == "__main__":
    main()
