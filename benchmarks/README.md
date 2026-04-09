# Benchmarking CuLE Atari v5

This directory contains the throughput and profiler entrypoints for modernized CuLE Atari benchmarking on Atari v5 semantics.

## Throughput harness

Run the env-only benchmark from the repo root:

```bash
source .venv/bin/activate
python benchmarks/benchmark_env.py --steps 200 --warmup-steps 50 --num-envs 1024 --device cuda:0 --json
```

The benchmark reports:

- the Atari env id
- the device
- the compile-time `ATARI_ENV_BLOCK_SIZE`
- the compile-time `ATARI_PROCESS_BLOCK_SIZE`
- the effective `frameskip` and `repeat_prob`
- `env_steps_per_second`
- `env_steps_per_hour`
- `emulator_frames_per_second`
- `emulator_frames_per_hour`

`env_steps_per_second` is full `Env.step()` throughput. On Atari v5 that is not raw emulator-frame rate because the default `frameskip` is `4`, so the harness also reports the derived raw emulator-frame metrics.

## Rebuilding with different CUDA launch sizes

The hot CUDA env kernels and the frame preprocessing kernel are compiled with fixed launch sizes so that `__launch_bounds__` stays consistent with the generated code.

Supported values:

- `1`
- `32`
- `64`
- `128`
- `256`

Local tuning build for the RTX 3070:

```bash
source .venv/bin/activate
TORCH_CUDA_ARCH_LIST=8.6 \
CULE_ATARI_ENV_BLOCK_SIZE=1 \
CULE_ATARI_PROCESS_BLOCK_SIZE=64 \
python setup.py build_ext --inplace
```

Publication-style Ampere build for both RTX 3070 and A100:

```bash
source .venv/bin/activate
TORCH_CUDA_ARCH_LIST="8.0 8.6" python setup.py build_ext --inplace
```

## Current 3070 measurements

Measured on the local RTX 3070 on April 8, 2026.

Baseline build:

```bash
TORCH_CUDA_ARCH_LIST=8.6 \
CULE_ATARI_ENV_BLOCK_SIZE=1 \
CULE_ATARI_PROCESS_BLOCK_SIZE=1 \
python setup.py build_ext --inplace
```

Observed env-only results:

- `ENV=1, PROCESS=1, 1024 envs`: about `17.19k env-steps/s` or `61.87M env-steps/hour`
- `ENV=1, PROCESS=1, 4096 envs`: about `20.62k env-steps/s` or `74.22M env-steps/hour`

Process-kernel sweep with `ENV=1`:

- `PROCESS=32, 4096 envs`: about `30.56k env-steps/s` or `110.03M env-steps/hour`
- `PROCESS=32, 8192 envs`: about `32.39k env-steps/s` or `116.61M env-steps/hour`
- `PROCESS=64, 4096 envs`: about `32.63k env-steps/s` or `117.46M env-steps/hour`
- `PROCESS=64, 8192 envs`: about `35.55k env-steps/s` or `127.99M env-steps/hour`
- `PROCESS=128, 4096 envs`: about `31.70k env-steps/s` or `114.11M env-steps/hour`
- `PROCESS=128, 8192 envs`: about `34.91k env-steps/s` or `125.67M env-steps/hour`

Current best finding:

- `CULE_ATARI_ENV_BLOCK_SIZE=1` and `CULE_ATARI_PROCESS_BLOCK_SIZE=64` is the fastest measured configuration on this 3070.
- The earlier process-block experiments were initially invalid because the tuning knob was accidentally wired to reset-time initialization instead of `process_kernel`. That bug is now fixed in the dispatch layer.
- `ATARI_ENV_BLOCK_SIZE=32` remains a correctness-sensitive path because RAM storage becomes interleaved. The associated `get_data`, `get_states`, and `set_states` layout bug is fixed, but the env kernel itself is still slower than `1` on this card.

## Nsight Compute workflow

The Nsight Compute wrapper profiles the env benchmark with the sections most relevant to the current tuning hypothesis:

- `SpeedOfLight`
- `Occupancy`
- `LaunchStats`
- `SchedulerStats`
- `WarpStateStats`
- `MemoryWorkloadAnalysis`

Run it like this:

```bash
benchmarks/profile_env_ncu.sh --steps 200 --warmup-steps 50 --num-envs 1024 --device cuda:0
```

The report is written under `benchmarks/results/`.

If `ncu` fails with `ERR_NVGPUCTRPERM`, the current shell does not have permission to access NVIDIA GPU performance counters. On this WSL/WDDM setup, retrying `ncu` under `sudo` still failed, which strongly suggests the remaining gate is the Windows host-side performance-counter setting documented by NVIDIA here:

- https://developer.nvidia.com/ERR_NVGPUCTRPERM

## PyTorch profiler fallback

When Nsight Compute counters are unavailable, use the PyTorch profiler entrypoint:

```bash
source .venv/bin/activate
python benchmarks/profile_env_torch.py --num-envs 256 --steps 5 --warmup-steps 3 --row-limit 15
```

This profiler worked reliably in the current environment and showed that:

- `step_kernel` is still the largest CUDA hotspot
- `process_kernel` is the second-largest CUDA hotspot
- `get_data_kernel` is comparatively small
- preprocess launch tuning materially changes large-env throughput even when the small-env profile looks noisy

## Why these sections

These sections line up with the current modernization question:

- whether the env kernels are under-occupying the GPU
- whether warp scheduling is starved by low eligible warps
- whether memory behavior regresses when the launch geometry changes
- whether a preprocess launch-size change trades registers for usable occupancy on Ampere

That matches the guidance in NVIDIA's documentation:

- CUDA C++ Best Practices Guide: memory optimizations and coalesced access
- CUDA C++ Programming Guide: `__launch_bounds__` and launch configuration effects
- Nsight Compute Profiling Guide: `SpeedOfLight`, `Occupancy`, `LaunchStats`, `SchedulerStats`, and `WarpStateStats`
