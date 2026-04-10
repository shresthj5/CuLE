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
CULE_ATARI_ENV_BLOCK_SIZE=32 \
CULE_ATARI_PROCESS_BLOCK_SIZE=64 \
python setup.py build_ext --inplace
```

Publication-style Ampere build for both RTX 3070 and A100:

```bash
source .venv/bin/activate
TORCH_CUDA_ARCH_LIST="8.0 8.6" python setup.py build_ext --inplace
```

## Current 3070 measurements

Measured on the local RTX 3070 on April 8-9, 2026. These are env-only Pong v5 results with Atari v5 defaults (`frameskip=4`, `repeat_prob=0.25`) and random action sampling included in the timed loop.

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

Current tuned default after Nsight-guided env-kernel tuning:

- `ENV=32, PROCESS=64, 4096 envs`: about `31.09k env-steps/s` or `111.94M env-steps/hour`
- `ENV=32, PROCESS=64, 8192 envs`: about `53.20k env-steps/s` or `191.52M env-steps/hour`
- At Atari v5 `frameskip=4`, the 8192-env result corresponds to about `766.08M emulator frames/hour`.
- A short April 10, 2026 8192-env rebuild spot check measured about `59.13k env-steps/s`, `212.87M env-steps/hour`, or `851.48M emulator frames/hour`.

Important correctness note:

- `ENV=32` requires an interleaved CUDA RAM layout. This path now pads the CUDA RAM allocation to the block geometry and specializes CUDA kernels to 128-byte RAM for normal cartridges and 256-byte RAM for F8SC/Superchip cartridges.
- The full audited v5 smoke test matrix passes on CPU and CUDA with `ENV=32, PROCESS=64`.
- `ALE/ElevatorAction-v5`, the F8SC/Superchip representative, passes CUDA spot benchmarks with the corrected 256-byte RAM path. A short April 10, 2026 8192-env spot run measured about `260.21M env-steps/hour`.
- A100 compatibility is built through `TORCH_CUDA_ARCH_LIST="8.0 8.6"`, but the A100 runtime performance number still needs to be measured on an actual A100.

## Current v5 validation

The stricter validation entrypoint checks the currently supported ALE v5 subset:

```bash
source .venv/bin/activate
python tools/validate_ale_v5_subset.py --steps 16 --num-envs 2 --json-output benchmarks/results/ale-v5-validation-20260410.json
```

April 10, 2026 result on the local RTX 3070 build (`ENV=32`, `PROCESS=64`, `sm_80` and `sm_86` cubins):

- `63/63` supported ROM MD5 checks pass against installed `ale-py` ROMs.
- `63/63` supported minimal action-name checks pass against `ale-py`.
- `63/63` public state roundtrip checks pass on CPU and CUDA, excluding the transient `reward` field that the public state bridge does not encode as restorable state.
- `63/63` restored CPU/CUDA transition checks pass for reward, done, and lives over the deterministic validation window.
- `13/63` restored CPU/CUDA pixel checks still differ, which means the public state bridge is not yet a complete pixel-perfect framebuffer/TIA snapshot for every game.
- Strict black-box CuLE-vs-`ale-py` trajectory parity still fails for `58/63` supported games in the first 16 validation steps. The passing games in that short check are `ALE/CrazyClimber-v5`, `ALE/Freeway-v5`, `ALE/IceHockey-v5`, `ALE/KungFuMaster-v5`, and `ALE/SpaceInvaders-v5`.

Do not treat the supported subset as publication-grade ALE v5 parity yet. The validator is intentionally split so that CPU/CUDA regression gates can pass while the stricter `ale-py` parity gap remains visible.

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

By default the wrapper filters to CuLE kernels with `NCU_KERNEL_NAME='regex:.*cule::atari::cuda::.*'`. Override that environment variable if you intentionally want to include PyTorch helper kernels in the same report.

If `ncu` fails with `ERR_NVGPUCTRPERM`, the current shell does not have permission to access NVIDIA GPU performance counters. On this WSL/WDDM setup, enabling the NVIDIA Control Panel setting for all users fixed Nsight Compute access. NVIDIA documents the permission failure here:

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

- CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#memory-optimizations
- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-bounds
- Ampere Tuning Guide: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- Nsight Compute Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
