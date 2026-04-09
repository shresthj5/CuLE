# CuLE Modernization Blueprint

Date: 2026-04-08

Purpose: capture the current state of the repo, the environment audit, the technical deductions from source review and documentation, and the concrete roadmap for getting CuLE compiling, running correctly, and performing well on modern Ampere-class NVIDIA GPUs such as the RTX 3070 and A100.

This document is intentionally deeper and more operational than `CuLEModernizationOverview.md`. The overview is a concise audit. This file is the working blueprint.

## Executive Summary

CuLE should be treated as a modernization and hardening project, not as a blind Ampere rewrite.

The biggest obstacles are:

1. Legacy build and packaging machinery.
2. Python-side dependency drift (`atari_py`, old Gym naming, old PyTorch assumptions).
3. A few likely correctness hazards in the extension boundary and state transfer path.
4. Very old launch geometry assumptions in the hot CUDA path.

The highest-value first steps are:

1. Rebuild the project around a modern PyTorch CUDA extension flow.
2. Define the compatibility target explicitly: Ampere `sm_80` + `sm_86`.
3. Fix correctness-risk items before tuning for performance.
4. Establish a correctness and benchmark harness.
5. Use Nsight-driven launch tuning before attempting structural kernel rewrites.

The project is very likely salvageable on modern hardware, but it is not currently in a runnable state in the active development environment.

## Scope

### Primary goals

- Make CuLE build on a modern Linux/WSL development environment.
- Support Ampere GPUs cleanly, especially:
  - NVIDIA A100 (`sm_80`)
  - GeForce RTX 3070 (`sm_86`)
  - Other Ampere workstation/consumer GPUs that also map to `sm_86`
- Preserve the core CuLE design: large-scale GPU Atari emulation with GPU-side frame generation.
- Ensure that modernizing the build does not silently break the emulator semantics.
- Establish a repeatable profiling workflow and improve hot-path performance where it matters.

### Secondary goals

- Modernize examples enough to run smoke tests and benchmarking.
- Clean up outdated docs, Docker guidance, and developer bootstrap instructions.
- Make the project easier to keep alive after the first modernization pass.

### Non-goals for the first pass

- Rewriting every CUDA kernel "for Ampere."
- Replacing CuLE with ALE's modern vector environment.
- Preserving support for every historical GPU architecture in the same build lane.
- Redesigning the entire Python training stack before the emulator core is stable.

## Current Environment Audit

This section records what was observed in the local machine and shell used for the audit.

### Local machine and toolchain

Observed locally:

- GPU: NVIDIA GeForce RTX 3070
- Driver version: `580.88`
- `nvidia-smi` CUDA runtime version: `13.0`
- `nvcc`: CUDA compilation tools `12.8.93`
- `nvcc` path: `/usr/local/cuda/bin/nvcc`
- GCC: `13.3.0`
- Binutils `ld`: `2.42`
- Python on PATH: `python3 3.12.3`
- Shell environment: WSL2 Linux on a Windows host

Useful inference:

- This is a modern-enough Ampere development machine.
- The local toolkit supports Ampere compilation targets, including `compute_80` and `compute_86`.
- The local driver is newer than the local toolkit, which is normal and acceptable.

### Playwright tooling

The repo-specific `PlaywrightAccess.md` instructions were valid.

What was discovered:

- `node`, `npm`, and `npx` were initially missing from the Codex shell PATH.
- `~/.nvm/nvm.sh` exists locally.
- After sourcing `~/.nvm/nvm.sh`, the shell gained:
  - `node v24.14.1`
  - `npm 11.11.0`
  - `npx 11.11.0`
- The installed Codex Playwright wrapper then worked.

Implication:

- Playwright is available in this environment, but only after loading `nvm`.
- Any future browser-assisted research or UI verification in this repo should source `~/.nvm/nvm.sh` first.

### Missing project runtime pieces in the active Python environment

The active Python environment is effectively empty relative to CuLE's needs.

Missing at audit time:

- `torch`
- `Cython`
- `numpy`
- `gym`
- `gymnasium`
- `atari_py`
- `ale_py`
- `pybind11`

Other missing tools:

- `cmake`
- `ninja`
- `python` alias itself was missing, while `python3` exists

Implication:

- The project is not currently in a "make sure everything works" state.
- Any build or import failures right now are polluted by bootstrap issues, not just source incompatibility.

### Repo checkout state

Observed:

- Untracked `.playwright-cli/`
- Untracked `PlaywrightAccess.md`
- `third_party/agency` and `third_party/pybind11` are declared as submodules
- Submodule status shows leading `-`, meaning they are not initialized
- The directories exist locally but are effectively empty from a build perspective

Implication:

- The repo cannot be expected to compile until submodules are initialized.

## Repo Structure and Critical Files

The repo is compact enough that most modernization risk concentrates in a small set of files.

### Core files that determine whether CuLE builds and runs

- `setup.py`
- `examples/utils/runtime.py`
- `torchcule/frontend.cpp`
- `torchcule/backend.cu`
- `torchcule/atari/env.py`
- `torchcule/atari/rom.py`
- `cule/atari/wrapper.cpp`
- `cule/atari/wrapper.hpp`
- `cule/atari/cuda/dispatch.hpp`
- `cule/atari/cuda/kernels.hpp`
- `cule/cuda/parallel_execution_policy.hpp`

### Example and benchmark files that determine how the project is exercised

- `README.md`
- `envs/Dockerfile`
- `envs/environment.yml`
- `examples/utils/launcher.py`
- `examples/utils/initializers.py`
- `examples/vtrace/benchmark_vtrace.py`
- `examples/dqn/benchmark.config`
- `examples/ppo/benchmark.config`

## Confirmed Discoveries

This section contains findings that came directly from source inspection or environment inspection.

### 1. The build system is legacy and brittle

Confirmed in `setup.py`:

- It imports `torch` at import time.
- It derives architecture codes using:

```python
codes = [arch[-2] + '0' for arch in gpus]
```

- It manually patches distutils to send `.cu` files through `nvcc`.
- It manually locates CUDA via `examples/utils/runtime.py`.
- It relies on vendored pybind11 and agency include trees in `third_party`.

Why this matters:

- This architecture detection logic is too clever and too fragile for a 2026 modernization effort.
- It duplicates functionality that modern PyTorch extension tooling already solves more cleanly.
- It is tightly coupled to an environment where `torch` already imports cleanly before build bootstrap begins.

### 2. The repo documents at least two different eras of dependency expectations

The root `README.md` still advertises:

- CUDA 10.0
- incompatibility with CUDA 10.1
- `torch==1.2.0`
- `torchvision==0.4.0`
- `atari_py`
- Apex

The newer `envs/Dockerfile` instead advertises:

- CUDA 11.7.1
- Ubuntu 22.04
- Python 3.9
- `torch==1.11.0`
- `torchvision==0.12.0`
- `gym[atari]`

Implication:

- The repo has already drifted beyond its original README, but the migration was incomplete.
- Documentation and build expectations are inconsistent inside the repo itself.

### 3. The Python front-end still depends on legacy Atari tooling

Confirmed in `torchcule/atari/rom.py` and related example code:

- ROM lookup depends on `atari_py`.
- Environment names assume legacy forms such as `PongNoFrameskip-v4`.
- Example benchmarking code enumerates old Gym registry names matching `*NoFrameskip-v4`.

Implication:

- Even if the extension compiles, the user-facing Python environment surface is still coupled to an obsolete Atari stack.
- A modern build alone is not enough; the Python API compatibility layer also needs work.

### 4. The Python `Env` surface stores modern-seeming options but does not fully implement them

Confirmed in `torchcule/atari/env.py`:

- `frameskip` is used.
- `episodic_life` is used.
- `repeat_prob` is stored but not wired into emulator stepping logic.
- `max_episode_length` is accepted but not enforced by the core environment itself.

Implication:

- Some API parameters suggest ALE/Gym compatibility but are only partially implemented.
- Benchmark semantics must be declared explicitly rather than assumed.

### 5. The pybind stream sync methods are currently no-ops

Confirmed in `torchcule/frontend.cpp`:

- `sync_other_stream` and `sync_this_stream` are exposed to Python.
- The actual ATen stream plumbing is commented out.
- The Python `Env` methods still call these functions as if they are meaningful.

Implication:

- Stream synchronization semantics are currently suspect.
- This could produce hidden correctness bugs or accidental reliance on default stream behavior.
- This is a correctness issue first, not a performance optimization issue.

### 6. The CUDA backend orchestration is not the first performance problem, but it has some cleanup debt

Confirmed in `torchcule/backend.cu`:

- It acts mostly as an execution-policy switchboard between CPU and CUDA paths.
- It owns a `void*` policy pointer with manual `delete &get_policy<...>()` style lifetime management.
- It includes implementation `.cpp` files directly into the `.cu` translation unit.

Implication:

- It is not the first file to optimize for Ampere speed.
- It is still worth cleaning up later for maintainability and possibly compile robustness.

### 7. The hot CUDA path hardcodes `BLOCK_SIZE = 1` in multiple critical kernels

Confirmed in `cule/atari/cuda/dispatch.hpp`:

- `reset` launches `initialize_states_kernel` with `BLOCK_SIZE = 1`
- `reset_states` launches `reset_kernel` with `BLOCK_SIZE = 1`
- `step` launches `step_kernel` with `BLOCK_SIZE = 1`
- `preprocess` launches `process_kernel` with `BLOCK_SIZE = 1`

By contrast:

- `get_data` uses `256`
- `generate_random_actions` uses `256`
- `get_states` and `set_states` use `128`
- frame generation uses `1024`

Implication:

- The emulator hot path may be starving modern GPUs of occupancy.
- This is the single highest-ROI performance hypothesis in the repo.

### 8. The kernel templates couple launch size to memory layout behavior

Confirmed in `cule/atari/cuda/kernels.hpp`:

- `NT` feeds into `__launch_bounds__(NT)`.
- `NT` also appears in the RAM addressing formulas:

```cpp
((uint32_t*) ram_buffer) + (NUM_INT_REGS * NT * blockIdx.x) + threadIdx.x
```

with strided loads/stores of:

```cpp
ram_int[i * NT]
```

Implication:

- `NT` tuning is not just occupancy tuning.
- It also changes register pressure behavior and the degree of memory coalescing/interleaving across threads.
- Launch tuning needs to be treated as a real design parameter, not a cosmetic constant swap.

### 9. The most obvious performance kernels are exactly where expected

From static inspection, the main hot kernels are:

- `step_kernel`
- `process_kernel`
- `apply_rescale_kernel`
- `apply_palette_kernel`

Likely roles:

- `step_kernel`: branch-heavy emulator step path
- `process_kernel`: TIA/frame preprocessing path
- `apply_rescale_kernel`: grayscale resize path with multiple loads per output pixel
- `apply_palette_kernel`: cheaper frame-format conversion path

### 10. The state transfer path likely contains an indexing bug

In `get_states_kernel`:

- selected env index comes from `indices[global_index]`
- state struct fetch uses that selected index
- frame_state fetch uses that selected index
- but RAM input offset is advanced using `global_index`
- and output packed RAM offset is advanced using the selected env index

That appears backwards.

In `set_states_kernel`:

- the same pattern appears in reverse

This needs runtime confirmation, but the static reading strongly suggests that the RAM slice copy path is inconsistent with the state and frame-state indexing semantics.

Implication:

- `get_states` / `set_states` on CUDA may already be wrong for nontrivial index selections.
- This should be treated as a correctness blocker before any serious benchmarking.

### 11. The legacy profiling hook in examples is obsolete

Confirmed in `examples/utils/launcher.py`:

- profiling mode relies on legacy Nsight Systems injection libraries pinned to CUDA 10.0-era names

Implication:

- Modern profiling should use current Nsight Systems and Nsight Compute command-line flows, not this injection path.

## Strong Deductions and Working Hypotheses

These are not all proven by runtime yet, but they are the best current technical conclusions.

### A. CuLE probably does not need a full kernel rewrite to run on Ampere

Reasoning:

- NVIDIA's Ampere tuning guidance says well-written Volta/Turing-era CUDA code often runs and may speed up on A100 without code changes.
- The first compatibility question is whether the binary contains usable cubin/PTX for the target GPU.
- CuLE's biggest visible problem is the build and launch policy, not an obvious use of forbidden or removed CUDA features inside every kernel.

Working conclusion:

- First fix build compatibility and architecture targeting.
- Then profile.
- Rewrite only the kernels that profiling proves need structural work.

### B. The first big performance win is more likely to come from launch geometry than from algorithmic redesign

Reasoning:

- The hottest path uses one thread per block in multiple places.
- Ampere still benefits from the usual CUDA fundamentals: occupancy, latency hiding, coalescing, reduced divergence where possible.

Working conclusion:

- Parameterizing block size for `step`, `reset_states`, and `preprocess` is the first serious optimization experiment.

### C. CuLE needs a compatibility policy decision before example modernization starts

There are at least two plausible compatibility modes:

1. Legacy benchmark mode
   - Keep `NoFrameskip-v4` semantics
   - Keep frameskip and no-stickiness conventions aligned with historical papers

2. Modern ALE mode
   - Use `ALE/...-v5`
   - Accept modern sticky-action semantics and current API structure

Working conclusion:

- Both modes should probably exist.
- The core emulator modernization should not silently swap one for the other.

### D. Modernizing to CUDA 13-only is not necessary and may be strategically wrong

Important current ecosystem fact:

- CUDA 13.0 removed offline compilation support for Maxwell, Pascal, and Volta.

For this project:

- That is not a blocker for Ampere support.
- But it means a CUDA 13-only build lane would intentionally narrow the historical GPU support envelope.

Working conclusion:

- For the main modernization branch, CUDA 12.8 is a strong base choice on this machine.
- Driver 13.0 is fine.
- A CUDA 12.x build lane is the safer modernization baseline unless there is a strong reason to require toolkit 13.0 features.

## Recommended Technical Baseline

This is the baseline I would target first unless new constraints appear.

### Primary development baseline

- Python: `3.12` first choice on this machine
- If legacy example dependencies fight too hard: fall back to project-local Python `3.11`
- PyTorch: modern wheel line with CUDA 12.8 compatibility
- CUDA toolkit for extension compilation: `12.8`
- GPU targets: `8.0 8.6`
- Build mechanism: `torch.utils.cpp_extension.CUDAExtension` + `BuildExtension`

### Recommended architecture list

Primary:

```bash
TORCH_CUDA_ARCH_LIST="8.0 8.6"
```

Optional forward-compatibility variant:

```bash
TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

Reasoning:

- `8.0` covers A100
- `8.6` covers RTX 3070 and most Ampere workstation/consumer targets of interest
- Explicit archs are preferable to relying only on PTX when exact targets are known

## Roadmap

This roadmap is ordered by dependency and risk.

### Phase 0: Freeze Goals and Compatibility Policy

Deliverables:

- Decide whether the default Python API should present legacy Atari semantics or modern ALE semantics.
- Define the primary support matrix:
  - RTX 3070 local dev target
  - A100 deployment target
  - Optional "other Ampere" target via `sm_86`
- Decide whether pre-Ampere support is in scope for the first pass.

Planned outputs:

- Compatibility policy section in docs
- Target matrix section in docs

### Phase 1: Bootstrap a Clean Development Environment

Tasks:

- Initialize submodules:

```bash
git submodule update --init --recursive
```

- Create a project-local virtual environment or conda environment.
- Install baseline tooling:
  - `pip`
  - `setuptools`
  - `wheel`
  - `ninja`
  - `cython`
  - `numpy`
- Install a modern PyTorch wheel matching the local CUDA build strategy.
- Install modern Atari-side runtime pieces:
  - `gymnasium`
  - `ale-py`
  - supporting example dependencies such as `opencv-python`, `psutil`, `tqdm`, `cloudpickle`

Preferred starting commands on this machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja cython numpy
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install gymnasium ale-py opencv-python psutil tqdm cloudpickle
git submodule update --init --recursive
```

Acceptance criteria:

- `import torch`
- `import numpy`
- `import gymnasium`
- `import ale_py`
- `python -c "import torch; print(torch.cuda.is_available())"`

### Phase 2: Replace the Legacy Build Path

Tasks:

- Replace custom distutils compiler patching in `setup.py`.
- Migrate to:
  - `torch.utils.cpp_extension.CUDAExtension`
  - `torch.utils.cpp_extension.BuildExtension`
- Remove hand-rolled architecture code derivation.
- Honor `TORCH_CUDA_ARCH_LIST`.
- Keep source set minimal at first.
- Retain current include layout while submodules are still vendored.

Secondary cleanup in this phase:

- Stop relying on `examples/utils/runtime.py` as the primary CUDA build locator.
- Add a light-weight build README for the modern path.

Acceptance criteria:

- Clean extension build on RTX 3070 machine.
- Produced module imports successfully.
- Build works with `TORCH_CUDA_ARCH_LIST="8.6"` locally.
- Build works with `TORCH_CUDA_ARCH_LIST="8.0 8.6"` without source changes.

### Phase 3: Repair the Python and Extension Boundary

Tasks:

- Replace or wrap `atari_py` dependency in `torchcule/atari/rom.py`.
- Add environment-name normalization:
  - accept legacy `PongNoFrameskip-v4`
  - optionally accept modern `ALE/Pong-v5`
- Decide whether to:
  - preserve legacy ROM lookup behavior
  - or adopt `ale-py` as the canonical runtime dependency and provide compatibility mapping
- Repair stream synchronization hooks in `torchcule/frontend.cpp`.
- Validate `Env.to()`, `Env.reset()`, and `Env.step()` stream semantics on CUDA.

Acceptance criteria:

- Python API can resolve a ROM without `atari_py`.
- CUDA calls no longer rely on no-op stream hooks.
- Extension import and environment construction work on CPU and GPU.

### Phase 4: Correctness Hardening Before Performance Tuning

Tasks:

- Add focused smoke tests for:
  - extension import
  - CPU environment reset/step
  - GPU environment reset/step
  - frame generation
  - `get_states` / `set_states` roundtrip
  - deterministic seed behavior where expected
- Investigate and fix the likely RAM indexing bug in:
  - `get_states_kernel`
  - `set_states_kernel`
- Validate the "last frame" and max-over-two-frames observation path.
- Confirm reward, lives, terminal, and started semantics for a small ROM set.
- Use `compute-sanitizer` and/or `CUDA_LAUNCH_BLOCKING=1` on focused tests if needed.

Suggested ROM subset for early validation:

- Pong
- Breakout
- SpaceInvaders
- Seaquest

Acceptance criteria:

- CPU and GPU paths agree on basic state transitions under controlled seeds.
- `get_states` / `set_states` passes an explicit roundtrip test.
- No obvious memory errors in sanitizer-assisted smoke tests.

### Phase 5: Establish a Modern Profiling Workflow

Tasks:

- Ignore the old injected profiling code in `examples/utils/launcher.py`.
- Use modern tools directly:
  - Nsight Compute for kernel profiling
  - Nsight Systems for end-to-end timeline tracing
- Create a small benchmark harness that isolates:
  - env reset
  - env stepping
  - frame generation
  - end-to-end frames/sec across multiple `num_envs`
- Benchmark at:
  - `16`
  - `64`
  - `256`
  - `1024`
  - `2048`
  - `4096`

Metrics to capture:

- steps/sec
- frames/sec
- GPU utilization
- achieved occupancy
- launch statistics
- scheduler stats
- warp stall reasons

Acceptance criteria:

- Reproducible baseline report for RTX 3070.
- Reproducible baseline procedure that can be run later on A100.

### Phase 6: Low-Risk Performance Tuning

Tasks:

- Parameterize block sizes in `cule/atari/cuda/dispatch.hpp`.
- Sweep:

```text
BLOCK_SIZE in {1, 32, 64, 128, 256}
```

for:

- `step`
- `reset_states`
- `preprocess`

Interpret profiler results before changing kernel structure.

What to look for:

- achieved occupancy gaps
- low eligible warps
- memory dependency stalls
- branch-resolving stalls
- barrier stalls
- global memory efficiency

Acceptance criteria:

- A measured improvement on the RTX 3070 baseline without breaking correctness.
- A clear data-backed recommendation for A100 tuning.

### Phase 7: Structural Performance Work Only If Profiling Demands It

Only enter this phase if Phase 6 does not close the gap.

Possible work items:

- Restructure `step_kernel` if occupancy/divergence remains fundamentally poor.
- Rework memory layout if coalescing is still poor after launch tuning.
- Revisit `apply_rescale_kernel` if it dominates end-to-end time.
- Evaluate whether one warp per environment or one CTA per small environment batch is superior.
- Consider selective Ampere-aware techniques only where profiling justifies them:
  - async copy to shared memory
  - shared-memory staging
  - warp-level reduction primitives
  - shared/L1 carveout tuning

Important note:

- Ampere-specific features should be the end of the tuning process, not the start.

### Phase 8: Packaging, Docs, and Reproducibility

Tasks:

- Rewrite the root build instructions.
- Add a modern environment bootstrap guide.
- Replace contradictory CUDA 10.0 guidance.
- Add a clean Docker recipe for the modern branch.
- Add a benchmark document explaining:
  - legacy vs modern ALE semantics
  - local RTX 3070 results
  - A100 results when available

Acceptance criteria:

- A new contributor can build the project from scratch with one documented path.
- The build instructions match the actual code.

## Detailed Work Backlog

This is the practical backlog I would execute against.

### Build system

- [ ] Initialize `third_party` submodules.
- [ ] Replace custom NVCC distutils hook.
- [ ] Adopt `CUDAExtension` + `BuildExtension`.
- [ ] Make architecture targeting explicit and testable.
- [ ] Add `ninja` to documented prerequisites.

### Python API and dependencies

- [ ] Remove hard dependency on `atari_py`.
- [ ] Add `ale-py` / Gymnasium integration path.
- [ ] Add env-name normalization for legacy and modern naming.
- [ ] Audit unused or partially implemented API arguments.

### Correctness

- [ ] Repair pybind stream sync bridging.
- [ ] Confirm and fix `get_states_kernel` RAM indexing.
- [ ] Confirm and fix `set_states_kernel` RAM indexing.
- [ ] Add CPU/GPU parity smoke tests.
- [ ] Add state roundtrip tests.

### Performance

- [ ] Add benchmark harness for isolated env operations.
- [ ] Sweep block sizes on `step`, `reset_states`, `preprocess`.
- [ ] Profile `step_kernel`.
- [ ] Profile `process_kernel`.
- [ ] Profile `apply_rescale_kernel`.
- [ ] Only after profiling, decide whether to redesign kernel structure.

### Docs and dev UX

- [ ] Rewrite README compatibility and build instructions.
- [ ] Add modern Docker path.
- [ ] Add benchmark methodology document.
- [ ] Add explicit target support table.

## Validation Matrix

This is the test and validation matrix that should exist by the end of the modernization effort.

### Build matrix

- Linux / WSL2
- Python 3.12
- CUDA toolkit 12.8
- PyTorch CUDA extension build
- `TORCH_CUDA_ARCH_LIST="8.6"`
- `TORCH_CUDA_ARCH_LIST="8.0 8.6"`

### Device matrix

- RTX 3070 local verification
- A100 remote or deployment verification

### Functional matrix

- CPU env reset
- CPU env step
- GPU env reset
- GPU env step
- frame generation RGB
- frame generation grayscale + resize
- state export/import
- episodic life behavior
- fire reset behavior

### Performance matrix

- 16 envs
- 64 envs
- 256 envs
- 1024 envs
- 2048 envs
- 4096 envs

### Regression checks

- no kernel launch failures
- no invalid memory access in sanitizer-assisted smoke tests
- no obvious stream-ordering mistakes
- no silent shift in benchmark semantics without documentation

## Performance Expectations and Guardrails

### What "good" looks like

- CuLE builds cleanly with modern toolchains.
- CuLE runs correctly on the RTX 3070 and A100.
- Performance scales with environment count in a sensible way.
- There is no obvious occupancy starvation caused by accidental one-thread-per-block hot paths.
- Any remaining bottlenecks are understood and documented.

### What not to do

- Do not assume poor performance means "Ampere needs new kernels."
- Do not silently switch legacy benchmark semantics to modern ALE defaults.
- Do not change launch geometry and memory layout at the same time without profiling in between.
- Do not trust the examples as proof of correctness until the emulator core has dedicated tests.

## Risks and Unknowns

### High-risk items

- Stream semantics are currently ambiguous because the Python layer calls no-op sync hooks.
- `get_states` / `set_states` CUDA indexing may be wrong.
- The old Python Atari stack is obsolete and may fight modern Python.

### Medium-risk items

- `agency` may introduce modern compiler friction once submodules are initialized and real compilation begins.
- The monolithic compile structure in `backend.cu` may be painful for build times and diagnostics.
- Example training code may need more modernization than the emulator core.

### Lower-risk items

- The core pybind wrapper is relatively thin and probably not the major blocker.
- The wrapper/orchestration layer is likely serviceable once the build and stream issues are corrected.

## What I Plan to Fix First

The first concrete work order should be:

1. Initialize submodules and build a clean Python environment.
2. Replace the legacy build system with a modern PyTorch CUDA extension path.
3. Repair stream-interop correctness and investigate the state-transfer indexing bug.
4. Modernize ROM/env-name compatibility away from `atari_py` lock-in.
5. Add smoke tests and a minimal benchmark harness.
6. Profile and tune launch geometry.

This order is intentional:

- It removes bootstrap noise first.
- It fixes correctness before performance.
- It gathers profiler evidence before kernel surgery.

## Reference Notes from Official Documentation

These references directly shaped the roadmap:

- PyTorch `torch.utils.cpp_extension` docs:
  - `CUDAExtension` and `BuildExtension` are the modern mixed C++/CUDA build path.
  - `TORCH_CUDA_ARCH_LIST` should be used instead of hand-rolled arch detection.
  - Explicit arch targets are better than relying only on PTX when the desired GPUs are known.

- NVIDIA CUDA GPU compute capability table:
  - A100 is compute capability `8.0`
  - RTX 3070 is compute capability `8.6`

- NVIDIA Ampere compatibility guidance:
  - Existing applications can work on Ampere if they include compatible cubins or PTX.
  - PTX should be included for forward compatibility.
  - Native cubins for exact targets are still preferable for startup and performance.

- NVIDIA Ampere tuning guidance:
  - Start with standard CUDA best practices.
  - Good Volta/Turing-era code often works and can speed up on Ampere without source rewrites.
  - Occupancy, coalescing, and latency hiding remain primary concerns.

- NVIDIA CUDA best practices:
  - Coalesced memory access is still one of the highest-priority concerns.
  - On compute capability `6.0+`, warp accesses coalesce into the minimum number of needed 32-byte transactions.

- Nsight Compute profiling guidance:
  - stall reasons and scheduler metrics are the correct way to identify whether a kernel is blocked by memory, divergence, barriers, or occupancy limitations.

- ALE/Gymnasium current docs:
  - `ale-py` is the supported modern Python interface.
  - `ALE/Pong-v5` is not semantically identical to `PongNoFrameskip-v4`.
  - ALE now also has its own vector environment implementation, which is useful as a comparison point but not as a replacement for CuLE.

## Final Blueprint Statement

The modernization target is not "make the old code compile somehow."

The real target is:

- a modern, reproducible build,
- explicit Ampere architecture support,
- repaired correctness boundaries,
- clear benchmark semantics,
- and profiler-backed performance tuning.

If that sequence is followed, CuLE has a realistic path to running well on both the RTX 3070 and A100 without unnecessary kernel rewrites.
