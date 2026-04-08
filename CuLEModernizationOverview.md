I went through the parts of your fork that determine whether CuLE will **compile, run, and benchmark well**: the old README/toolchain, `setup.py`, the pybind frontend, the CUDA backend, the Atari wrapper and CUDA dispatch layer, the kernel file, the runtime locator, and the benchmark entrypoints. I did **not** fully audit every vendored header under `third_party/`, so this is a focused audit of the parts that will actually decide whether a modern Ampere port works.     

The main answer is still: **do not begin by rewriting every kernel**. NVIDIA’s Ampere tuning guide explicitly says well-written Volta/Turing-era CUDA code should often run and even speed up on A100 **without code changes**, and the compatibility guide says the first question is whether your binary contains compatible cubins/PTX for Ampere. The first engineering task is therefore **modern build + correct architecture targeting + profiling**, not a blind kernel rewrite. ([NVIDIA Docs][1])

What makes CuLE tricky is that the repo is frozen in a **CUDA 10.0 / Python 3.6 / torch 1.2.0 / torchvision 0.4.0 / atari_py / apex** world, while current PyTorch publishes wheels for modern stacks like **CUDA 11.8, 12.6, 12.8, and 13.0**. PyTorch’s current extension docs also already solve the architecture-selection problem more cleanly than CuLE’s hand-rolled `setup.py`.  ([PyTorch][2])

Here is the concrete checklist I would use.

## 1. First repo changes: fix the build system before touching kernel bodies

### `setup.py`

This is the first file I would change. Right now it manually hacks distutils to dispatch `.cu` files and it derives architecture codes with:

```python
codes = [arch[-2] + '0' for arch in gpus]
```

That is brittle. It can collapse newer arch strings incorrectly, and it is exactly the wrong place to be clever in 2026. 

What I would do:

* replace the custom compiler hook with `torch.utils.cpp_extension.CUDAExtension` + `BuildExtension`
* stop computing arch flags manually
* use `TORCH_CUDA_ARCH_LIST` explicitly for your targets
* start with `8.0 8.6` or `8.0 8.6+PTX`

PyTorch’s docs say CUDA extensions already compile for visible architectures plus PTX by default, and they specifically recommend listing exact CCs when you know them. They also explain that `+PTX` helps forward compatibility but can be modestly slower on newer GPUs than compiling exact CCs directly. ([PyTorch Documentation][3])

For your hardware targets, NVIDIA’s compute-capability table lists:

* **A100 = 8.0**
* **RTX 3070 = 8.6**
* **RTX A5000 = 8.6**
  and NVIDIA’s A4500 product page confirms it is an **Ampere** workstation GPU. ([NVIDIA Developer][4])

So your first sane target is:

```bash
TORCH_CUDA_ARCH_LIST="8.0 8.6"
```

and only after that is stable, consider:

```bash
TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
```

if you want a little forward cushion.

### `examples/utils/runtime.py`

This file manually locates `nvcc` and `libcudart.so` from `CUDA_HOME`/`PATH`. That is okay for a legacy repo, but once you modernize the extension build, this file should become much less central. I would keep it for device-info printing at first, but I would stop relying on it as the core source of truth for how CuLE finds and builds CUDA. 

## 2. Keep the pybind layer mostly intact at first

### `torchcule/frontend.cpp`

This file is not the problem. It is a thin pybind11 wrapper that passes raw pointers into `AtariEnv`. It does **not** look deeply entangled with old THC/ATen extension internals, which is good. The commented-out stream sync hooks are something to revisit later, but they are not your first blocker. 

Initial recommendation:

* keep this file mostly unchanged
* only revisit stream interop after the extension builds and runs

## 3. The first performance hotspot is not the kernel math — it is the launch policy

### `cule/atari/cuda/dispatch.hpp`

This is the most important file to stare at before you even think “Ampere rewrite.”

Right now:

* `reset` launches `initialize_states_kernel` with `BLOCK_SIZE = 1`
* `reset_states` launches `reset_kernel` with `BLOCK_SIZE = 1`
* `step` launches `step_kernel` with `BLOCK_SIZE = 1`
* `preprocess` launches `process_kernel` with `BLOCK_SIZE = 1`
  while `get_data`, `action`, and state-transfer kernels use much more normal sizes like 128 or 256, and frame generation uses 1024. 

That is a massive red flag for modern GPUs. It does **not** automatically mean “wrong,” because Atari stepping is branchy and serialized per environment. But it does mean your first profiling hypothesis should be:

> “CuLE’s biggest Ampere problem may simply be under-occupancy and poor latency hiding from a one-thread-per-block execution strategy in the hottest path.”

NVIDIA’s profiling docs explain exactly why this matters: low achieved occupancy reduces the GPU’s ability to hide latency, and scheduler / warp-state analysis tells you whether warps are stalled on memory, execution dependencies, or barriers. ([NVIDIA Docs][5])

So the **first optimization experiment** I would run is **not rewriting `step_kernel`**. It is:

* parameterize the block size in `dispatch.hpp`
* benchmark `BLOCK_SIZE ∈ {1, 32, 64, 128, 256}` for:

  * `step`
  * `reset_states`
  * `preprocess`

If the kernels already scale materially from just changing launch geometry, that is your first win.

## 4. The kernel bodies to inspect first

### `cule/atari/cuda/kernels.hpp`

This file contains the actual CUDA kernels:

* `initialize_states_kernel`
* `reset_kernel`
* `step_kernel`
* `get_data_kernel`
* `process_kernel`
* `apply_palette_kernel`
* `apply_rescale_kernel`
* `action_kernel`
* `get_states_kernel`
* `set_states_kernel` 

The first kernels I would profile are:

**A. `step_kernel`**
This is the core environment-step path. It is branchy, it loads/stores Atari RAM, it conditionally handles actions and life-reset logic, and it writes TIA updates. If CuLE is slow on Ampere, this kernel is one of the first suspects. 

**B. `process_kernel`**
This is the state-to-frame path before palette/rescale. If frame generation is eating time, this tells you whether the problem is preprocessing or just final image conversion. 

**C. `apply_rescale_kernel`**
This does the 84×84 grayscale rescale path with multiple loads per output pixel. It is a very plausible memory-throughput hotspot. 

**D. `apply_palette_kernel`**
Likely lighter, but still worth checking if full-color rendering is used. 

## 5. What to profile in Nsight Compute

Use Nsight Compute first, not heroic code surgery. NVIDIA’s profiling guide says the sections you care about are:

* **SpeedOfLight**
* **Occupancy**
* **Launch Statistics**
* **SchedulerStats**
* **WarpStateStats**
* sampled **warp stall reasons** ([NVIDIA Docs][5])

What I would look for:

### If `step_kernel` shows:

* **very low achieved occupancy**
* many skipped issue slots
* few eligible warps
* large gaps between theoretical and achieved occupancy

then your first fix is **launch geometry / register pressure**, not a full rewrite. ([NVIDIA Docs][5])

### If it shows:

* heavy **memory dependency stalls**
* poor memory throughput relative to theoretical
* low global-load efficiency

then inspect memory layout and access patterns. NVIDIA’s best-practices guide emphasizes coalesced global memory access as a top priority, and for CC 6.0+ devices, warp accesses coalesce into as few 32-byte transactions as necessary. ([NVIDIA Docs][6])

### If it shows:

* heavy **barrier stalls**
* heavy **branch-resolving stalls**

then the issue is divergence / synchronization structure, not the GPU generation itself. Nsight’s warp-stall docs call these out explicitly. ([NVIDIA Docs][7])

## 6. When a real kernel rewrite is justified

Only rewrite a kernel if profiling proves one of these is true:

1. **The block-size change is not enough** and occupancy remains fundamentally poor.
2. **Memory access is uncoalesced or layout-bound** in a way launch tuning cannot fix.
3. **Warp divergence dominates** because each thread is walking very different control flow.
4. A kernel is hot enough that Ampere-specific features would matter:

   * async copy from global to shared
   * better use of shared memory / L1
   * split arrive/wait barriers
   * warp-level reductions
     Those are exactly the fine-tuning opportunities called out in the Ampere tuning guide. ([NVIDIA Docs][1])

If you do reach that stage, begin with **`step_kernel`**, not everything else.

A likely rewrite path, if needed, would be:

* current model: **one thread ≈ one environment step**
* first real redesign to test: **one warp ≈ one environment** or **one CTA ≈ small batch of environments**

But I would only go there after the cheap experiments fail.

## 7. Files I would leave alone initially

### `torchcule/backend.cu`

This file is mostly orchestration: switching between CPU and GPU execution policies, copying state snapshots, and calling the wrapper/dispatch code. It is important, but it is not the first file I would optimize. I would only revisit it after you know whether the hot kernels are compute-bound, memory-bound, or launch-bound. 

### `cule/atari/wrapper.hpp` and `wrapper.cpp`

These are mostly structural. They define the buffers and route work through `dispatch`. Keep them stable early; they are not your first speed bottleneck.  

## 8. Modern Atari compatibility: decide your benchmark semantics early

CuLE’s examples are built around old-style env names like `PongNoFrameskip-v4` and a legacy Gym/OpenAI-style flow. Current ALE/Gymnasium docs use the `ALE/...-v5` namespace by default, and the semantics differ: `ALE/Pong-v5` uses frameskip 4 and sticky actions by default, while `PongNoFrameskip-v4` has frameskip 1 and no sticky actions. ALE’s current install flow is through `ale-py`/Gymnasium.  ([ale.farama.org][8])

So do this:

* **paper compares against current Atari tooling**
  Then align with current ALE/Gymnasium semantics and say so explicitly.

## 9. The exact docs to read, in order

Read these in this order:

1. **PyTorch C++ extension docs**
   Because your first job is replacing CuLE’s fragile custom build with `CUDAExtension`/`BuildExtension` and explicit arch control. ([PyTorch Documentation][3])

2. **Ampere Compatibility Guide**
   So you understand cubin vs PTX, `CUDA_FORCE_PTX_JIT=1`, and what “compatible” means before tuning. ([NVIDIA Docs][9])

3. **Ampere Tuning Guide**
   To know which Ampere-specific features are worth using later. ([NVIDIA Docs][1])

4. **CUDA C++ Best Practices Guide**
   Especially coalescing, occupancy, latency hiding, and transfer minimization. ([NVIDIA Docs][6])

5. **CUDA Programming Guide**
   For the execution model, streams, launch bounds, memory model, and any advanced CUDA feature you actually decide to use. ([NVIDIA Docs][10])

6. **Nsight Compute Profiling Guide**
   So you know how to interpret the profiler, rather than guessing from numbers. ([NVIDIA Docs][5])

7. **ALE docs**
   For current ROM/env naming and benchmark semantics. ([ale.farama.org][11])

## 10. My recommended development order on your machine

On your 3070 box, I would do this:

### Phase A

* replace `setup.py` build machinery with `CUDAExtension`
* set `TORCH_CUDA_ARCH_LIST="8.6"`
* get one end-to-end build working on current torch

### Phase B

* extend to `TORCH_CUDA_ARCH_LIST="8.0 8.6"`
* confirm the extension still builds cleanly
* run a smoke test on one game

### Phase C

* benchmark current launch sizes
* then only change `dispatch.hpp` block sizes
* profile again

### Phase D

* only then inspect `step_kernel` / `process_kernel` for structural rewrites

That is the fastest path to a working, modern CuLE.

## My blunt summary

For this repo, the first real job is:

**build-system modernization + exact architecture targeting + profiler-driven launch tuning**

—not

**“rewrite all the kernels for Ampere.”**

The single highest-ROI first experiment is in `cule/atari/cuda/dispatch.hpp`: stop assuming `BLOCK_SIZE = 1` is sacred in the `step` / `reset` / `preprocess` path, and let Nsight tell you whether the current hot path is simply starving Ampere of occupancy.  ([NVIDIA Docs][5])

[1]: https://docs.nvidia.com/cuda/archive/12.3.1/ampere-tuning-guide/index.html "NVIDIA Ampere GPU Architecture Tuning Guide"
[2]: https://pytorch.org/get-started/previous-versions "Previous PyTorch Versions"
[3]: https://docs.pytorch.org/docs/stable/cpp_extension.html "torch.utils.cpp_extension — PyTorch 2.11 documentation"
[4]: https://developer.nvidia.com/cuda-gpus "CUDA GPU Compute Capability | NVIDIA Developer"
[5]: https://docs.nvidia.com/nsight-compute/2024.3/ProfilingGuide/index.html?utm_source=chatgpt.com "2. Kernel Profiling Guide — NsightCompute 12.6 documentation"
[6]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/?utm_source=chatgpt.com "CUDA Best Practices Guide — CUDA C++ Best Practices Guide 13.2 documentation"
[7]: https://docs.nvidia.com/nsight-compute/2023.1/ProfilingGuide/index.html?utm_source=chatgpt.com "Kernel Profiling Guide :: Nsight Compute Documentation"
[8]: https://ale.farama.org/environments/pong/?utm_source=chatgpt.com "Pong - ALE Documentation"
[9]: https://docs.nvidia.com/cuda/archive/12.2.1/ampere-compatibility-guide/index.html "NVIDIA Ampere GPU Architecture Compatibility"
[10]: https://docs.nvidia.com/cuda/cuda-programming-guide/?utm_source=chatgpt.com "CUDA Programming Guide — CUDA Programming Guide"
[11]: https://ale.farama.org/main/getting-started/ "Installation - ALE Documentation"