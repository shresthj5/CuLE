# Specification Sheet: Research-Ready CUDA Atari ALE-v5 Compatibility

## 1. Goal

The CUDA Atari backend must be acceptable as a drop-in experimental substitute for **CPU ALE-v5** for the supported game set.

That means:

* same environment semantics,
* same seeded trajectories,
* same observations,
* same rewards,
* same lives and termination behavior,
* same save/restore behavior,
* and materially higher throughput on GPU.

The standard is **behavioral equivalence**, not “close enough.”

If a paper says results were produced on this CUDA backend, another researcher should be able to trust that the run is effectively the same run they would have gotten on CPU ALE-v5, except faster.

---

## 2. Supported target games

The current required game set is:

* `ALE/BattleZone-v5`
* `ALE/DoubleDunk-v5`
* `ALE/NameThisGame-v5`
* `ALE/Phoenix-v5`
* `ALE/Gravitar-v5`
* `ALE/Qbert-v5`
* `ALE/Assault-v5`
* `ALE/MsPacman-v5`
* `ALE/YarsRevenge-v5`

A game is **research-ready** only if it passes every required gate below.

The backend as a whole is **research-ready for the 9-game benchmark** only if **all 9** pass.

---

## 3. Canonical ALE-v5 configuration

Every validation and benchmark run must explicitly specify:

* `frameskip=4`
* `repeat_action_probability=0.25`
* `full_action_space=False`
* grayscale observations for parity tests unless a test explicitly targets RGB
* exact seeded reset behavior

Also require a deterministic validation sub-matrix with:

* `frameskip=1`, `repeat_action_probability=0.0`
* `frameskip=4`, `repeat_action_probability=0.0`
* `frameskip=4`, `repeat_action_probability=0.25`

Reason:

* `fs1_rp0` catches core emulator/reset issues.
* `fs4_rp0` catches action-repeat / public-step / intermediate-frame issues.
* `fs4_rp025` catches full ALE-v5 sticky-action semantics.

---

## 4. Non-negotiable passing standard

For all required equivalence tests, the standard is:

* **exact observation equality**
* **exact reward equality**
* **exact lives equality**
* **exact done / terminal / truncation equality**
* **exact RAM equality where required**
* **exact save/restore continuation equality**
* **exact action-set / metadata equality**

No tolerance. No “small number of pixels.” No “close trajectory.”
If there is any mismatch in a required exact test, that game is **not research-ready**.

---

## 5. Test categories

## A. Metadata and API compatibility

Each game must pass:

### A1. Environment metadata parity

Verify exact match with CPU ALE-v5 for:

* env ID
* default `frameskip`
* default `repeat_action_probability`
* minimal action set length and ordering
* noop action index
* screen dimensions
* grayscale output shape
* lives reporting behavior

### A2. Reset API parity

For a fixed seed:

* CPU ALE-v5 and CUDA backend must both reset successfully
* returned initial observation must match exactly
* legal minimal-action indexing must match exactly

### A3. Step API parity

For a fixed action sequence:

* reward type and values match
* done / truncation semantics match
* info fields needed for research, especially lives, match exactly

A game fails this section if any metadata or API semantics differ.

---

## B. Deterministic reset parity

These tests use `repeat_prob=0.0`.

### B1. `fs1_rp0` reset parity

For seeds `0..31`:

* reset observation must match exactly
* RAM after reset must match exactly
* lives after reset must match exactly
* internal public frame number, if externally inspectable, must match

### B2. `fs4_rp0` reset parity

For seeds `0..31`:

* same requirements as B1

This section exists because some games only diverge after public-step framing is introduced.

A game fails this section if **any** seed mismatches.

---

## C. Early-frame deterministic rollout parity

### C1. `fs1_rp0` short rollout

For seeds `0..31`, use a fixed action script of at least:

* 64 public steps

Check at **every public step**:

* observation exact equality
* reward exact equality
* done exact equality
* lives exact equality

### C2. `fs4_rp0` short rollout

For seeds `0..31`, fixed action script of at least:

* 64 public steps

Same checks as above.

### C3. Action coverage requirement

The action script must not be only NOOP.
It must include:

* NOOP
* FIRE if legal
* directional actions used by the game
* at least one diagonal if legal
* at least one fire-combination if legal

Purpose:
This prevents false confidence from only testing passive trajectories.

---

## D. Sticky-action parity

This is the most important ALE-v5 test.

### D1. Sticky reset parity

For seeds `0..31` under:

* `frameskip=4`
* `repeat_prob=0.25`

Check:

* reset observation exact equality
* RAM exact equality
* lives exact equality

### D2. Sticky rollout parity

For the same seeds, run at least:

* 64 public steps
* fixed requested action script

At each step, require exact equality for:

* observation
* reward
* done
* lives

This test must validate the **actual ALE-v5 sticky semantics**, not a locally convenient approximation.

If sticky reset matches but sticky rollout diverges, fail.
If sticky rollout matches but reset image differs, fail.

---

## E. Save / restore / clone-state fidelity

A research backend must support state checkpointing correctly.

### E1. CPU save -> CUDA restore parity

From CPU ALE-v5:

* reset with seed
* run N steps with fixed actions
* save state
* restore into CUDA backend equivalent state
* continue for M steps with fixed actions

Require exact equality for each continued step:

* observation
* reward
* done
* lives

### E2. CUDA save -> CUDA restore idempotence

For CUDA backend:

* save state
* restore same state
* continue with same action sequence

Require exact continuation equality against the un-restored branch.

### E3. CPU save -> CPU restore baseline

Must also pass on CPU reference harness so the test itself is sane.

### E4. Public-state roundtrip

If the backend exposes public state objects:

* get state
* set state immediately
* require no change in next-step behavior

A game fails this section if restored continuation differs at all.

---

## F. Life-loss semantics

Some Atari evaluations care about lives.

For seeds `0..31`, run trajectories that are likely to trigger life change.

### F1. Raw lives parity

At each step:

* exact life count must match CPU ALE-v5

### F2. Episodic life parity

If episodic-life mode is supported:

* done flag under episodic-life must match CPU behavior exactly
* life-loss step must be identical
* post-life reset semantics must match

---

## G. Terminal and truncation semantics

### G1. True terminal parity

For trajectories reaching game over:

* exact terminal step must match
* final reward must match
* lives at terminal must match

### G2. Time-limit / truncation parity

If max-frame truncation is enabled:

* truncation step must match exactly

---

## H. Observation rendering parity

This is separate from RAM/state parity.

### H1. Grayscale parity

All required tests above must use exact grayscale equality.

### H2. RGB parity

For seeds `0..7`, deterministic rollout of at least 32 steps:

* exact RGB observation equality

### H3. Frame-selection semantics under frame skip

For `frameskip=4`:

* returned observation must match CPU ALE-v5 final public frame semantics exactly
* if the game uses max-pooling behavior in the relevant mode, that must also match exactly
* no hidden “almost identical” buffer parity bugs allowed

This section is especially important because a backend can have exact state and still return the wrong visible frame.

---

## I. Long-horizon drift test

Short tests are not enough.

### I1. Deterministic long rollout

For seeds `0..7`, under `fs4_rp0`:

* run 512 public steps
* fixed action script
* require exact equality at every step

### I2. Sticky long rollout

For seeds `0..7`, under `fs4_rp025`:

* run 512 public steps
* fixed requested action script
* require exact equality at every step

Purpose:
catch rare timing drift like “first mismatch at emulator frame ~128” or later.

Any drift is failure.

---

## J. Multi-env independence

A GPU backend is usually batched; batch coupling is unacceptable.

### J1. Single-env vs batched equivalence

Take the same seed/action script and compare:

* run as `num_envs=1`
* run as part of `num_envs=32` or `64`

The target env’s trajectory must be identical in both cases.

### J2. Cross-env independence

Changing neighboring env seeds/actions must not change a target env’s trajectory.

This catches shared RNG state, buffer aliasing, and indexing bugs.

---

## K. CPU vs CUDA backend parity inside the same codebase

If the codebase has both CPU and CUDA execution paths:

### K1. Local CPU backend vs reference ALE CPU

Must match exactly.

### K2. Local CUDA backend vs reference ALE CPU

Must match exactly.

### K3. Local CPU vs local CUDA

Must also match exactly.

This helps isolate whether a failure is:

* semantic vs ALE,
* or only CPU/CUDA internal divergence.

---

## L. Performance qualification

A game is not useful as a CUDA benchmark if it is exact but slow.

Performance is secondary to fidelity, but still required.

### L1. Benchmark conditions

Benchmark with:

* `num_envs=4096`
* `frameskip=4`
* `repeat_prob=0.25`
* random legal actions
* warmup before timing
* report env-steps/hour and emulator-frames/hour

### L2. Minimum throughput gate

Set a repository-specific minimum acceptable throughput on the target GPU.
For an RTX 3070-class GPU, a reasonable bar for these games is:

* no supported research-ready game should fall below a clearly documented floor established by your known-good baselines on that hardware

Do not hardcode a universal number for all future hardware.
Instead, define:

* hardware
* driver
* CUDA version
* block sizes
* exact command line
* measured throughput

### L3. Performance non-regression rule

No change may reduce average throughput for the 9-game suite by more than:

* **5%** unless it fixes a documented fidelity blocker
* **10%** only with explicit justification and benchmark note

### L4. Hot-path purity rule

Reset-only fixes may be slower.
Step-path fixes must not introduce unnecessary host/device syncs, per-subframe Python allocations, or repeated tiny transfers in the hot loop.

---

## 6. Required artifacts for every game

For each game, maintain a validation record containing:

* pass/fail for each category A–L
* failing seed list, if any
* first mismatching public step, if any
* first mismatching emulator frame, if known
* mismatch type:

  * observation
  * RAM
  * reward
  * done
  * lives
  * save/restore
  * sticky reset
  * sticky rollout
* exact command used
* git commit SHA
* GPU model / driver / CUDA version
* benchmark throughput

No game is “research-ready” without a complete validation record.

---

## 7. Required pass matrix

A game is **research-ready** only if it passes all of the following:

* A1–A3 metadata/API parity
* B1–B2 deterministic reset parity
* C1–C3 deterministic short rollout parity
* D1–D2 sticky reset + rollout parity
* E1–E4 save/restore fidelity
* F1–F2 lives semantics
* G1–G2 terminal/truncation parity
* H1–H3 observation rendering parity
* I1–I2 long-horizon drift tests
* J1–J2 multi-env independence
* K1–K3 local CPU/CUDA/reference parity
* L1–L4 performance qualification

Anything less is **not research-ready**.

---

## 8. What does not count as passable

The following are automatic failures:

* “Only a few pixels differ”
* “Only one seed fails”
* “RAM matches and rollout matches, but reset image is slightly off”
* “Only sticky reset fails”
* “Only save/restore fails”
* “Works in single-env but not batch”
* “Works on CPU path but not CUDA path”
* “Exact but much slower due to host-driven hot path”
* “Passes our custom harness but not CPU ALE-v5”

If another researcher could reproduce a different trajectory or different initial observation using CPU ALE-v5, the backend is not ready.

---

## 9. Release rule for the 9-game benchmark

The benchmark may be advertised to other researchers only when:

* all 9 games pass all required gates,
* all results are tied to a specific commit SHA,
* performance numbers are documented on named hardware,
* and the validation harness is committed and runnable by others.

---

## 10. Recommended test execution order

For a new agent working a failing game:

1. metadata parity
2. deterministic reset parity (`fs1_rp0`)
3. deterministic rollout parity (`fs1_rp0`)
4. deterministic public-step parity (`fs4_rp0`)
5. sticky reset parity (`fs4_rp025`)
6. sticky rollout parity
7. save/restore
8. long-horizon drift
9. multi-env independence
10. performance benchmark

That order minimizes wasted work.

---

## 11. One-sentence acceptance rule

**A game is research-ready only if a seeded ALE-v5 run on the CUDA backend is behaviorally indistinguishable from the same seeded run on CPU ALE-v5 across reset, rollout, sticky actions, save/restore, batching, and performance qualification.**


What must match is the **observable ALE-v5 behavior**, not the internal implementation path.

So the CUDA backend does **not** need to:

* use the same call structure,
* use the same buffering scheme,
* use the same control flow,
* or mirror CPU ALE line by line.

It **does** need to produce the same results for the same seeded run:

* same reset observation,
* same rollout observations,
* same rewards,
* same lives,
* same done/truncation behavior,
* same sticky-action semantics,
* same save/restore continuation.

That’s the right standard for a research backend:
**semantic equivalence, not implementation equivalence.**

The whole point of CUDA is that you can reorganize execution to be much more efficient:

* fuse loops,
* keep data on device,
* batch work,
* use different kernel structure,
* avoid host round-trips,

as long as the final behavior matches CPU ALE-v5 exactly where researchers care.

So when I say “apples to apples,” I mean:

**same experiment, same results, faster execution**

not

**same code path, same internal mechanics**

That’s why some changes belong in `cule/` for correctness, while others belong in `torchcule/` or CUDA kernels for speed. You are free to implement the backend in a much more GPU-efficient way, provided the outputs stay exact.

A good mental model is:

* **CPU ALE is the reference specification**
* **CUDA CuLE is the optimized implementation**
