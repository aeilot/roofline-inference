---
name: roofline-engine-study
description: >-
  End-to-end workflow for studying and improving LLM inference on a specific
  device and model: calibrate the hardware, build roofline-style resource
  reference lines from the model graph, measure and decompose the gap of
  existing engines (mlx-lm, vLLM, llama.cpp), let an agent optimize an engine
  under explicit math-equivalence rules, and verify any speedup with a
  pre-registered, process-isolated, thermally and power gated paired benchmark.
  Use this skill whenever the user wants to port the roofline-inference study
  to another device, GPU, Mac, model or engine; asks how fast inference "could"
  be on some hardware; wants an inference engine faster than mlx-lm / vLLM /
  llama.cpp; wants to benchmark or compare inference engines rigorously;
  or asks whether a claimed speedup is real — even if they do not say
  "roofline". 中文触发：迁移到其他设备/模型、推理上界、性能参考线、差距分解、
  优化推理引擎、引擎对比验证、预注册基准测试、证明比 mlx 快。
---

# Roofline engine study

This skill packages the workflow developed in the roofline-inference repository
(Qwen3-0.6B on a MacBook Air M4, 2026-10). It has six phases. Each phase produces
an artifact the next one needs, and each has traps that cost real time the first
time around — they are summarized here and detailed in `references/`.

The reference implementation lives in this repository:

| Phase | Reference code |
|---|---|
| 1 Calibrate device | `benchmarks/{bandwidth,compute,hierarchy,concurrency,kernel_chain,compute_shapes,contention}.py`, `scripts/{calibrate,fit_kernel_chain}.py`, `configs/hardware/*.yaml` |
| 2 Reference lines | `src/roofline/` (`python -m roofline ...`), `configs/models/*.yaml` |
| 3 Baselines and gap | `benchmarks/{baseline_mlx,ablation_mlx}.py`, `scripts/gap_report.py` |
| 4 Engine optimization | `agent-test/v2/` (engine, kernels, `check.py` FP32 gate, tests) |
| 5 Verification | `agent-test/v3/` (`run.py`, `worker.py`, `adapters.py`, `gate.py`, `divergence.py`, `analyze.py`, `CONFIRM.md`) |
| 6 Reporting | `docs/report.md`, `agent-test/v3/results-confirm-*/report-corrected.md` |

When porting, copy and adapt these rather than rewriting from scratch; `references/porting.md`
lists exactly which parts are device-, model- or engine-specific.

## Before starting: fix the scope with the user

Agree on these and write them down (they become the header of every report):

- **Device** (exact model, memory, power source), **model** (exact checkpoint, dtype), **engines** to compare.
- **Relaxation level.** A = only change the implementation (same math; summation order, fusion,
  recomputation and dead-code removal allowed). B = precision (quantized weights/KV). C = algorithm
  (speculative decoding, KV compression, sparsity). Each level needs its own quality budget; level A needs
  only a numerical-equivalence check. Start at A unless the user says otherwise — it is the only level
  where "faster" does not need a quality argument.
- **Scenarios and metrics**: TTFT(prompt length), decode time per token at a context, whole-request time;
  batch size; greedy or sampling.
- **What "better" means** in numbers, and what would count as "not better". Users often ask for
  "clearly faster than X"; phase 2 tells you whether that is physically possible (decode on our M4 was
  already at ~92% of the reference line, so no level-A engine could be "much" faster there).

## Phase 1 — Calibrate the device

Measure, don't trust datasheets: DRAM read bandwidth, matrix-unit (MMA / tensor-core) peak per dtype,
on-chip capacities, dependent-kernel fixed cost, sync latency, concurrency needed to saturate bandwidth
and compute, CPU+GPU contention. Record every number with source and confidence in the device YAML.

Key traps (details: `references/1-calibration.md`):
- A best-observed burst value is a calibration, not a certified ceiling. Label it so everywhere.
- Measure the peak of the unit GEMM/attention actually use (simdgroup MMA, tensor cores), not a scalar FMA chain.
- Thermal state can move results more than any optimization (32k prefill: 55.8 s cold vs 70.7 s warm;
  4k TTFT 1.9 s vs 3.1 s after sustained load). Calibrate cold and sustained separately.

## Phase 2 — Reference lines from the model graph

Write the model YAML, then compute per-scenario reference times `T >= max(W/P, Q_b/B_b)` with the useful
graph (unused final-layer rows removed; every layer's K/V still produced). See `references/2-reference-lines.md`.

The point of this phase is navigation: it says where the time must go (decode: bytes; long prefill:
attention FLOPs) and how much headroom exists at each level. Treat it as a hypothesis generator, not
a proof of optimality — "close to the calibrated line" is not "close to the hardware limit".

## Phase 3 — Baselines and gap decomposition

Measure each existing engine on its own default path, then decompose the gap with a telescoping ablation
(GEMMs only → + attention → full forward → full generate). See `references/3-baselines-gap.md`.
The decomposition gives ordered marginal effects, not independent causal losses; isolated kernel times
do not add up (they overlap and compete for bandwidth).

## Phase 4 — Optimize an engine (agent loop)

Let the agent build an engine that attacks the gaps phase 3 found, under the phase-0 level.
The loop that worked: profile by stages → form a hypothesis tied to a gap → test it with a controlled
micro-experiment → keep or reject → record negative results too. Details, the list of changes that paid
off, and the ones that did not: `references/4-engine-optimization.md`.

Decide the correctness criterion **before** optimizing. A bitwise-identical requirement is stricter than
level A and rejected every useful change in our first attempt. Use an FP32 reference: the new engine's
error must not exceed the baseline's own BF16 error (median RMS, p99 max, p99 top-2 margin), and any
greedy divergence must be an FP32 near-tie.

## Phase 5 — Verify the speedup (the part people skip)

An in-process A/B loop showing "1.7× faster" is not evidence. Our first comparison ran both engines in one
process, synchronized only the default stream, timed lengths in blocks, and chose the analysis after
seeing data; short-prompt order effects were −16% to +30%. The protocol that fixed this
(`references/5-verification.md`, template `assets/CONFIRM_TEMPLATE.md`):

1. **Pre-register** in a committed file: hypotheses, scenarios, pairs per scenario, decision rule, exclusion
   and retry rules, divergence rule. Commit before collecting; never change it after seeing formal data —
   put post-hoc analyses in a separate, clearly labeled file.
2. **One fresh process per timed request**, one engine loaded, full-length warm-up with different content,
   synchronize the engine's own stream(s).
3. **Gate every request**: thermal state nominal, AC power, no sleep during the measurement (kernel sleep/wake
   timestamps). A closed lid on battery made a whole smoke run 3× slower while the thermal gate stayed green.
4. **Randomize** engine order within pairs and scenario order within blocks, fixed seed.
5. **Correctness**: FP32 numerical gate per engine; for generated-sequence divergences use a **signed** rule —
   void only when the engine's token is FP32-worse than the baseline's beyond the noise threshold
   (the baseline itself deviates; an unsigned rule once voided our engine for mlx-lm's own BF16 error).
   Calibrate the noise threshold at the context lengths you test — a threshold from ≤2k-token prompts was
   too tight at 4k and voided three scenarios.
6. **Decide** with a one-sided exact sign test on the paired-ratio median against a practical margin (1.02),
   Holm within metric families, two sessions both pointing the same way; bootstrap intervals are
   descriptive only. Size samples from pilot variance (short-prompt TTFT is ~10× noisier than decode).
7. **Smoke-test the whole pipeline** with real hardware conditions before the formal run, and estimate
   duration from the smoke, not from guesses.
8. Run unattended (launchd/systemd/tmux), resumable, with a log.

## Phase 6 — Report

Report verdicts per scenario exactly as the pre-registered rule produced them (faster / undecided / void),
with the reason for every void, plus descriptive ratios. State the conditions, the controls, and what the
experiment still cannot show (attribution, cross-day replication, other devices). Keep raw records and the
original analysis; mark every post-collection change. Language and evidence levels: `references/6-reporting.md`.

## Porting checklist (short)

New device → redo phase 1, write a device YAML, adapt the platform gates (`scripts/platform_state.py`
covers macOS and Linux/NVIDIA). New model → model YAML, re-check engine kernel assumptions (head dim,
GQA ratio, tied embeddings, norms), new corpus snapshot, FP32 reference must fit in memory. New engine →
one adapter with `stream`/`synchronize`, `generate(prompt, n)`, `teacher_forced(prompt, cont)`.
Full list: `references/porting.md`.

## Working with the user

These studies run for hours on the user's own machine. Before any long run: tell them the estimated
duration, the conditions it needs (plugged in, lid open, no other GPU load), and get a yes. Report progress
at milestones, surface anomalies immediately with evidence (e.g. "32k TTFT is 3× slower than in isolation;
power log shows clamshell sleep from 14:43"), and never quietly re-run or drop data.
