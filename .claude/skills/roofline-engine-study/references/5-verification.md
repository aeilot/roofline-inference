# Phase 5 — Verification protocol

Reference implementation: `agent-test/v3/`. Registered plan of the run that produced the current
result: `agent-test/v3/CONFIRM.md`; results: `agent-test/v3/results-confirm-20261008/`.

## Contents
1. Why a protocol (defects it fixes)
2. Components
3. Pre-registration
4. Correctness and divergence
5. Statistics
6. Running it
7. Time budget

## 1. Defects of a naive A/B loop

| Defect | Consequence | Fix |
|---|---|---|
| Engines alternate in one process | Shared allocator/compile caches and GPU state; order effects of −16%…+30% on short TTFT | One fresh process per timed request, one engine loaded |
| `synchronize()` only on the default stream | In-flight work leaks into the next measurement; request time undercounted | Synchronize each engine's own stream(s) |
| Lengths measured in consecutive blocks | Time drift confounded with length | Block-randomized order, fixed seed |
| Sample size chosen ad hoc | 6 pairs where log-ratio SD was 0.07–0.27 | Size from pilot variance |
| Analysis chosen after seeing data | Researcher degrees of freedom | Pre-registered rule, committed before collection |
| One afternoon, developer runs everything | No replication | Two sessions; the user launches it; frozen analysis script |
| Mixed scheduling and kernel effects | Cannot say what is faster | Optional fair-schedule baseline (E1) and ablations |
| Random tokens only | External validity | Half real text from a frozen corpus snapshot |
| Thermal gate only | A closed-lid smoke run was 3× slow with thermalState=0 | Gate AC power and sleep/wake per measurement |

## 2. Components (copy from `agent-test/v3/`)

- `adapters.py` — one class per engine: `stream` (or `synchronize()`), `generate(prompt, n)` yielding token
  ids with fixed `n` (ignore EOS), `teacher_forced(prompt, cont)` returning logits through the real path.
  `vllm_adapter.py` is the pattern for an external engine with its own scheduler.
- `worker.py` — one timed request per process: load → full-length warm-up with different content →
  thermal gate → idle/warm regime → time (TTFT at first token, decode gaps, request until synchronized) →
  power/sleep check → JSON. Records prompt hash, model snapshot, implementation identity.
- `run.py` — scheduler that never loads a model: preflight (AC power, freeze hashes of code + corpus +
  plan + environment + weights), probe calibration, gates, randomized units, per-unit AC wait, retry of
  excluded units once at session end, divergence judging, analysis. Profiles (`full`, `quick`, `simple`,
  `confirm`) are just dicts of scenarios × pairs × engines × gating.
- `gate.py` — FP32 numerical screen for every engine (phase-4 criterion); `vllm_gate.py` — token screen for
  engines without logit access.
- `divergence.py` — FP32 judgement of first divergences in generated sequences.
- `analyze.py` — the frozen decision rule; writes `report-corrected.md` / `summary-corrected.json`.
- `prompts.py` + `corpus/` — random-token prompts by seed and real-text prompts from a frozen snapshot
  (live docs change; a corpus read from them changed under us once).
- `run_both.sh` — two sessions with a gap, resumable; launch via launchd (macOS) or systemd/tmux so it
  survives the agent session.

## 3. Pre-registration (use `assets/CONFIRM_TEMPLATE.md`)

Write and commit before any formal data: engines and exact versions; environment; scenarios and metrics;
pairs per scenario and session split; prompt kinds; gating; warm-up; exclusion rules (gate timeout, worker
crash, not on AC, slept during measurement); retry rule (excluded units re-run once at session end, first
attempt kept); divergence rule; decision rule; what is descriptive only. Fixes found during smoke tests are
allowed **before** formal collection — record each with its reason and date. After formal data, the rule is
frozen; any reanalysis goes to a separate, labeled file and the original output stays.

## 4. Correctness and divergence

- Gate each engine with the phase-4 FP32 criterion at session start; stop the run if the baseline or the
  candidate fails. Comparators without logit access get a generated-token screen on fixed fixtures.
- Judge every first divergence between an engine and the baseline with an FP32 forward on the shared prefix.
  **Signed rule:** void the scenario only if the engine's token is rated lower than the baseline's by more
  than the noise threshold; if the engine's token is FP32-preferred, record "baseline deviated". (Two
  independent engines agreed on FP32's top token where mlx-lm took the runner-up by 0.252 > 0.224.)
- The threshold (baseline's p99 top-2 margin error) must come from prompts as long as the tested contexts.
  A ≤2k-token calibration was too tight at 4k: the baseline itself exceeded it 7 times, and one engine
  deviation of 0.240 voided three 4k scenarios under the registered rule.

## 5. Statistics

- Unit: one pair = all engines on the same prompt in the same block, order randomized.
- Statistic: per-pair ratio t_baseline / t_candidate (log scale).
- Decision: one-sided exact sign test that the paired median exceeds a practical margin (1.02), Holm within
  families (TTFT, decode, request), require both sessions present and each session's estimate > 1.
  With 6 pairs the smallest possible sign-test p is 1/64 — after Holm across 7 scenarios nothing can pass;
  24 pairs per scenario worked.
- Bootstrap intervals of the geometric mean: descriptive only (coverage not certified at small n).
- Do not call "interval inside ±2%" equivalence, or "half-width" a minimum detectable effect.
- Minimum-sample rule: below the planned pairs, report "insufficient", never a verdict from 1–2 pairs
  (a 1-pair bootstrap has zero width and gives p ≈ 0).
- Verdict words: faster / undecided / void (+ reason). A descriptive third engine adds no tests.

## 6. Running it

1. Dry run (`--dry-run`) to print the schedule; check unit IDs are unique.
2. Smoke run (`--smoke`): one block of every scenario, all gates and analysis. Inspect the numbers, not just
   the exit code — compare a few against isolated measurements (this is how the 3× clamshell slowdown was
   caught). Smoke data is never evidence.
3. Commit everything (manifests record `git_dirty`; only tracked files count).
4. Confirm with the user: duration, plugged in, lid open, no other GPU work, logged in.
5. Launch detached and resumable; monitor milestones; on anomalies, check the power log
   (`pmset -g log` on macOS) and the per-unit gate records before anything else.

## 7. Time budget

Estimate from smoke wall times per scenario, not from guesses. On the M4: a 32k unit with full-length
warm-up and gating ≈ 3–4 min; short units 6–20 s; the 648-unit confirm run took ~3.5 h per session.
Per-request thermal gating made the original full profile ~10 h per session (80% waiting); gating once per
(block, scenario) group cuts it ~5× at the cost of some within-group heat, balanced by randomized order.
