# Navigation pilot: does the resource reference line help an optimizing agent?

Registered 2026-10-08 before the first agent run. Design: `docs/navigation-experiment.md`; this
pilot runs two of its arms with two runs each. It estimates run-to-run variance and checks the
pipeline; it cannot support a confirmatory conclusion.

## Arms (identical except the materials list in TASK.md)

- **P (profiling)**: neutral starting engine (behaves like stock mlx-lm), `tools/` (check, bench,
  profile_ops, platform_state), `docs/HARDWARE.md`, `docs/BASELINE.md`.
- **B (bound)**: P plus `docs/BOUND.md` (reference lines per scenario, mlx-lm gap decomposition)
  and `roofline_tool/` (what-if calculator).

## Agent and budget

Codex CLI 0.161.0, model gpt-6.1-sol, reasoning effort medium, non-interactive (`codex exec`),
2 h wall clock per run (revised from 2.5 h, see below), hard stop; the engine at the last git commit is
evaluated. Order: P1, B1, B2, P2 (sequential: one GPU, agents benchmark on it).

## Isolation

Fresh workspace under `~/nav-runs/`; fresh CODEX_HOME with only auth (no memories, skills, MCP,
past sessions); macOS sandbox profile: no reads of this repository (except the Python environment),
`~/.claude` or `~/.codex`; writes only to the workspace, CODEX_HOME, caches and temp; network only
via a local proxy that tunnels to OpenAI hosts. Self-test T0 (2026-10-08) confirmed GPU access,
blocked repo / ~/.claude / ~/.codex reads, blocked github.com, and working git commits; `check.py`
and `bench.py` work inside the profile. Transcripts and proxy logs are audited afterwards.

## Evaluation (after all four runs)

Each final engine gets a blinded name and is evaluated in the v3 harness against stock mlx-lm
(E0): FP32 numerical gate (failing engine = void), process isolation, thermal and power gating,
the 8 TASK scenarios, signed divergence rule. Primary metric per run: geometric mean over
scenarios of the paired median ratio t_E0 / t_engine. Secondary: share of agent time spent in
directions with little reference-line headroom (from transcripts and NOTES.md), and the error of
each agent's predicted headroom against the measured result.

## Interpretation limits

Two runs per arm; agent variance is unknown and likely large. The model may already know roofline
analysis and Apple GPU tricks. Results are descriptive and are used to size a full experiment.

## Revision before the first complete run (2026-10-08 14:55)

- The first P1 (started 13:10) was stopped after ~72 min because the machine could not stay on AC
  power for the afternoon; it is kept as `runs/P1-aborted` and excluded. All four runs restart.
- The machine is available 17:30–08:30 with possible short interruptions, so: 2 h per agent run
  (was 2.5 h); no agent run starts after 02:30; the evaluation stops launching timed units at 08:15
  (resumable later); `~/nav-runs/PAUSE` pauses before the next agent run or evaluation unit.
  Order P1, B1, B2, P2 keeps one run per arm among the first two if the night is cut short.
- `tools/bench.py` in the kit now waits for AC power with the lid open without limit and repeats a
  measurement disturbed by power loss or sleep (previously it waited at most 5 minutes and then
  timed on battery). Both arms get the same tool.
- The evaluation covers every completed run, blinded (make_eval.py), 8 pairs per scenario.

## Revision 2 (2026-10-08, before any complete run): agent usage records

- Each run gets its own fresh CODEX_HOME and keeps Codex's session rollout (previously `--ephemeral`
  discarded it): timestamped model requests with cumulative input / cached / output / reasoning
  tokens, tool calls and reasoning summaries (`model_reasoning_summary = "auto"`, both arms).
  `nav_usage.py` reports tokens, requests, tool calls and the wall-time split (tool vs model) per run;
  these are additional descriptive outcomes (cost of each arm), not decision criteria.
- The sandbox now also denies reading `~/nav-runs` except the run's own workspace and CODEX_HOME:
  previously a later run could have read earlier runs' workspaces. Self-test T1 confirmed the
  denial, GPU access, blocked repo / ~/.claude / ~/.codex / github.com, and the rollout capture.

## Revision 3 (2026-10-08, before any complete run): review fixes

Primary metric (replaces "geometric mean over scenarios" above and the 10-metric version in the
first nav_analyze.py): scenario score = geometric mean of the paired median ratios t_E0 / t_engine
of the scenario's metrics (TTFT for ttft16/64/256/1024; TTFT and decode for ctx512/4096/32768; whole
request for req256x256); primary = geometric mean of the 8 scenario scores. TASK.md now states this
rule (ctx512 TTFT was named in TASK.md but missing from the metric list).
- A scenario voided by the signed divergence rule scores 1.0 (no speed-up), so every candidate is
  scored on the same 8 scenarios; previously the primary used whatever subset survived (v2 itself
  would lose ctx4096 on noise: 0.240 vs threshold 0.224). Sensitivity: valid scenarios only.
- Correctness the agent can see now matches the evaluation: `tools/check.py` adds the gate's rule
  that every top-1 disagreement must be within mlx-lm's p99 margin noise, and a long-prompt part
  (3000 and 9000 tokens through `generate`, chunked prefill) applying the evaluation's signed
  divergence rule. The starting engine passes (largest regret 0.185 < noise 0.336; long prompts identical), and so does the v2 engine through an API wrapper (0.184; identical), in ~2 min.
- TASK.md: the engine must be self-contained in `engine/` and committed (only `engine/` of the last
  commit is evaluated); no files written at run time, no state between processes.
- Isolation is an allowlist: nothing under the home directory is readable except the run's
  workspace, its box (own HOME, CODEX_HOME, TMPDIR), the Python environment and the model cache;
  /tmp and the user's temp dir are closed (git runs as the real binary, not the xcrun shim that caches there). The denylist had left
  ~/.claude.json, the Claude and Codex app data (15 and 1 files mentioning this project) and shared
  /tmp, ~/.cache readable. Self-test T2 confirmed all denials, GPU access, git and the private TMPDIR.
- Machine log every 60 s during agent runs (AC, lid, thermal state, load, HID idle time); run.json
  and the report give minutes on battery / lid closed / user active / thermal not nominal per run.
  `bench.py` waits for AC without limit while the agent's clock runs, so such minutes are unequal
  budget between runs; they are reported, not corrected.
- Before timing: blinded copies (runs/blinded/C-n, paths no longer name the run), static audit of
  file / process / network / serialization calls (nav_audit.py, manual review); after timing: any
  metric faster than 0.9x its resource reference line is flagged for review. Freezing covers every
  file in a candidate's engine/, not only .py; candidate subpackages cannot resolve to v2 modules.
- Schedule: estimated 2 h 12 min per agent run and ~3 h 25 min for the evaluation (320 units,
  from the confirm run's unit times), ~12 h 15 min in total. LAST_START moves to 02:00 so the
  evaluation of a run started then ends before NAV_STOP_AT 08:15 with ~1 h of margin.

Interpretation additions: arm P also has HARDWARE.md (calibrated rates), so the manipulation is
"pre-computed reference lines and gap decomposition", not "knowledge of the hardware". The aborted
P1 already found v2-like optimizations unaided, so a ceiling effect is plausible; with two runs per
arm the pilot sizes variance and cannot detect it. Usage numbers are approximate (requests counted
as token_count events; tool time can be low when long commands are polled; model time includes
streaming and network). Blinding keeps run identities out of logs; the evaluation is automatic and
pre-registered, so it is not a safeguard against analyst choices.

## Deviation log (during the run of 2026-10-08)

- 21:20: B1 ended with the operator's Codex 5-hour quota at 15% (P1 + B1 used ~85%). B2 was held with
  `~/nav-runs/PAUSE` until the quota reset at 22:50 (a ~1.5 h gap, no agent running). P2 follows B2
  in the same quota window by the operator's choice; whether any run hit a usage limit is checked
  in its transcript and reported with the results.
- 21:42: the operator switched Codex to another account (plan "prolite", weekly window only, 12% used),
  so B2 and P2 run on it and the hold ended at ~21:45 instead of 22:50. Same CLI, model
  (gpt-6.1-sol) and reasoning effort; each run copies the current auth.json when it starts.
  P1 and B1 ran on the first account, B2 and P2 on the second: one run per arm on each account.

## Results (2026-10-08/09)

All four runs completed: P1 17:48–19:16 (1.46 h, stopped itself), B1 19:21–21:20 (1.98 h), B2
21:45–23:19 (1.57 h), P2 23:24–01:22 (1.96 h). Evaluation 01:37–03:38 (after a 10-min cooldown), two sessions of 160 units,
no excluded units; every candidate passed the FP32 gate in both sessions. Report:
`agent-test/v3/results-nav-pilot/nav_report.md`; blinding key `runs/blind_key.json`
(C-1 = P2, C-2 = B1, C-3 = P1, C-4 = B2).

| Run | Arm | Primary | Valid-only | Void scenario |
|---|---|---|---|---|
| B2 | B | 1.224 | 1.224 | — |
| B1 | B | 1.137 | 1.158 | ctx512 |
| P1 | P | 1.116 | 1.116 | — |
| P2 | P | 1.090 | 1.104 | ctx512 |
| Arm B (geometric mean) | | 1.180 | 1.191 | |
| Arm P (geometric mean) | | 1.103 | 1.110 | |

- Direction: both B runs scored above both P runs (difference ~7%). With 2 vs 2, complete separation
  has probability 1/6 under no effect (one-sided exact permutation p ≈ 0.17): not evidence by itself.
  Within-arm spread was ~8% (B) and ~2% (P), the same order as the difference, so a confirmatory
  experiment needs roughly 6–10 runs per arm.
- Where the difference is: short-prompt TTFT and 32k prefill. Both B runs made a small-M GEMM work
  (B1: a 16-row kernel; B2: an adapted MLX steel GEMM, which also gave the best 32k TTFT, 1.22);
  P1 tried a small GEMM, found it slower than MLX and dropped it; P2's small-matrix path gained
  less. All four runs found the same first steps (packed projections, pruned final layer, no extra
  decode step). Decode gained 0–3% in every run, consistent with its ~5–8% reference-line headroom.
- Reference point: v2 scores ~1.22 on the same rule (confirm data, different sessions); B2 reached
  that level in 1.57 h, both P runs stayed below.
- Cost did not differ systematically: 13.5–18.7 M input tokens, 83–129 k reasoning tokens per run.
  The extra B performance did not come from extra compute.
- The two void scenarios are the same prompt and position (ctx512, session 1, block 4, token 10;
  FP32 gap 0.300): P2 and B1 chose the same token. The evaluation threshold is 0.224 (E0's p99
  margin noise on the gate's prompts) while kit `check.py` derives 0.336 from its own prompts, so
  Revision 3 aligned the rule but not the threshold value; B1 checked its timed prompts against
  0.3354 and passed. One void per arm; the valid-only ranking is unchanged. A full experiment must
  put the evaluation threshold into `check.py`.
- E0 itself chose the FP32-worse token beyond noise at ctx4096 and ctx32768; all candidates agreed
  with FP32 there (signed rule: not counted against them).
- Conditions: no battery or lid-closed minutes; user activity 7 / 20 / 28 / 0 min (P1/B1/B2/P2), the
  best run had the most; thermal state not nominal 20–29 min per run. P1, B1 on the first Codex
  account, B2, P2 on the second. No usage-limit errors. 32k TTFT medians 59–71 s (machine hot,
  baseline 54 s); its sign tests are uninformative.
- Audit: no proxy denials (only OpenAI hosts); no attempts to read outside the workspace. Sandbox
  denials were harmless and present in all runs: git's xcrun cache (the PATH fix did not apply,
  because the login shell `zsh -lc` reorders PATH), zsh here-documents in /tmp, and `ps`. Static audit
  flags were benign (`lru_cache` of a compiled kernel; `load(` in Metal code). No metric beat its
  reference line. B2 copied MLX's GEMM source with its license from the installed package (allowed).
- Headroom predictions (secondary): B runs quoted BOUND.md's reference ratios (e.g. ttft16 ~5x,
  marked optimistic); P runs estimated their own, more conservative limits (e.g. ttft16 ~1.6x).
  Not yet scored against the measured results.

Next steps for a full experiment: evaluation threshold in `check.py`; 6–10 runs per arm; an arm with
BOUND.md's conclusions but no calculator, to separate "being told the gap" from the tool; fix PATH
for git inside the login shell.
