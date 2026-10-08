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
