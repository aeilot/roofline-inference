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
