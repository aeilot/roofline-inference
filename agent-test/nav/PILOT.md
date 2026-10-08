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
2.5 h wall clock per run, hard stop; the engine at the last git commit is evaluated.
Order: P1, B1, B2, P2 (sequential: one GPU, agents benchmark on it).

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
