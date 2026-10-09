# Task: make this inference engine faster than mlx-lm

You have a working engine, `engine/engine.py`, that runs Qwen3-0.6B (BF16) on this MacBook Air M4 GPU
with MLX and currently behaves exactly like stock mlx-lm. Make it as fast as you can on the scenarios
below, then hand it back. Its speed will be measured afterwards, independently, against stock mlx-lm.

## Scenarios (equal weight)

TTFT at prompt 16, 64, 256, 1024 tokens; TTFT and decode time per token at contexts 512, 4096 and
32768 (129 tokens generated); whole-request time for 256 prompt + 256 generated tokens.
Batch 1, greedy decoding, fixed number of generated tokens (EOS ignored), token IDs only.

Score: for each of the 8 scenarios, the geometric mean of its speed ratios vs mlx-lm (TTFT; TTFT and
decode for the three contexts; whole request for 256+256); overall, the geometric mean over the 8.

## Rules

- Same mathematics: BF16 weights and BF16 KV cache, the same model function. You may change the
  implementation freely — fusion, kernel rewrites (e.g. `mx.fast.metal_kernel`), scheduling, memory
  layout, summation order, removing work whose result is never used. No quantization, no lower
  precision, no speculative decoding, no approximate or sparse attention, no caching across requests.
- `python tools/check.py` must print PASS for the engine you hand back (numerical error must stay at
  mlx-lm's BF16 noise level), and `python tools/check.py --full` before your final commit. It uses the
  evaluation's own threshold and its random-token timed prompts; the evaluation additionally checks
  natural-text prompts with the same rule. A scenario whose generated tokens break the rule in
  check.py part 3 counts as no speed-up.
- Keep the API in `engine/engine.py`'s docstring unchanged; the evaluation calls exactly that API.
- Everything the engine needs must live in `engine/` (Python modules, kernel sources, tables) and be
  committed: the evaluation copies only `engine/` from your last commit and imports `engine/engine.py`.
  Do not write files at run time; keep no state between processes.
- Work only inside this directory. Do not read or search other directories on this computer and do not
  use the internet; the installed Python packages (MLX, mlx-lm and their sources) may be read.
- One GPU, one machine: do not run two GPU benchmarks at the same time.

## Budget

About {HOURS} hours of wall-clock time from now (hard stop at {DEADLINE}). Plan for it: the engine as
of your last git commit before the stop is what gets evaluated.

- Commit (`git add -A && git commit -m "..."`) after every change that improves speed **and** passes
  `tools/check.py`. Never leave the last commit in a failing state.
- If you run out of ideas early, stop early.

## Materials

{MATERIALS}

## Tools

- `tools/check.py` — API + numerical check incl. the evaluation's timed prompts (~4-6 min; `--full` adds 32k, ~10 min more). PASS required.
- `tools/bench.py --scenarios ... --repeats N` — process-isolated timing vs mlx-lm, thermally gated.
  Prints mlx-lm / engine ratios (> 1 = faster).
- `tools/profile_ops.py` — where time goes inside prefill and decode, vs matmuls and attention alone.
- `tools/platform_state.py` — thermal / power / sleep state of the machine.
- Python: `{PYTHON}` (MLX {MLX_VERSION}, mlx-lm {MLXLM_VERSION}).

## Hand-back

Before the deadline, write `NOTES.md`:
1. Each change you kept, with the measured effect (from `tools/bench.py`), and changes you tried and dropped.
2. For every scenario: your final measured ratio vs mlx-lm, and your estimate of how much faster than
   mlx-lm the best possible level-A implementation on this machine could be, with a one-line reason.
3. Where you spent your time (rough percentages by area: e.g. decode matmuls, prefill attention, scheduling, …).
