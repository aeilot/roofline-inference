# Phase 3 — Baselines and gap decomposition

## Baselines

Measure each existing engine through its own default user-facing path (e.g. mlx-lm `generate_step` or
`stream_generate`, `llama-bench`, vLLM offline `LLMEngine`), token IDs in and out, detokenizer excluded.
Record versions, settings that differ from defaults, and anything the engine does that the others do not
(mlx-lm computes full-vocab logprobs, clears its allocator cache between prefill chunks, grows KV by
concatenation, runs prefill as L−1 tokens then 1). These become explanations later.

Scenarios: TTFT at several prompt lengths (16 … 32k), decode ms/token at several contexts, a whole request.
Use the same prompts for all engines; random tokens plus real text.

## Efficiency vs the reference line

`efficiency = T_reference / T_measured` per scenario (`scripts/gap_report.py`). Present it as "fraction of
the calibrated reference", never as "fraction of the hardware limit".

## Gap decomposition

Telescoping ablation on the real weights (`benchmarks/ablation_mlx.py`): GEMMs only → + attention →
full forward → full generate. Each step's increase minus its ideal share is that step's loss.

- These are ordered **marginal** effects. Adding an operator changes caching, fusion, register pressure,
  overlap; a different order gives different numbers.
- Do not sum isolated kernel times: decode GEMVs measured alone added up to more than the whole step,
  because in a real step they overlap and compete for bandwidth.
- Validate the per-plan cost model against measured times (decode was over-predicted by 0.3–11% because the
  isolated-cost model assumes no overlap; 16k–32k prefill was under-predicted by 16–24%, partly throttling).

Outcome: a ranked list of gaps with a hypothesis for each — this is the input to phase 4.
