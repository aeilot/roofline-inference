# Qwen3-0.6B BF16 engine

## Kept changes and measurements

All weights and KV entries remain BF16. Accumulation remains at least as precise
as stock MLX. Every retained implementation stage passed `tools/check.py` before
being committed. The API is unchanged, and all added runtime files live in
`engine/`; the Metal GEMM helpers are accompanied by MLX's MIT license.

The following are process-isolated `tools/bench.py` measurements against mlx-lm
at each implementation stage. They include changes already retained at that
stage, rather than being isolated ablations. Short TTFT varies with GPU wake-up
and thermal history, so small differences between separate calls are noisy.

| Change | Measured stage result versus mlx-lm |
|---|---|
| Process the final prompt chunk together; compute only the last final-layer query, attention output and MLP; share post-attention normalization | One repeat: TTFT ratios 1.130 / 1.317 / 1.738 / 1.257 at 16 / 64 / 256 / 1024; 4K and 32K essentially unchanged. |
| Combine Q/K/V and gate/up BF16 projection weights once at engine construction | One repeat: 64-token TTFT 1.421; 4K decode 1.010; 256+256 whole request 1.010. |
| Compile preparation and MLP subgraphs, with array-valued RoPE offsets | Two repeats: 64-token TTFT 1.498; 256+256 whole request 1.017. |
| Specialize 16-row matmuls with a Metal 16x32x64 tile and FP32 accumulators; yield the first token before preparing lookahead; stop before unused final lookahead | Two repeats: TTFT 1.500 / 1.450 / 1.222 at 16 / 64 / 256; whole request 1.010. A separate interleaved warm comparison showed about 4 ms saved by the 16-row matmul. |
| Fuse Q/K head RMS normalization, RoPE and Q/K/V layout conversion | Two repeats: decode 1.008 / 1.008 at 512 / 4K; 4K TTFT 1.032; whole request 1.013. Preserve both BF16 rounding steps in head normalization before rotating. |
| Adopt the first prefill's K/V arrays directly, then use ordinary MLX cache growth | Two repeats: 16-token TTFT 1.440. An interleaved warm comparison showed 1.4 ms lower 16-token TTFT. |
| Submit the first prefill layer early to overlap CPU graph construction with GPU work | Final measurement below. Interleaved warm comparison saved about 0.8 / 1.5 / 3.1 / 0.6 ms at 16 / 64 / 256 / 1024. |

## Tried and dropped

- Shapeless compilation: MLX 0.32.1 cannot infer output shapes for the required
  split/slice primitives. Shape-specific compilation reuses the decode graph.
- New MetalPerformancePrimitives matrix helpers: basis/random tests failed on
  this machine. The retained GEMM uses the older, correct MLX matrix helpers.
- Larger-row GEMM tuning: about 5% on one 64-row matrix shape, with other tile
  choices slower or invalid; no convincing general improvement.
- Prefill attention tile changes and removal of scheduling barriers: the best
  correct variants matched stock attention, while others were slower or exceeded
  threadgroup memory limits.
- Direct sixteen-token causal attention: slower in the full engine.
- Partitioned decode attention with FP32 partials: roughly 10% faster in the
  attention-only 32K microbenchmark, but slightly slower in the full engine
  (52.21 versus 51.72 ms per synchronous step). Slower at short contexts.
- Fused residual addition/RMS normalization: numerically matched standalone
  tests, but full-engine gains were too small and inconsistent to retain.

## Final benchmark and level-A estimate

Ratios greater than one mean faster. Final measurements used three repeats for
the seven shorter scenarios and two repeats for 32K, with the unmodified supplied
benchmark, fresh processes, randomized engine order, and its thermal gate.
Each metric is the ratio of the two engines' median times. The overall score,
computed exactly as TASK.md specifies, is **1.1600x**.

| Scenario | Final measured ratio | Scenario score | Optimistic level-A ratio | Reason for remaining resource limit |
|---|---:|---:|---:|---|
| TTFT 16 | 1.3267 | 1.3267 | 5.1083 | Weight bandwidth; launch and wake-up costs dominate practical short-prompt latency. |
| TTFT 64 | 1.6567 | 1.6567 | 4.6145 | Useful-product compute; small GEMMs achieve less than the calibrated arithmetic rate. |
| TTFT 256 | 1.2776 | 1.2776 | 2.3240 | Prefill matrix-product compute and dispatch overhead. |
| TTFT 1024 | 1.0982 | 1.0982 | 1.4999 | Prefill compute; larger stock MLX GEMMs already use the GPU efficiently. |
| Context 512 | TTFT 1.2007; decode 1.0161 | 1.1045 | TTFT 1.7721; decode 1.0969; score 1.3942 | Compute for prefill, weight bandwidth for decode. |
| Context 4096 | TTFT 1.0268; decode 1.0025 | 1.0146 | TTFT 1.3063; decode 1.0881; score 1.1922 | Prefill compute; decode streams weights and the growing KV cache. |
| Context 32768 | TTFT 0.9161; decode 0.9636 | 0.9395 | TTFT 1.3494; decode 1.0465; score 1.1883 | Attention compute for prefill, weights plus KV bandwidth for decode. |
| Request 256+256 | Whole request 1.0099 | 1.0099 | 1.1288 | The many bandwidth-bound decode steps dominate total time. |

The 32K scenario regressed in this final measurement. The retained implementation
improves the equal-weight overall score mainly through short-prompt TTFT; it does
not demonstrate a long-context speed-up. Decode has little bandwidth headroom.

The level-A estimates use the supplied calibrated resource calculator's
useful-product graph at 3.68 TFLOP/s and 102.7 GB/s, divided into the baseline
times in `docs/BASELINE.md`. The measured ratios instead use the paired baseline
workers in each final benchmark. These are optimistic resource estimates, not
certified ceilings or an attainable schedule: launch costs, GPU wake-up,
synchronization, and sustained thermal limits reduce practical gains. The
request estimate includes 255 decode steps after the first token from prefill.

Raw final timings and calculated ratios are preserved in `measurements/`.
`final_short.json` and `final_long.json` are the benchmark's original output;
`summary.json` contains medians, ratios and the overall score; `reference.json`
contains calculator times. `python measurements/report.py` reproduces the
summary from those files. Commands used:

```sh
python tools/bench.py --scenarios ttft16 ttft64 ttft256 ttft1024 ctx512 ctx4096 req256x256 --repeats 3
python tools/bench.py --scenarios ctx32768 --repeats 2
python tools/check.py
```

The final code passed `tools/check.py`: 1,216 teacher-forced positions, error
ratios 0.988 (median RMS), 0.951 (p99 maximum) and 0.893 (p99 top-two margin).
The sign-test p-value was 0.773 and worst FP32 regret was 0.1623, below the
baseline noise of 0.3355. Both 3,000- and 9,000-token prompts matched all 32
generated tokens. The complete output is in `measurements/check_final.txt`.
An additional audit used the benchmark's 32,768-token random prompts for
variants 0, 1 and 2, generating 129 tokens each. Variants 0 and 2 matched every
token. Variant 1 first differed at token 2; FP32 rated the engine's token
0.03978 logits higher than mlx-lm's, passing the same near-tie criterion.
The audit printed **PASS**. Its script and complete output are preserved as
`measurements/audit_long.py` and `measurements/audit_long.txt`; run it from the
repository root with `python measurements/audit_long.py --variants 0 1 2`.

## Time allocation

Approximately 20% small-prefill matmuls and projection fusion, 15% normalization
and layout fusion, 15% attention experiments, 10% scheduling/cache work, and 40%
correctness checks, thermal waits, final benchmarks and documentation. One GPU
benchmark ran at a time. No internet or unrelated directories were used.
