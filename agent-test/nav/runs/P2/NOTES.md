# Optimization notes

Work began at approximately 23:24 CDT on October 8, 2026. The hard stop is
01:24 CDT on October 9. All GPU experiments ran sequentially, offline, using
the supplied Python environment.

## Kept changes and measurements

The first passing commit combines Q/K/V projections and gate/up projections,
prefills short prompts in one pass, computes only the final query and MLP in
the final layer when logits are needed, skips unused final-layer output during
intermediate prefill chunks, retains MLX's allocation cache during prefill,
and avoids generating a discarded token after the requested continuation.
Weights and KV states remain BF16. Combined projection weights are derived
once from model weights; no request activations or KV states are reused.

`tools/bench.py --scenarios ttft16 ttft256 ctx512 --repeats 2` measured:

| Scenario | TTFT ratio | Decode ratio | Request ratio |
|---|---:|---:|---:|
| ttft16 | 1.260 | 1.012 | 1.187 |
| ttft256 | 1.172 | 0.988 | 1.151 |
| ctx512 | 1.139 | 1.003 | 1.026 |

The full supplied correctness check passed for this commit. Median RMS error
ratio was 1.001, p99 maximum error ratio 0.991, and p99 margin error ratio
0.874. Both long-prompt generation tests matched all 32 reference tokens.

## Experiments

- Fused residual addition and RMS normalization: dropped. A sequential paired
  microbenchmark showed essentially unchanged prefill and slightly worse
  synchronous decode. The stock RMS kernel also rounds normalized values to
  BF16 before multiplying by normalization weights; preserving this step is
  necessary when replacing it.
- Compiled prefill plus fused Q/K normalization, RoPE and layout conversion:
  aggregate numerical metrics passed, but the largest FP32 regret exceeded
  the check's noise bound (0.3989 versus 0.3355). Not committed.

Compiled prefill alone passed with the same numerical metrics as the first
commit. Its two-repeat process-isolated benchmark measured TTFT ratios 1.297
(16), 1.249 (64), 1.120 (256), and 1.158 (1024). The synchronous paired
comparison showed its clearest incremental benefit at 64 tokens (30.5 to
28.2 ms); larger prefills remain dominated by GPU computation. Compilation
reuses execution graphs, not request data.

For prefills longer than 64 tokens, the final layer now projects Q only for
the last token, and omits Q entirely in intermediate prefill chunks. This
passed the supplied check with unchanged metrics. Paired synchronous forward
timings fell from 84.5 to 83.8 ms at 256 and 343.5 to 340.9 ms at 1024.
The two-repeat supplied benchmark measured TTFT ratios 1.407 (256), 1.059
(1024), and 1.012 (4096). These small incremental savings are within the
variation between separate benchmark sessions; the final table uses the
frozen final engine's measurements instead of selecting historical bests.

Chunked long requests reserve their rounded BF16 KV capacity on the first
update instead of repeatedly allocating and concatenating larger buffers.
The supplied correctness check again passed with unchanged metrics and both
long continuations matching. A one-repeat supplied benchmark measured
TTFT/decode ratios 1.011/1.005 at 4096 and 1.019/1.011 at 32768.

A custom small-row projection uses 16x32 output tiles, 16-wide K blocks,
FP32 simdgroup matrix accumulation, and BF16 inputs/outputs. It replaces only
16-row, 1024-feature packed QKV and gate/up matmuls; other shapes stay native.
Sampled 1024-wide products matched MLX bit-for-bit. The complete paired
16-token forward improved from 23.1 to 19.2 ms, and the supplied three-repeat
benchmark measured 39.08 versus 50.80 ms TTFT (ratio 1.300). The full supplied
check passed with unchanged metrics. Wider output tiles and larger K blocks
were slower; 64-row and narrower-output projections remain native.

At 512 or more prefill rows, native separate Q/K/V and gate/up projections
are retained instead of packed GEMMs. In paired tests this saved about 1%
at 1024, 2048 and 4096, while separating at 256 was slower. The supplied
two-repeat benchmark measured TTFT ratios 1.091 at 1024 and 1.026 at 4096.
The full supplied correctness check again passed with unchanged metrics.

Additional discarded experiments:

- Fixed-capacity, masked, compiled decode: matched the passing numerical
  metrics in an instrumented copy of the checker, including both long tests.
  Process-isolated decode ratios were 0.988 at 512 and 0.990 at 4096, so it
  was discarded despite a favorable synchronous microbenchmark.
- Combining Q/K normalization with native RMS normalization, a broadcast
  multiply and a single RoPE call: slightly slower in paired measurements.
- Retiling the installed NAX attention source: output disagreed with native
  attention; discarded before integrating into the engine.
- Shape-independent prefill graphs: MLX 0.32.1 cannot infer the relevant
  Split/Slice output shapes. Shape-specialized compiled graphs are retained.
- CPU/GPU projection splitting: one standalone gate/up shape improved, but
  the complete forward slowed from 755 to 930 ms at 2048 and from 1856 to
  2227 ms at 4096. Discarded; all retained computation stays on the GPU.
- Prefill chunk sizes 1024, 2048, 4096 and 8192: only roughly 1% differences
  at 4096 and 9000 in a quick sequential scan. The default remains 2048.

## Final scenario measurements and remaining headroom

The final engine implementation is commit `b6831ff`. The seven shorter
scenarios used three repeats in one supplied benchmark invocation; 32768
used two separate one-repeat invocations (the same variant-0 prompt).
Ratios below divide stock median time by engine median time. Context scores
are the geometric mean of TTFT and decode ratios; the request score uses
whole-request time. Raw measurements and exact scores are archived in
`BENCHMARK_RESULTS.json`.

| Scenario | TTFT ratio | Decode ratio | Scored ratio | Estimated best Level-A score | Reason for estimate |
|---|---:|---:|---:|---:|---|
| ttft16 | 1.248 | — | 1.248 | 1.65 | Small-row projections have substantial bandwidth and launch-cost headroom. |
| ttft64 | 1.382 | — | 1.382 | 1.60 | Calibrated 64-row matmuls reach 2.60 TFLOP/s versus about 3.5 peak. |
| ttft256 | 1.245 | — | 1.245 | 1.55 | Better fused scheduling and matmul utilization can reduce the remaining prefill work. |
| ttft1024 | 1.095 | — | 1.095 | 1.30 | Dense matmuls and causal attention already approach compute limits. |
| ctx512 | 1.182 | 1.005 | 1.090 | 1.22 | Prefill has compute headroom; decode is dominated by streaming weights. |
| ctx4096 | 1.030 | 1.005 | 1.017 | 1.13 | Dense attention can improve, but decode's roughly 1.67 GB of reads limits gains. |
| ctx32768 | 1.039 | 1.008 | 1.024 | 1.15 | Dense attention has some compute headroom; roughly 4.95 GB of decode reads implies a 48 ms floor. |
| req256x256 | 1.256 | 1.009 | 1.019 | 1.07 | The whole-request score is dominated by 255 bandwidth-bound decode steps. |

The eight-scenario geometric mean is **1.1333x**. Final short-prompt TTFT
medians were 40.76, 49.44, 108.31 and 365.53 ms. The two long timing pairs
gave TTFT ratios 1.071 and 1.007, illustrating the remaining measurement
variation on this fanless machine. They are pooled using median times, not
by choosing the faster pair.

The Level-A estimates are rough optimistic attainable scores for exact
BF16 dense computation, not measured results or proven bounds. They use
the calibrated 102.7 GB/s read bandwidth and approximately 3.5 TFLOP/s MMA
throughput, allowing for idle wake-up and kernel overhead. Sustained thermal
limits can reduce attainable throughput. Decode estimates account for BF16
weights plus 114688 bytes of KV reads per context token across all layers.

Final `tools/check.py` printed **PASS**: median RMS error ratio 1.001,
p99 maximum error ratio 0.991 and p99 top-two-margin error ratio 0.874.
The largest FP32 regret was 0.1835, below the 0.3355 noise bound. Both
supplied long tests (3000 and 9000 prompt tokens) matched all 32 generated
tokens. Additional 32768-token prompts with seeds 10000, 10001 and 10002
each matched all 129 generated tokens: **387/387**. These three distinct
prompts were correctness tests; the two final long timing pairs reused the
benchmark's variant-0 prompt.

All model packing and evaluation use the advertised generation stream.
The engine's only additional dependency is `engine/small_mm.py`; imports
work both as a package and when the evaluator imports the module directly.
The final hand-back commit changes documentation/results only after the
passing engine commit.

## Time allocation

Approximate wall-clock allocation, including time waiting for GPU work:

- 50% correctness checks and process-isolated benchmarks.
- 20% prefill graphs, final-layer pruning, cache allocation and scheduling.
- 15% small-row matmul, normalization and attention kernel experiments.
- 5% decode compilation and cache-mask experiments.
- 5% CPU/GPU projection splitting and chunk-size exploration.
- 5% recording results and preparing the hand-back.
