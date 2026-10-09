# Inference engine hand-back

The API is unchanged. Weights and valid KV entries remain BF16. There is no
quantization, speculation, sparse attention, request reuse, runtime file I/O,
or state persisted between processes. All engine dependencies added here are in `engine/`.

## Changes kept and measurements

These are interleaved, process-isolated measurements from `tools/bench.py`.
Separate sessions had visibly different cold-start and thermal behavior; their
ratios are observations, not additive attribution of individual changes.

1. Combine Q/K/V and gate/up weights, evaluate post-attention normalization once,
   process the final prompt chunk without a separate last-token pass, and prune
   final-layer outputs: cache-only chunks need only K/V in the final layer;
   the last chunk needs its final attention/MLP row only. Also omit the unused
   decode after the last output and retain temporary allocations between chunks.
   First two-repeat session: TTFT ratios 1.204/1.787/1.636/1.083 at
   16/64/256/1024; ctx512 1.296 TTFT and 1.012 decode; ctx4096
   1.025/1.015; ctx32768 0.975/1.004; whole request 1.014.
2. A custom 16-row GEMM for the combined projections: BF16 input/output,
   FP32 SIMD-group accumulation, a 16x64 tile without padded rows or staging.
   Three-repeat `ttft16`: mlx-lm 50.50 ms, engine 33.59 ms, ratio 1.503.
   The custom path is used only for sixteen rows; other shapes use MLX.
3. Compile initial prefill as a pure function returning logits and every layer's
   K/V, and yield the first token before scheduling the next decode. Subsequent
   tokens retain the pipeline. Two-repeat session with native decode:
   ttft64 66.26/51.40 ms (1.289); req256x256 3515.05/3480.96 ms (1.010).
   An interleaved scheduling ablation at 512 tokens measured 213.42 -> 198.08 ms.
4. Reserve long-request KV buffers once, rounded to 256 slots, for capacities
   >=4096. This avoids repeated cache growth/copies and the associated allocator
   accumulation. Final combined ctx4096 and ctx32768 ratios are in the table
   below; there was no isolated benchmark attribution for this cache change.
   Views expose only valid tokens; no padded slots participate in attention.

## Tried and dropped

- Fused decode Q/K normalization and RoPE: preserved explicit BF16 rounding,
  passed the full checker, and matched native Q/K on random probes at positions
  0 through 65535. Paired full-forward decode timings improved about 1%:
  14.087 -> 13.952 ms at 512; 18.448 -> 18.273 ms at 4096. Two-repeat
  benchmark ratios were ctx512 1.160 TTFT/1.009 decode, ctx4096 1.022/1.011,
  request 0.999. However, the final timed ctx512 prompt differed at token 9
  with FP32 regret 0.371557 > noise 0.3355. Native Q/K operations moved its
  first difference to token 13 with regret 0.144821. Dropped the fused kernel
  and reran the final correctness and benchmark suite.
- Full compiled decode with padded-capacity attention: correct, but no consistent
  decode gain and extra reads. Two-repeat ratios: ctx512 decode 1.021,
  ctx4096 decode 0.992, req256x256 0.993. Retained compiled prefill only.
- Shape-independent full decode compilation: MLX rejected Split/Slice shape
  inference. Capacity bucketing worked, but was part of the padded trial above.
- NAX/MPP register-tile experiment: produced large numerical errors on this M4;
  rejected before integration. The kept GEMM uses ordinary float SIMD matrices.
- Split-K variants: reduction/storage overhead defeated the increased parallelism.
  The kept small GEMM computes each entire dot product in one SIMD group.
- A 64-row QKV tile: operator-only testing suggested ~1.47x, and nine prompts
  matched logits and every cache exactly. Complete warm forwards measured
  25.393 ms native vs 25.558 ms custom, so the 64-row specialization was dropped.
- Fused RMS without the intermediate BF16 cast: mismatched native normalization.
  Corrected before integration; the full final check passed.

## Final measurements and remaining room

Final run: all eight scenarios, one process-isolated pair per scenario, with
32K last. This small sample is noisy; earlier two-/three-repeat sessions above
provide additional context. Final raw records are in `bench_final.json`.
The timing tool was extended only to retain already-yielded token IDs after each
timestamp and report exact matches; sampling, warm-up, gating and timing are unchanged.

| Scenario | Final mlx-lm/engine ratios | Scored ratio | Optimistic best level-A estimate | Reason |
|---|---|---|---|---|
| ttft16 | TTFT 2.389 | 2.389 | 5.09 | Weight traffic dominates; eliminate the separate last-token pass and small-matrix inefficiency. |
| ttft64 | TTFT 1.432 | 1.432 | 4.60 | Small matrix throughput and launch overhead leave substantial compute slack. |
| ttft256 | TTFT 1.248 | 1.248 | 2.32 | Dense projections and MLP throughput dominate prefill. |
| ttft1024 | TTFT 1.276 | 1.276 | 1.50 | Matmuls already approach the calibrated compute rate. |
| ctx512 | TTFT 1.162; decode 0.991 | 1.073 | 1.39 | Prefill has compute slack; decode is already close to the bandwidth line. |
| ctx4096 | TTFT 1.028; decode 1.008 | 1.018 | 1.19 | Prefill matmuls/attention have modest compute slack; decode is bandwidth limited. |
| ctx32768 | TTFT 1.038; decode 1.005 | 1.021 | 1.19 | Dense attention dominates prefill; KV traffic limits decode. |
| req256x256 | request 1.017 | 1.017 | 1.13 | The request is mostly decode, whose weight traffic is nearly irreducible. |

Overall geometric mean of the eight scored ratios: **1.256x**.
6 of 8 timed request pairs matched every generated token exactly.
Every other pair passed the task's first-difference FP32 noise rule:

- ctx512, repeat 0: first difference at token 13; FP32 rates the engine token +0.144844 below mlx-lm's (conservative noise threshold 0.3354; PASS).
- req256x256, repeat 0: first difference at token 12; FP32 rates the engine token +0.084368 below mlx-lm's (conservative noise threshold 0.3354; PASS).

Detailed results are in `timed_validation.json`.

The best-level-A column estimates resource headroom from `docs/BOUND.md` and
the supplied calculator, using the documented baseline medians. For ctx512,
the calculator gives 126.74 ms prefill; for the whole request it gives 3.08647 s.
Context estimates combine TTFT and decode geometrically, as the scoring does.
These calibrated resource lines are optimistic, not certified ceilings or an
attainability claim. They omit fixed startup/synchronization, dependencies and
thermal degradation; short-prompt practical gains will be much smaller.

## Correctness and time allocation

Final `python tools/check.py`: PASS, 1216 positions. Engine/mlx-lm errors:
median RMS 0.0464/0.0464; p99 maximum 0.5748/0.5798; p99 top-two margin
0.2933/0.3355. Only-engine/only-mlx-lm top-one disagreements 9/7,
sign-test p=0.402; largest FP32 regret 0.1835 < noise 0.3355.
The 3000- and 9000-token generation tests matched all 32 tokens exactly.

Approximate wall-time allocation: 25% prefill/matrix kernels, 25% scheduling
and compilation, 15% decode fusion and cache layout, 35% correctness checks,
benchmarking, thermal waits and hand-back. One GPU workload ran at a time.
