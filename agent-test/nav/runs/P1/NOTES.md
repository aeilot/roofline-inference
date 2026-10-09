# Qwen3-0.6B BF16 engine optimization

## Kept changes

The engine packs shared-input Q/K/V and gate/up projections into BF16 matrices;
prefills a short prompt in one forward call; computes only the final query and
feed-forward result in the final transformer layer while retaining every KV
entry; fuses query/key RMS normalization and RoPE for up to 64 positions; uses
4096-token prefill chunks; and yields the first token before starting decode
lookahead. Decode remains pipelined, with no extra forward after the requested
last token. Weight packing and inference run on `Engine.stream`.

The fused normalization kernel preserves MLX's intermediate BF16 rounding:
normalize in FP32, round to BF16, multiply the BF16 learned scale, round again,
then apply rotary embedding and round. Keys and values use the stock BF16 cache.

Each retained stage passed `tools/check.py` and was committed after a supplied
benchmark showed an improvement. These are whole-engine ratios at each stage,
not isolated attribution to every operation within a stage:

| Stage / retained changes | Measured effect from `tools/bench.py` | Evidence |
| --- | --- | --- |
| Packed projections; one-pass short prefill; normalize only the final output | TTFT 16: 1.378x; 64: 1.373x; 1024: 1.054x. Decode essentially unchanged. | `bench_projection.log`, `check_projection.log` |
| Eliminate unused final-layer queries/MLP outputs; stop after the requested last token | TTFT 256: 1.529x; 1024: 1.074x; 256+256 request: 1.033x. | `bench_final_layer.log`, `check_final_layer.log` |
| Fused BF16 Q/K normalization and RoPE | TTFT 64: 1.331x; ctx512 decode: 1.008x. A direct interleaved comparison against the preceding engine found 6–7% lower warm short-prompt TTFT. | `bench_norm_rope.log`, `compare_norm_rope.log`, `check_norm_rope.log` |
| 4096-token chunks; first yield before lookahead | TTFT 16: 1.210x; 256: 1.208x; ctx4096 TTFT: 1.034x, decode: 1.005x. | `bench_schedule.log`, `check_schedule.log` |

Short-prompt TTFT varies substantially between sessions on this fanless machine.
In particular, the final stock 64-token median is slower than earlier stock
measurements. Treat differences between the stage rows cautiously; every ratio
within a row is from an interleaved stock/engine comparison.

## Tried and dropped

- MLX compilation of projection and MLP regions: roughly 0–2% differences,
  mixed signs, no consistent decode improvement (`compare_compile.log`).
- Residual-add plus RMS normalization fusion: isolated outputs exactly matched
  MLX, but full-engine TTFT was about 0.5–1% slower, with unchanged decode
  (`test_add_norm.log`, `compare_add_norm.log`).
- Custom FP32-accumulating SIMD-group GEMM for 16-row BF16 inputs: MLX's matrix
  sweep took 14.43 ms versus 15.15 ms for the first kernel. Larger K tiles and
  padded shared-memory layouts were slower still (26.21 and 17.79 ms).
  These kernels were never used by the retained engine (`test_small_gemm*.log`).
- QKV GEMV fused with normalization/RoPE: 2.82 ms for MLX versus 2.87 ms for the
  custom 1024-thread kernel; a 256-thread variant also lost
  (`test_qkv_gemv*.log`).
- Gate/up GEMV fused with SwiGLU: 3.87 ms for MLX versus 4.11 ms for the custom
  kernel, with additional BF16 rounding differences (`test_gated_gemv.log`).
- Exact greedy output-head completion using conservative Cauchy bounds: the
  512/768-feature prefixes left roughly 90–100% of vocabulary rows active;
  3.33 ms for the dense head versus about 3.65–3.70 ms for bounded completion.
  Reordering features by final normalization weights did not tighten the bounds
  enough, and the prototype's packing layout was much slower. Neither variant
  was integrated (`test_bounded_head.log`, `test_bounded_head_sorted.log`).
- 1024-token prefill chunks: consistently slower at 4k and 9k. 8192-token chunks
  were slightly better at 9k but worse at 4k, without convincing long-context
  evidence. The direct 32k sweep drifted into OS thermal state 1 and is not a
  reliable estimate of speedup. Default 4096 was selected from the shorter
  comparisons, then checked with the supplied benchmark (`chunks.log`).

## Final measurements and estimated remaining headroom

Final commands: `tools/bench.py --repeats 3`, followed by
`tools/bench.py --scenarios ttft64 --repeats 7` to resolve the unusually noisy
64-token result, using the task's Python interpreter, offline model loading and
a workspace-local platform-state cache. The table uses seven repeats for
ttft64 and three for the other scenarios. Results are
process-isolated, with the supplied thermal/power gate and randomized engine
order. Ratios greater than one are faster. Context scenario scores are the
geometric mean of TTFT and decode ratios; the request scenario uses total time.

| Scenario | TTFT ratio | Decode ratio | Whole-request ratio | Scored ratio |
| --- | ---: | ---: | ---: | ---: |
| ttft16 | 1.166 | — | — | 1.166 |
| ttft64 | 1.280 | — | — | 1.280 |
| ttft256 | 1.172 | — | — | 1.172 |
| ttft1024 | 1.117 | — | — | 1.117 |
| ctx512 | 1.164 | 1.005 | — | 1.081 |
| ctx4096 | 1.031 | 1.006 | — | 1.019 |
| ctx32768 | 1.032 | 1.009 | — | 1.021 |
| req256x256 | — | — | 1.019 | 1.019 |

**Equal-weight overall geometric score: 1.106x.** Raw timings and all repeats
are in `bench_final.log`, `bench_ttft64_confirm.log`, and
`experiments/bench_report.json`. The report combines the more stable ttft64
confirmation with the other seven rows from the complete benchmark.
At 32k, stock/engine TTFT was 54.945 / 53.225 s and decode was 50.56 / 50.10 ms.
The 256+256 request took 3.463 / 3.397 s.

The initial 64-token stock samples were 93.15, 93.42 and 60.58 ms; engine samples were
51.60, 46.40 and 70.34 ms. The initial 1.805x median ratio is therefore noisier
than the consistent ~1.3–1.4x ratios in earlier sessions. The initial complete
three-repeat run scored 1.155x overall, but the seven-repeat follow-up gave
69.90 / 54.59 ms, or 1.280x. The reported table and 1.106x overall score use
this follow-up rather than the inflated initial ratio.

The best-possible estimates below concern the allowed BF16 implementation
changes on this machine. They are engineering estimates, not demonstrated
ceilings. They use the calibrated ~102.7 GB/s bandwidth and ~3.5 TFLOP/s compute
numbers, and retain cold-GPU/startup costs. The model reads approximately
1.192 GB of weights per decode step, plus 0.059 / 0.470 / 3.758 GB of KV at
512 / 4096 / 32768 positions. This leaves considerably less decode headroom
than short-prefill headroom. Timing variability makes the short-prompt estimates
particularly uncertain.

For comparison with the final table, estimates use its stock medians, including
the 64-token confirmation run.

| Scenario | Estimated best possible scored ratio | Reason |
| --- | ---: | --- |
| ttft16 | ~1.6x | A specialized small-M matrix implementation could approach the weight-read floor; output-head reads and cold GPU startup remain. |
| ttft64 | ~1.7x | ~56 GFLOP of projections can approach 3.5 TFLOP/s, without the extra decode step; head reads and cold GPU startup remain. |
| ttft256 | ~1.5x | About 219 GFLOP after pruning unused final-layer work, plus attention, head and startup, suggests ~85–90 ms. |
| ttft1024 | ~1.3x | ~0.87 TFLOP of projections and ~0.12 TFLOP of causal attention suggest a ~310 ms request front end. |
| ctx512 | ~1.2x | TTFT could reach ~1.35x; dense decode has only ~6–8% room beyond its ~1.25 GB/step memory traffic. |
| ctx4096 | ~1.13x | TTFT could reach ~1.20x from compute/attention efficiency; ~1.66 GB/step gives ~6% decode headroom. |
| ctx32768 | ~1.16x | Better attention compute efficiency could reach ~1.28x TTFT; ~4.95 GB/step leaves about 4–5% dense decode room. |
| req256x256 | ~1.08x | Most time is repeated weight/KV reads; even ideal prefill and fewer dependent kernels save a modest fraction. |

## Correctness

Final retained candidate: `check_schedule.log` prints PASS over 1216 positions.
Engine / stock errors: median RMS 0.997x, p99 maximum 0.919x, p99 top-two margin
0.923x. Engine-only / stock-only FP32 top-one disagreements: 5 / 9, sign-test
p = 0.910. Largest FP32 regret: 0.1835, below stock margin noise 0.3355.
The 3000- and 9000-token prompts match stock on all 32 generated tokens.

Additional validation of all three benchmark prompt variants at all eight
scenarios prints PASS (`check_timed.log`, `experiments/check_timed.py`).
It applies the same first-difference FP32 rule, with a conservative 0.3350
noise threshold. All 387 generated tokens on the three 32768-token prompts
match stock exactly. Of the 24 requests, 21 match exactly; the other three
have first-difference FP32 gaps of -0.0377, +0.1448 and +0.0844, all below
the threshold.
The four additional 64-token variants from the seven-repeat confirmation also
print PASS (`check_ttft64_extra.log`). Two match exactly; the other two have
FP32 gaps +0.0327 and -0.0493. All 28 distinct timed prompt/scenario cases
therefore pass the requested rule.

## Time allocation

Rough allocation: 45% numerical checks and thermally gated benchmarking;
25% normalization, projection and matrix-kernel experiments; 20% prefill
attention/chunk sizing; 5% generation scheduling; 5% analysis and documentation.
GPU experiments were run sequentially on the single GPU.
