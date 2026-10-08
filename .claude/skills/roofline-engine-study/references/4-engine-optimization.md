# Phase 4 — Engine optimization (agent loop)

Reference: `agent-test/v2/` (engine.py, kernels.py, attention.py, check.py, test_engine.py) and its README,
which lists every change with the evidence type behind it. `agent-test/lessons-report.md` is the first
attempt that failed to beat mlx-lm and explains why.

## 0. Correctness criterion first

Write the acceptance test before optimizing (`agent-test/v2/check.py`):

- Reference: the same weights upcast to FP32, teacher-forced over the FP32 model's own greedy continuation
  (neutral to both engines), through each engine's **real** prefill/decode path.
- Accept if the engine's median RMS logit error, p99 per-position max error and p99 FP32 top-2 margin error
  are each ≤ 1.1× the baseline's BF16 error; top-1 disagreements not significantly more frequent
  (paired sign test); every free-running divergence an FP32 near-tie.
- Use ≥ ~1500 positions: with ~600, p99 statistics rest on the 8th-largest value and a baseline-equivalent
  engine failed by 3% from noise.
- Bitwise equality is stricter than level A (sum order may change) — it rejected every useful change in the
  first attempt. FP32 is a reference, not exact arithmetic; passing is an empirical screen, not a quality
  non-inferiority proof.

## 1. Loop

1. Profile by **stages**: cumulative timing with pieces added one at a time (e.g. norm+QKV → +qk-norm/RoPE
   → +cache write → +attention → +O → +MLP). Compare each piece with its ideal (GEMM at measured MMA peak,
   memory ops at bandwidth). This finds the "other" time that per-kernel views hide.
2. Pick the largest gap that the reference line says is recoverable.
3. Test a hypothesis with a controlled micro-experiment: same inputs, variants differing in one thing,
   interleaved repeats, thermal gate.
4. Keep, reject, or narrow. Write negative results down with numbers.
5. Re-run the FP32 check and unit tests after every kept change.

## 2. What paid off (Qwen3-0.6B, M4, vs mlx-lm)

| Gap | Change | Effect |
|---|---|---|
| TTFT = L−1 prefix + 1-token forward + two syncs + allocator clear | One prefill pass, logits for last row only, last layer only for the last row, no cache clear, async chunk submission | Main source of short-prompt TTFT gain |
| Small-M GEMM (M ≤ ~56) under-fills the GPU | Skinny MMA GEMM, per-shape tuned table, used only where faster | L=16 prefill 22 → 17 ms |
| Decode GEMV at 74–90 GB/s | Own BF16 GEMV: 16-byte vector loads, RMSNorm prologue, residual / SwiGLU epilogue | ~95 GB/s; decode ~6–7% faster |
| ~17 kernels per decode layer | 7 kernels: fused qk-norm+RoPE, one merged K/V write, packed QKV and gate/up | part of the decode gain |
| KV grown by concatenation | Preallocate for the request | long-context decode |
| Attention at 71–77% of MMA peak | GQA-paired flash attention (two query heads share K/V tiles), skip O-rescale when the row max is unchanged (exact) | +4.5% on attention |
| Element-wise passes and transposes in prefill | Fused qk-norm+RoPE writing attention-native layout, residual+RMSNorm fusion, compiled SwiGLU | ~50 ms at L=2048 |

## 3. What did not, and why

- **CPU + GPU split GEMM**: per-GEMM handoffs cost 1–2 ms through framework streams and ~1 ms with direct
  shared-memory writes; under sustained load the shared power budget erased the burst gain. Only consider
  coarse-grained splits at long context, and measure sustained.
- **Attention variants** (K transposed in threadgroup memory, V pre-converted, larger tiles): within noise.
- **Split-K skinny GEMM** for small-N shapes: reduction overhead exceeded the gain.
- **"Softmax ALU is the missing cost"**: removing exp changed nothing, removing softmax made the kernel
  slower — refuted.

## 4. Experimental pitfalls that produced wrong numbers

- **Unrolled register arrays.** Arrays of simdgroup matrices indexed in loops spilled to the stack; the first
  attention kernel was 10× slow. Fully unroll (`_Pragma("clang loop unroll(full)")`).
- **Fast-math UB deletes work.** A "no softmax" variant left a normalizer at 0; `1/0` under fast-math let
  the compiler delete the kernel body and the variant "ran" at 581 TFLOP/s. Before timing any variant,
  check it computes what you think (compare with a reference on a small input).
- **Source-cutting scripts.** Variants built by string-cutting kernel source cut the wrong span after a
  template substitution changed the marker text. Cut the template before substitution and assert the markers.
- **Same-process A/B** loops share allocator caches, compiled kernels and GPU state, and the default-stream
  `synchronize()` may not wait for the engine's stream. Fine for screening; never as final evidence (phase 5).
- **Heat.** Back-to-back screening runs throttle; gate on thermal state between variants and interleave.
- **Pick-the-best tuning tables** are noisy; require a margin (e.g. ≥ 5%) before switching implementations.
