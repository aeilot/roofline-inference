# Resource reference lines and gap decomposition

## Model

For a given computation, time is at least the largest of the resource times:

    T >= max( W / P ,  Q_DRAM / B )

W = floating-point work the computation must do, P = compute rate; Q_DRAM = bytes that must cross
DRAM, B = bandwidth. Rates here are the calibrated values (P = 3.68 TFLOP/s BF16, B = 102.7 GB/s),
not certified ceilings — a better kernel or a hotter machine can move them.

Level A (your constraint): every product term of the *useful* computation must be computed at least
once; summation order, fusion, recomputation and removing unused work are free. For prefill the useful
outputs are the last token's logits and every layer's full K/V, so the final layer needs attention, O
projection and MLP only for the last row. Weights and KV stay BF16.

## Reference times (Qwen3-0.6B BF16, M4)

| Scenario | Reference | Binding resource | mlx-lm measured | mlx-lm / reference |
|---|---|---|---|---|
| Decode, context 512 | 12.0 ms/token | DRAM (1.19 GB weights + KV per token) | 13.17 ms | 1.10 |
| Decode, context 4k | 16.0 ms/token | DRAM | 17.42 ms | 1.09 |
| Decode, context 32k | 48.0 ms/token | DRAM (KV dominates) | 50.26 ms | 1.05 |
| Prefill 16 | 11.5 ms | DRAM (weights once) | 58.5 ms (TTFT) | 5.1 |
| Prefill 64 | 15.1 ms | compute | 69.5 ms | 4.6 |
| Prefill 256 | 61.4 ms | compute | 142.8 ms | 2.3 |
| Prefill 1024 | 269 ms | compute | 403.7 ms | 1.5 |
| Prefill 4096 | 1.45 s | compute | 1.90 s | 1.31 |
| Prefill 32768 | 39.9 s | compute (attention ~80%) | 53.8 s | 1.35 |

Decode per token: weights 1.19 GB / 102.7 GB/s = 11.6 ms, plus KV 112 KiB per cached token
(1.12 µs per token of context). Prefill: projections+MLP ≈ 0.88 GFLOP per token, causal attention
≈ 114 688 · L² FLOP; TTFT also includes one vocabulary projection (311 MB read) for the last row.

## Where mlx-lm loses time (ordered ablation, marginal effects in ms)

Variants timed on the real weights: matmuls only → + attention → full forward → full generate.
Each piece's loss = its measured increment minus its share of the reference. Ordered marginal
effects, not independent causes; the pieces overlap.

| Scenario | Reference | Measured | Matmul/GEMV loss | Attention loss | Small kernels loss | Engine/scheduling loss |
|---|---|---|---|---|---|---|
| Decode ctx 512 | 12.2 | 13.7 | 0.6 | 0.7 | 0.1 | 0.1 |
| Decode ctx 4k | 16.2 | 18.4 | 0.9 | 0.5 | 0.6 | 0.1 |
| Decode ctx 32k | 48.2 | 50.7 | 0.9 | 1.9 | 0.6 | −1.0 |
| Prefill 16 | 11.6 | 30.7 | 9.5 | 0.3 | 6.0 | 3.2 |
| Prefill 64 | 15.5 | 45.8 | 17.0 | 3.2 | 5.9 | 4.2 |
| Prefill 512 | 130.8 | 192.4 | 34.7 | 7.4 | 14.0 | 5.5 |
| Prefill 1024 | 277.9 | 382.9 | 56.4 | 17.5 | 17.0 | 14.2 |
| Prefill 4096 | 1504 | 1992 | 216 | 189 | 53 | 30 |
| Prefill 32768 | 41308 | 70692 | 2683 | 22479 | 3489 | 733 |

(The ablation used the historical full-graph reference and a warmer machine for the long prompts, so
its reference column differs slightly from the table above; 32k includes thermal throttling.)

Prefill rows measure a forward pass; the whole TTFT path of `generate` adds its own scheduling on top
(compare Prefill 16 = 30.7 ms here with TTFT 58.5 ms in BASELINE.md).

## What-if tool

`roofline_tool/` contains the calculator that produced these numbers.

    cd roofline_tool && python -m roofline --model configs/models/qwen3_0.6b.yaml decode --context 4096
    python -m roofline --model configs/models/qwen3_0.6b.yaml prefill --prompt-len 1024
    python -m roofline --model configs/models/qwen3_0.6b.yaml request --prompt-len 256 --gen 256
    python -m roofline --bound hierarchy --model configs/models/qwen3_0.6b.yaml prefill --prompt-len 4096

JSON output: per-resource time, binding resource, slack, provenance of each device number. You can
edit `configs/hardware/m4_mba_24g.yaml` (a copy) to ask "what if this rate were X".
