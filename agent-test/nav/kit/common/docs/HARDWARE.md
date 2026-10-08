# Hardware calibration — MacBook Air M4 (Mac16,13), 24 GB

Measured with custom Metal/MLX micro-benchmarks, 2026-10-04..07. Best observed values in short
windows ("burst"), cool machine, on AC power. These are calibrations, not guaranteed ceilings.

## GPU (10 cores, unified memory)

| Quantity | Value |
|---|---|
| DRAM read bandwidth, GPU alone | 102.7 GB/s (datasheet 120) |
| DRAM read, CPU alone / CPU + GPU together | 46–54 / 89.6–92.0 GB/s (both together never beat GPU alone) |
| BF16 compute, scalar FMA chain | 3.68 TFLOP/s |
| BF16 compute, register-only simdgroup matrix (MMA) loop | 3.44–3.56 TFLOP/s |
| On-chip L2 + SLC (read knee) | ~12 MiB; reads ~230 GB/s up to 2.4 MiB |
| DRAM latency | ~290 ns |
| Concurrency for 90% of bandwidth / compute | ~1k / ~32k threads |
| Dependent-kernel fixed cost (GPU) | 2.44 µs per kernel |
| CPU encode per kernel / commit per command buffer | 2.46 µs / 17.4 µs |
| GPU→CPU synchronization round trip | ~148 µs |

## MLX matmul throughput on Qwen3-0.6B shapes (BF16, [M, K] @ [N, K]^T, 28 layers' worth)

| M (rows) | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|
| TFLOP/s | 0.79 | 1.52 | 2.60 | 2.88 | 3.00 | 3.13 | 3.16 | 3.14 | 3.09 |

MLX causal attention (`mx.fast.scaled_dot_product_attention`, 2048 queries, 2k–32k keys): 2.5–2.8 TFLOP/s.
MLX decode attention (1 query, 4k–32k keys, 28 layers): 84–96 GB/s.

## CPU

4 performance + 6 efficiency cores; Accelerate SGEMM ~1.9 TFLOP/s FP32 in bursts. Under sustained
concurrent GPU load the CPU matrix throughput dropped to ~0.7 TFLOP/s (shared power budget).

## Measurement hygiene on this machine

- Fanless: sustained load throttles. 4k-token TTFT went from 1.9 s to 3.1 s after back-to-back runs.
  Compare engines only within one measurement session, interleaved, with cool-down between runs.
- `pmset -g therm` reports nothing useful here. `tools/platform_state.py` reads the OS thermal state
  (0 = nominal), AC power and sleep markers; `tools/bench.py` already waits for state 0.
- After ~1–2 s idle the GPU clocks down; the first request after idle pays ~30–40 ms extra (any engine).
- Keep the machine on AC power with the lid open; a closed lid forces sleep.
- MLX work is asynchronous: synchronize the right stream before reading a timer.
