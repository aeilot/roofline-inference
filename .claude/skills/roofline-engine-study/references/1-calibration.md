# Phase 1 — Device calibration

Goal: a device YAML (see `configs/hardware/m4_mba_24g.yaml`) where every number has `value`, `source`
(datasheet / estimate / system), `confidence`, and `measured` (filled by a benchmark). The reference lines
in phase 2 and the cost model in phase 3 read this file.

## What to measure, and with what

| Quantity | Why it matters | Reference benchmark |
|---|---|---|
| DRAM read bandwidth (GPU alone) | decode is bytes-bound | `benchmarks/bandwidth.py` (custom kernels, long windows, best of N) |
| CPU, GPU and both at once | shared memory: does adding the CPU add bandwidth? | `benchmarks/contention.py` |
| Matrix-unit peak per dtype | prefill GEMM and attention ceiling | `benchmarks/compute.py`, plus a register-only MMA loop (see below) |
| On-chip levels (capacity, bandwidth) | pebble / hierarchy bounds | `benchmarks/hierarchy.py` (size sweep, knee = capacity) |
| Concurrency to reach 90% of bandwidth / compute | why small GEMMs underperform | `benchmarks/concurrency.py` |
| Dependent-kernel fixed cost, CPU encode, commit, sync | per-plan cost model, many-small-kernel decode | `benchmarks/kernel_chain.py` + `scripts/fit_kernel_chain.py` |
| P(shape) for the model's real GEMM / attention shapes | prefill cost at small M | `benchmarks/compute_shapes.py` + `scripts/calibrate.py` |

Record the date, power source, thermal state, and library versions next to each measurement.

## Traps

**Burst best ≠ ceiling.** The best value from a short window is the right *calibration*, but it is not a
proven upper bound: the hardware may go faster with a better kernel (M4 datasheet 120 GB/s vs measured
102.7), or slower once hot. Always label outputs "calibrated reference, not certified ceiling" and keep a
sensitivity number (e.g. 120 GB/s would lower the decode line by ~14%).

**Measure the unit the workload uses.** A scalar FMA chain gave 3.68 TFLOP/s on M4; a register-only
simdgroup-MMA loop gave 3.44–3.56 (BF16). GEMM and attention run on the MMA path, so use that peak when
judging GEMM/attention efficiency. The MMA microbenchmark is ~20 lines: 8 independent accumulators,
`simdgroup_multiply_accumulate` in an unrolled loop, many threadgroups, count 2·512 FLOP per 8×8×8 MMA.
On NVIDIA, the equivalent is a WMMA/MMA loop or cuBLAS on large square GEMMs.

**Thermal and power state dominate.** On a fanless laptop, sustained load throttles: 4k TTFT 1.9 s → 3.1 s
after back-to-back runs; pure MMA does not throttle over 60 s while attention drops 12% (more data
movement, more power). Calibrate both a cold burst and a sustained (several-minute) regime and say which
one a reference line uses.

- macOS: `pmset -g therm` recorded nothing on the M4 Air — do not rely on it. `NSProcessInfo.thermalState`
  (0 nominal … 3 critical) works but is coarse. A fixed-workload probe must ramp clocks first (the GPU
  downclocks after a few seconds idle; a cold single shot measured 11–25 ms for the same 11 ms work).
- GPU idle downclock adds ~30–40 ms to the first request after 1–2 s idle, for every engine. This is real
  user-facing latency; report idle and warm regimes separately.
- Power source changes performance policy; laptops must be on AC, and a closed lid without an external
  display forces sleep regardless of `caffeinate`.

**CPU + GPU co-processing.** On unified-memory SoCs the CPU matrix unit (SME/AMX via Accelerate) reached
1.9 TFLOP/s FP32 and the burst combined GEMM throughput was 4.39 vs 3.1 GPU-only — but under sustained load
the shared power budget cut the CPU to 0.67 TFLOP/s and the GPU by 5%, and per-GEMM handoffs cost 1–2 ms.
Measure sustained combined throughput before planning on it.
