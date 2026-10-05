"""GPU compute peak micro-benchmark (MLX).

Measures two numbers per dtype (fp32, fp16, bf16):
  fma    a custom Metal kernel of independent FMA chains, no memory traffic.
         This is the ALU peak, the number the physical bound should use.
  gemm   mx.matmul on square matrices. What a real library reaches today;
         a lower bound on achievable compute.

FMA kernel: each thread keeps CHAINS independent scalar accumulators and runs
ITERS rounds of x = fma(x, a, b). One FMA counts as 2 FLOP. Enough chains hide
ALU latency and loop overhead; best fp32/fp16/bf16 TFLOP/s by chain count:
8 -> 3.17/3.49/3.47, 16 -> 3.38/3.60/3.58, 64 -> 3.65/3.71/3.67; at 128 fp32
spills registers (0.30). Threadgroup count and size (1024-4096 groups of
256-1024) make no difference. The result is written out so the compiler
cannot drop the loop.

Timing is wall-clock around mx.eval; each sample runs long enough (>50 ms)
that launch + sync is noise.

    python benchmarks/compute.py
    python benchmarks/compute.py --trials 30 --gemm-sizes 2048 4096 8192
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx

THREADS_PER_GROUP = 256
GROUPS = 4096
CHAINS = 64
ITERS = 1024

DTYPES = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}

_fma = mx.fast.metal_kernel(
    name="peak_fma",
    input_names=["seed"],
    output_names=["out"],
    source="""
        uint tid = thread_position_in_grid.x;
        T a = T(seed[0]);
        T b = T(seed[1]);
        T x[CHAINS];
        for (int c = 0; c < CHAINS; ++c)
            x[c] = T(tid + c) * T(1e-3f);
        for (int i = 0; i < ITERS; ++i) {
            #pragma unroll
            for (int c = 0; c < CHAINS; ++c)
                x[c] = fma(x[c], a, b);
        }
        T s = T(0);
        for (int c = 0; c < CHAINS; ++c)
            s += x[c];
        out[tid] = s;
    """,
)


def fma_op(dtype):
    # a close to 1 and b close to 0 keep values finite in every dtype.
    seed = mx.array([0.999, 1e-4], dtype=mx.float32)
    n = THREADS_PER_GROUP * GROUPS
    return lambda: _fma(
        inputs=[seed],
        template=[("T", dtype), ("CHAINS", CHAINS), ("ITERS", ITERS)],
        output_shapes=[(n,)],
        output_dtypes=[dtype],
        grid=(n, 1, 1),
        threadgroup=(THREADS_PER_GROUP, 1, 1),
    )[0]


def fma_flops() -> float:
    return 2.0 * THREADS_PER_GROUP * GROUPS * CHAINS * ITERS


def repeated(op, n: int):
    return lambda: [op() for _ in range(n)]


def time_op(op, trials: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        mx.eval(op())
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        mx.eval(op())
        samples.append(time.perf_counter() - t0)
    return samples


def summarize(flops: float, samples: list[float]) -> dict:
    return {
        "flops": flops,
        "best_TFLOPs": flops / min(samples) / 1e12,
        "median_TFLOPs": flops / statistics.median(samples) / 1e12,
        "best_s": min(samples),
        "median_s": statistics.median(samples),
        "trials": len(samples),
    }


def reps_for(op, flops: float, target_s: float = 0.1) -> int:
    """Launches per sample so one sample takes about target_s."""
    s = min(time_op(op, 3, warmup=2))
    return max(1, round(target_s / s))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--gemm-sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--out", default="results/raw")
    args = p.parse_args()

    load1 = os.getloadavg()[0]
    print(f"device: {mx.device_info()['device_name']}, load avg (1 min): {load1:.2f}")
    if load1 > 3:
        print("warning: system is busy; close other apps for cleaner numbers")

    results: dict = {"fma": {}, "gemm": {}}
    for name, dt in DTYPES.items():
        op = fma_op(dt)
        reps = reps_for(op, fma_flops())
        results["fma"][name] = {"reps": reps, **summarize(fma_flops() * reps, time_op(repeated(op, reps), args.trials))}

    for name, dt in DTYPES.items():
        rows = []
        for n in args.gemm_sizes:
            a = mx.random.normal((n, n)).astype(dt)
            b = mx.random.normal((n, n)).astype(dt)
            mx.eval(a, b)
            op = lambda a=a, b=b: a @ b
            flops = 2.0 * n**3
            reps = reps_for(op, flops)
            rows.append({"n": n, "reps": reps, **summarize(flops * reps, time_op(repeated(op, reps), args.trials))})
            del a, b
        results["gemm"][name] = rows

    print(f"\n{'fma':<8}{'best TFLOP/s':>14}{'median':>10}")
    for name, r in results["fma"].items():
        print(f"{name:<8}{r['best_TFLOPs']:>14.2f}{r['median_TFLOPs']:>10.2f}")
    print(f"\n{'gemm':<8}{'n':>6}{'best TFLOP/s':>14}{'median':>10}")
    for name, rows in results["gemm"].items():
        for r in rows:
            print(f"{name:<8}{r['n']:>6}{r['best_TFLOPs']:>14.2f}{r['median_TFLOPs']:>10.2f}")

    record = {
        "benchmark": "gpu_compute",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": mx.device_info()["device_name"],
        "mlx_version": mx.__version__,
        "macos": platform.mac_ver()[0],
        "load_avg_1min": load1,
        "args": vars(args),
        "kernel": {"threads_per_group": THREADS_PER_GROUP, "groups": GROUPS, "chains": CHAINS, "iters": ITERS},
        "results": results,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"compute-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
