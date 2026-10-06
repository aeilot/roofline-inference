"""Bandwidth and compute vs available parallelism (design notes 6.11, plan 16d).

The second axis of the capacity figure: how much concurrency an
implementation must keep to reach the roofs. By Little's law, a memory
system with latency L delivers at most

    B(n) = min(B_max, n * bytes_per_request / L)

with n requests in flight. Measured here:

  latency     one thread chasing pointers through a random cycle (one 128 B
              line per hop), for working sets from 16 KiB to 256 MiB
  bandwidth   read bandwidth with T concurrent threads, each keeping ILP
              independent 16-byte loads in flight (ILP = 1, 4), T = 32 .. 2^20,
              64 MiB buffer (beyond the on-chip caches)
  compute     FMA throughput with T concurrent threads (64 independent chains
              per thread, bf16), T = 32 .. 2^20

Each sample is sized to run >= 20 ms. Wall-clock around mx.eval.

    python benchmarks/concurrency.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np

LINE = 128  # bytes per hop in the pointer chase
VEC = 16

_chase = mx.fast.metal_kernel(
    name="conc_chase",
    input_names=["nxt"],
    output_names=["out"],
    source="""
        uint i = 0;
        for (uint s = 0; s < STEPS; ++s)
            i = nxt[i];
        out[0] = i;
    """,
)

_read = mx.fast.metal_kernel(
    name="conc_read",
    input_names=["src"],
    output_names=["out"],
    source="""
        uint tid = thread_position_in_grid.x;
        uint stride = threads_per_grid.x;
        const device uint4* v = (const device uint4*)src;
        uint4 acc[ILP];
        for (uint k = 0; k < ILP; ++k) acc[k] = uint4(0);
        for (uint p = 0; p < PASSES; ++p) {
            // The start moves every pass; otherwise, with one iteration per
            // thread, the loads are pass-invariant and get hoisted.
            uint off = (p * 7919u * 64u) % NVEC;
            for (uint i = tid; i + (ILP - 1) * stride < NVEC; i += ILP * stride) {
                #pragma unroll
                for (uint k = 0; k < ILP; ++k) {
                    uint j = i + k * stride + off;
                    acc[k] ^= v[j >= NVEC ? j - NVEC : j];
                }
            }
        }
        uint4 a = acc[0];
        for (uint k = 1; k < ILP; ++k) a ^= acc[k];
        out[tid] = a.x ^ a.y ^ a.z ^ a.w;
    """,
)

_fma = mx.fast.metal_kernel(
    name="conc_fma",
    input_names=["seed"],
    output_names=["out"],
    source="""
        uint tid = thread_position_in_grid.x;
        bfloat a = bfloat(seed[0]), b = bfloat(seed[1]);
        bfloat x[64];
        for (int c = 0; c < 64; ++c) x[c] = bfloat(tid + c) * bfloat(1e-3f);
        for (uint i = 0; i < ITERS; ++i) {
            #pragma unroll
            for (int c = 0; c < 64; ++c) x[c] = fma(x[c], a, b);
        }
        bfloat s = bfloat(0);
        for (int c = 0; c < 64; ++c) s += x[c];
        out[tid] = s;
    """,
)


def timed(fn, min_s: float = 0.02, trials: int = 3) -> float:
    mx.eval(fn())
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        mx.eval(fn())
        best = min(best, time.perf_counter() - t0)
    return best


def launch(T: int) -> dict:
    tg = min(T, 256)
    return {"grid": (T, 1, 1), "threadgroup": (tg, 1, 1)}


def latency(nbytes: int, steps: int, seed: int = 0) -> float:
    """Seconds per dependent load through a random cycle of nbytes / LINE lines."""
    n = nbytes // LINE
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    nxt = np.zeros(n * (LINE // 4), dtype=np.uint32)  # one used word per line
    stride = LINE // 4
    nxt[order * stride] = np.roll(order, -1) * stride
    buf = mx.array(nxt)
    mx.eval(buf)
    f = lambda: _chase(inputs=[buf], template=[("STEPS", steps)], output_shapes=[(1,)],  # noqa: E731
                       output_dtypes=[mx.uint32], grid=(1, 1, 1), threadgroup=(1, 1, 1))[0]
    return timed(f) / steps


def bandwidth(src: mx.array, T: int, ilp: int) -> float:
    nvec = src.nbytes // VEC
    passes = 1
    while True:
        f = lambda: _read(inputs=[src], template=[("NVEC", nvec), ("ILP", ilp), ("PASSES", passes)],  # noqa: E731
                          output_shapes=[(T,)], output_dtypes=[mx.uint32], **launch(T))[0]
        t = timed(f, trials=2)
        if t >= 0.02 or passes >= 64:
            return src.nbytes * passes / t
        passes *= max(2, int(0.03 / max(t, 1e-6)))
        passes = min(passes, 64)


def compute(T: int) -> float:
    seed = mx.array([0.999, 1e-4], dtype=mx.float32)
    iters = 64
    while True:
        f = lambda: _fma(inputs=[seed], template=[("ITERS", iters)], output_shapes=[(T,)],  # noqa: E731
                         output_dtypes=[mx.bfloat16], **launch(T))[0]
        t = timed(f, trials=2)
        if t >= 0.02 or iters >= 1 << 20:
            return T * 64 * iters * 2 / t
        iters *= max(2, int(0.03 / max(t, 1e-6)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-log2-threads", type=int, default=5)
    p.add_argument("--max-log2-threads", type=int, default=20)
    p.add_argument("--buffer-mib", type=int, default=64)
    p.add_argument("--out", default="results/raw")
    args = p.parse_args()

    load1 = os.getloadavg()[0]
    print(f"device: {mx.device_info()['device_name']}, load avg (1 min): {load1:.2f}")

    lat = []
    print(f"\n{'chase working set':>18}{'ns / load':>11}")
    for kib in (16, 128, 1024, 8192, 65536, 262144):
        steps = 200_000 if kib >= 8192 else 1_000_000
        s = latency(kib * 1024, steps)
        lat.append({"bytes": kib * 1024, "s_per_load": s})
        print(f"{kib:>15} KiB{s * 1e9:>11.1f}", flush=True)

    src = mx.random.randint(0, 2**31, (args.buffer_mib * 2**20 // 4,), dtype=mx.uint32)
    mx.eval(src)
    threads = [2**k for k in range(args.min_log2_threads, args.max_log2_threads + 1)]
    bw = {1: [], 4: []}
    comp = []
    print(f"\n{'threads':>9}{'GB/s ILP1':>11}{'GB/s ILP4':>11}{'TFLOP/s':>9}")
    for T in threads:
        b1, b4, c = bandwidth(src, T, 1), bandwidth(src, T, 4), compute(T)
        bw[1].append(b1)
        bw[4].append(b4)
        comp.append(c)
        print(f"{T:>9}{b1 / 1e9:>11.2f}{b4 / 1e9:>11.2f}{c / 1e12:>9.3f}", flush=True)

    record = {
        "benchmark": "concurrency",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": mx.device_info()["device_name"],
        "mlx_version": mx.__version__,
        "macos": platform.mac_ver()[0], "load_avg_1min": load1,
        "args": vars(args),
        "results": {"latency": lat, "threads": threads,
                    "bandwidth_ilp1": bw[1], "bandwidth_ilp4": bw[4], "compute_bf16": comp},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"concurrency-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(record, indent=1))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
