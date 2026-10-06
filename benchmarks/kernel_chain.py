"""Dependent-kernel chain fit (design notes 5.3, plan item 11).

Measures the per-plan cost model for decode-like work (fit in
scripts/fit_kernel_chain.py):

    CPU   encode + commit:  K * (N * e + c_commit)        (time until the last
                                                           async_eval returns)
    GPU   per kernel:       g(b) = t0 + b / B_eff
    T(N, b, K) = pipeline of the two + c_sync

The GPU starts a command buffer when it is committed, so with K > 1 encoding
the next buffer overlaps executing the current one; with K = 1 they add up.

N dependent kernels per command buffer, each reading b bytes it has not read
before (weights), K command buffers per GPU->CPU sync. Each kernel reads one
word of its predecessor's output, so the GPU must finish kernel i before
kernel i+1 starts (a barrier, as in a decode layer chain).

Command buffers are controlled explicitly: MLX's automatic commit limits are
raised before import, every chain of N kernels is committed with
mx.async_eval, and the last one is waited on with mx.eval (one sync).

Buffers come from a pool of >= 128 MiB that the chains walk through, so every
read comes from DRAM even when N * b is small, as decode weights do. The
lazy graph is built before the clock starts; the timed part is encoding,
GPU execution and the final wait.

Also measured: N independent kernels in one command buffer (no data
dependency), to see whether the fixed cost shrinks when kernels can overlap.

    python benchmarks/kernel_chain.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "1000000")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402

THREADS = 256
VEC_BYTES = 16
POOL_BYTES = 256 * 2**20

_chain_read = mx.fast.metal_kernel(
    name="chain_read",
    input_names=["src", "dep"],
    output_names=["out"],
    source="""
        uint tid = thread_position_in_grid.x;
        uint stride = threads_per_grid.x;
        const device uint4* v = (const device uint4*)src;
        uint4 acc = uint4(dep[0]);
        for (uint i = tid; i < NVEC; i += stride)
            acc ^= v[i];
        out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
    """,
)


def groups_for(nbytes: int) -> int:
    """Enough threads to saturate DRAM for big reads, >= 4 vectors per thread."""
    return max(1, min(4096, nbytes // VEC_BYTES // (THREADS * 4)))


class Pool:
    """Distinct buffers of b bytes, walked round-robin so reads miss the caches."""

    def __init__(self, b: int):
        self.b = b
        n = max(4, POOL_BYTES // b)
        self.bufs = [mx.random.randint(0, 2**31, (b // 4,), dtype=mx.uint32) for _ in range(n)]
        mx.eval(*self.bufs)
        self.i = 0

    def next(self) -> mx.array:
        buf = self.bufs[self.i]
        self.i = (self.i + 1) % len(self.bufs)
        return buf


def read_kernel(src: mx.array, dep: mx.array) -> mx.array:
    groups = groups_for(src.nbytes)
    return _chain_read(
        inputs=[src, dep],
        template=[("NVEC", src.nbytes // VEC_BYTES)],
        output_shapes=[(groups * THREADS,)],
        output_dtypes=[mx.uint32],
        grid=(groups * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
    )[0]


def run_chain(pool: Pool, N: int, K: int, dep: mx.array) -> tuple[float, float]:
    """Build K command buffers of N dependent reads; return (time until all are
    encoded and committed, time until the GPU result is back)."""
    chains, x = [], dep
    for _ in range(K):
        for _ in range(N):
            x = read_kernel(pool.next(), x)
        chains.append(x)
    t0 = time.perf_counter()
    for y in chains:
        mx.async_eval(y)
    t1 = time.perf_counter()
    mx.eval(chains[-1])
    return t1 - t0, time.perf_counter() - t0


def run_independent(pool: Pool, N: int, dep: mx.array) -> float:
    outs = [read_kernel(pool.next(), dep) for _ in range(N)]
    t0 = time.perf_counter()
    mx.eval(*outs)
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes-kib", type=int, nargs="*", default=[64, 256, 1024, 4096, 16384, 65536])
    p.add_argument("--N", type=int, nargs="*", default=[1, 4, 16, 64, 256])
    p.add_argument("--K", type=int, nargs="*", default=[1, 4, 16])
    p.add_argument("--max-bytes-mib", type=int, default=4096, help="skip configs reading more per sample")
    p.add_argument("--trials", type=int, default=9)
    p.add_argument("--out", default="results/raw")
    args = p.parse_args()

    load1 = os.getloadavg()[0]
    print(f"device: {mx.device_info()['device_name']}, load avg (1 min): {load1:.2f}")
    dep = mx.zeros((1,), dtype=mx.uint32)
    mx.eval(dep)

    rows = []
    print(f"\n{'mode':<6}{'b KiB':>8}{'N':>5}{'K':>5}{'best ms':>10}{'median ms':>11}{'per kernel us':>15}")
    for kib in args.sizes_kib:
        b = kib * 1024
        pool = Pool(b)
        for _ in range(3):  # warm-up: compile, page in
            run_chain(pool, 4, 2, dep)
        for N in args.N:
            for K in args.K:
                if N * K * b > args.max_bytes_mib * 2**20:
                    continue
                runs = [run_chain(pool, N, K, dep) for _ in range(args.trials)]
                s = [r[1] for r in runs]
                rows.append({"mode": "chain", "b": b, "N": N, "K": K,
                             "best_s": min(s), "median_s": statistics.median(s), "samples": s,
                             "submit_s": [r[0] for r in runs]})
                print(f"{'chain':<6}{kib:>8}{N:>5}{K:>5}{min(s) * 1e3:>10.3f}{statistics.median(s) * 1e3:>11.3f}"
                      f"{min(s) / (N * K) * 1e6:>15.1f}", flush=True)
            if N > 1 and N * b <= args.max_bytes_mib * 2**20:
                s = [run_independent(pool, N, dep) for _ in range(args.trials)]
                rows.append({"mode": "indep", "b": b, "N": N, "K": 1,
                             "best_s": min(s), "median_s": statistics.median(s), "samples": s})
                print(f"{'indep':<6}{kib:>8}{N:>5}{1:>5}{min(s) * 1e3:>10.3f}{statistics.median(s) * 1e3:>11.3f}"
                      f"{min(s) / N * 1e6:>15.1f}", flush=True)
        del pool

    record = {
        "benchmark": "kernel_chain",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": mx.device_info()["device_name"],
        "mlx_version": mx.__version__,
        "macos": platform.mac_ver()[0],
        "load_avg_1min": load1,
        "args": vars(args),
        "results": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"kernel_chain-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(record, indent=1))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
