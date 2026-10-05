"""GPU DRAM bandwidth micro-benchmark (MLX custom Metal kernels).

Measures what the GPU alone can reach for:
  read   stream a buffer, keep one word per thread
  write  fill a buffer
  copy   read one buffer, write another
plus a read sweep over buffer sizes (shows where on-chip caches stop helping)
and the round-trip cost of launching a trivial kernel and waiting for it.

Kernels use grid-stride loops over 16-byte vectors. Writes need many stores
in flight: with 1024 threadgroups of 256 threads they reach only ~36 GB/s,
so write and copy launch 16384 groups. Read saturates at 4096.

The sweep repeats a buffer with separate launches in one mx.eval. Repeating
inside one kernel is wrong: only a few threadgroups are resident at a time,
each re-reads its own small slice, and caches look ~10x larger than they are.

Timing is wall-clock around mx.eval, so each sample includes one launch +
sync; large buffers keep that under ~1%.

    python benchmarks/bandwidth.py            # defaults: 1 GiB, 20 trials
    python benchmarks/bandwidth.py --size-mib 2048 --trials 50
    python benchmarks/bandwidth.py --sweep-min-kib 256 --sweep-max-mib 64 --sweep-per-octave 4
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
READ_GROUPS = 4096
WRITE_GROUPS = 16384
VEC_BYTES = 16  # uint4

# Grid-stride loops over uint4 vectors. NVEC is a template constant.
_read = mx.fast.metal_kernel(
    name="bw_read",
    input_names=["src"],
    output_names=["out"],
    source="""
        uint tid = thread_position_in_grid.x;
        uint stride = threads_per_grid.x;
        const device uint4* v = (const device uint4*)src;
        uint4 acc = uint4(0);
        for (uint i = tid; i < NVEC; i += stride)
            acc ^= v[i];
        out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
    """,
)

_write = mx.fast.metal_kernel(
    name="bw_write",
    input_names=["seed"],
    output_names=["dst"],
    source="""
        uint tid = thread_position_in_grid.x;
        uint stride = threads_per_grid.x;
        device uint4* v = (device uint4*)dst;
        uint s = seed[0];
        for (uint i = tid; i < NVEC; i += stride)
            v[i] = uint4(i ^ s);
    """,
)

_copy = mx.fast.metal_kernel(
    name="bw_copy",
    input_names=["src"],
    output_names=["dst"],
    source="""
        uint tid = thread_position_in_grid.x;
        uint stride = threads_per_grid.x;
        const device uint4* a = (const device uint4*)src;
        device uint4* b = (device uint4*)dst;
        for (uint i = tid; i < NVEC; i += stride)
            b[i] = a[i];
    """,
)

def _launch(groups: int) -> dict:
    return dict(grid=(THREADS_PER_GROUP * groups, 1, 1), threadgroup=(THREADS_PER_GROUP, 1, 1))


def read_groups(nbytes: int) -> int:
    """At least 16 vectors per thread, so the per-thread result stays small."""
    return max(1, min(READ_GROUPS, nbytes // VEC_BYTES // (THREADS_PER_GROUP * 16)))


def read_bytes(nbytes: int) -> int:
    """Traffic of one read launch: the buffer plus one result word per thread."""
    return nbytes + read_groups(nbytes) * THREADS_PER_GROUP * 4


def read_op(src: mx.array):
    groups = read_groups(src.nbytes)
    return lambda: _read(
        inputs=[src],
        template=[("NVEC", src.size * 4 // VEC_BYTES)],
        output_shapes=[(groups * THREADS_PER_GROUP,)],
        output_dtypes=[mx.uint32],
        **_launch(groups),
    )[0]


def write_op(n_words: int):
    seed = mx.array([7], dtype=mx.uint32)
    return lambda: _write(
        inputs=[seed],
        template=[("NVEC", n_words * 4 // VEC_BYTES)],
        output_shapes=[(n_words,)],
        output_dtypes=[mx.uint32],
        **_launch(WRITE_GROUPS),
    )[0]


def copy_op(src: mx.array):
    return lambda: _copy(
        inputs=[src],
        template=[("NVEC", src.size * 4 // VEC_BYTES)],
        output_shapes=[src.shape],
        output_dtypes=[mx.uint32],
        **_launch(WRITE_GROUPS),
    )[0]


def repeated(op, n: int):
    """n independent launches evaluated together (one sync)."""
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


def summarize(nbytes: float, samples: list[float]) -> dict:
    return {
        "bytes": nbytes,
        "best_GBps": nbytes / min(samples) / 1e9,
        "median_GBps": nbytes / statistics.median(samples) / 1e9,
        "best_s": min(samples),
        "median_s": statistics.median(samples),
        "trials": len(samples),
    }


def sweep_sizes(lo: int, hi: int, per_octave: int) -> list[int]:
    """Geometric sizes from lo to hi, rounded to 64 KiB (keeps whole uint4 vectors)."""
    sizes, i = [], 0
    while (size := round(lo * 2 ** (i / per_octave) / 2**16) * 2**16) <= hi:
        if size and (not sizes or size > sizes[-1]):
            sizes.append(size)
        i += 1
    return sizes


def make_buffer(nbytes: int) -> mx.array:
    x = mx.random.randint(0, 2**31, (nbytes // 4,), dtype=mx.uint32)
    mx.eval(x)
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size-mib", type=int, default=1024, help="buffer size for read/write/copy")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--sweep-bytes-per-eval-mib", type=int, default=256,
                   help="minimum bytes read per timed sample in the size sweep")
    p.add_argument("--sweep-min-kib", type=int, default=1024, help="smallest buffer in the size sweep")
    p.add_argument("--sweep-max-mib", type=int, default=None, help="largest buffer (default: --size-mib)")
    p.add_argument("--sweep-per-octave", type=int, default=1,
                   help="sizes per doubling; 4 or more to locate the on-chip cache capacity")
    p.add_argument("--out", default="results/raw")
    args = p.parse_args()

    load1 = os.getloadavg()[0]
    print(f"device: {mx.device_info()['device_name']}, load avg (1 min): {load1:.2f}")
    if load1 > 3:
        print("warning: system is busy; close other apps for cleaner numbers")

    nbytes = args.size_mib * 2**20
    src = make_buffer(nbytes)

    results: dict = {}
    results["read"] = summarize(read_bytes(nbytes), time_op(read_op(src), args.trials))
    results["write"] = summarize(nbytes, time_op(write_op(nbytes // 4), args.trials))
    results["copy"] = summarize(2 * nbytes, time_op(copy_op(src), args.trials))

    tiny = make_buffer(2**20)
    overhead = time_op(read_op(tiny), max(args.trials, 200), warmup=20)
    results["launch_sync_roundtrip_s"] = {
        "best": min(overhead),
        "median": statistics.median(overhead),
        "note": "trivial kernel (1 MiB read): launch + GPU run + CPU wait",
    }

    sweep = []
    del src
    for size in sweep_sizes(args.sweep_min_kib * 2**10, (args.sweep_max_mib or args.size_mib) * 2**20,
                            args.sweep_per_octave):
        buf = make_buffer(size)
        reps = max(1, args.sweep_bytes_per_eval_mib * 2**20 // size)
        op = repeated(read_op(buf), reps)
        r = summarize(read_bytes(size) * reps, time_op(op, max(5, args.trials // 2)))
        sweep.append({"buffer_MiB": size / 2**20, "reps": reps, **r})
        del buf
    results["read_sweep"] = sweep

    print(f"\n{'test':<8}{'best GB/s':>12}{'median GB/s':>14}")
    for name in ("read", "write", "copy"):
        r = results[name]
        print(f"{name:<8}{r['best_GBps']:>12.1f}{r['median_GBps']:>14.1f}")
    o = results["launch_sync_roundtrip_s"]
    print(f"\nlaunch + sync round trip: best {o['best'] * 1e6:.0f} us, median {o['median'] * 1e6:.0f} us")
    print(f"\n{'read sweep MiB':>15}{'best GB/s':>12}{'median GB/s':>14}")
    for r in sweep:
        print(f"{r['buffer_MiB']:>15.2f}{r['best_GBps']:>12.1f}{r['median_GBps']:>14.1f}")

    record = {
        "benchmark": "gpu_bandwidth",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": mx.device_info()["device_name"],
        "mlx_version": mx.__version__,
        "macos": platform.mac_ver()[0],
        "load_avg_1min": load1,
        "args": vars(args),
        "results": results,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"bandwidth-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
