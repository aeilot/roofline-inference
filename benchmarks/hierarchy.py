"""GPU on-chip memory hierarchy micro-benchmark (MLX custom Metal kernels).

Measures what the multi-level bound (design notes 6.7, 6.8) needs below DRAM:

  shared-read sweep   every threadgroup streams the WHOLE buffer, starting at a
                      group-dependent offset. The working set is the buffer
                      size, shared by all cores, so the plateaus show the
                      per-core cache (L1), the shared on-chip cache (L2 + SLC)
                      and DRAM, and the knees show their capacities. Traffic
                      counted is bytes delivered to the cores.
  threadgroup         each threadgroup fills 16 KiB of threadgroup memory and
                      reads it repeatedly: on-core SRAM bandwidth.

One launch per sample, sized to run >= ~50 ms, so the launch + sync round
trip (~0.1 ms) stays under ~1%. Wall-clock timing around mx.eval.

Unlike bandwidth.py's sweep (separate launches per pass, each group reading a
slice), here one launch re-reads the buffer from every group: small buffers
are not dominated by launch overhead.

    python benchmarks/hierarchy.py
    python benchmarks/hierarchy.py --max-mib 128 --per-octave 2 --target-gib 4 --trials 9
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

THREADS = 256
VEC_BYTES = 16  # uint4

# Each group reads all NVEC vectors PASSES times in 16 KiB chunks (4 coalesced
# uint4 per thread), starting at a group-dependent chunk. NVEC % 1024 == 0.
_shared_read = mx.fast.metal_kernel(
    name="hier_shared_read",
    input_names=["src"],
    output_names=["out"],
    source="""
        uint t = thread_position_in_threadgroup.x;
        uint g = threadgroup_position_in_grid.x;
        const device uint4* v = (const device uint4*)src;
        uint4 a0 = uint4(0), a1 = uint4(0), a2 = uint4(0), a3 = uint4(0);
        for (uint p = 0; p < PASSES; ++p) {
            // Start moves every pass: otherwise a one-chunk buffer's loads are
            // pass-invariant and the compiler hoists them out of the loop.
            uint start = ((g * 977u + p) * 1024u) % NVEC;
            for (uint c = 0; c < NVEC; c += 1024u) {
                uint j = c + start;
                if (j >= NVEC) j -= NVEC;
                j += t;
                a0 ^= v[j]; a1 ^= v[j + 256u]; a2 ^= v[j + 512u]; a3 ^= v[j + 768u];
            }
        }
        uint4 acc = a0 ^ a1 ^ a2 ^ a3;
        out[thread_position_in_grid.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;
    """,
)

# 1024 uint4 = 16 KiB of threadgroup memory, read ITERS times per thread.
_tg_read = mx.fast.metal_kernel(
    name="hier_tg_read",
    input_names=["src"],
    output_names=["out"],
    source="""
        threadgroup uint4 buf[1024];
        uint t = thread_position_in_threadgroup.x;
        const device uint4* v = (const device uint4*)src;
        for (uint i = t; i < 1024u; i += 256u)
            buf[i] = v[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint4 acc = uint4(0);
        uint idx = t;
        for (uint k = 0; k < ITERS; ++k) {
            acc ^= buf[idx];
            idx = (idx + 257u) & 1023u;
        }
        out[thread_position_in_grid.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;
    """,
)


def time_op(op, trials: int, warmup: int = 2) -> list[float]:
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
        "trials": len(samples),
    }


def sweep_sizes(lo: int, hi: int, per_octave: int) -> list[int]:
    """Geometric sizes, rounded to 16 KiB (whole 1024-vector chunks)."""
    sizes, i = [], 0
    while (size := round(lo * 2 ** (i / per_octave) / 16384) * 16384) <= hi:
        if size and (not sizes or size > sizes[-1]):
            sizes.append(size)
        i += 1
    return sizes


def shared_read(size: int, groups: int, target_bytes: int, trials: int) -> dict:
    src = mx.random.randint(0, 2**31, (size // 4,), dtype=mx.uint32)
    mx.eval(src)
    passes = max(1, target_bytes // (size * groups))
    nbytes = size * groups * passes
    op = lambda: _shared_read(  # noqa: E731
        inputs=[src],
        template=[("NVEC", size // VEC_BYTES), ("PASSES", passes)],
        output_shapes=[(groups * THREADS,)],
        output_dtypes=[mx.uint32],
        grid=(groups * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
    )[0]
    return {"buffer_KiB": size / 1024, "groups": groups, "passes": passes, **summarize(nbytes, time_op(op, trials))}


def tg_read(groups: int, iters: int, trials: int) -> dict:
    src = mx.random.randint(0, 2**31, (4096,), dtype=mx.uint32)
    mx.eval(src)
    op = lambda: _tg_read(  # noqa: E731
        inputs=[src],
        template=[("ITERS", iters)],
        output_shapes=[(groups * THREADS,)],
        output_dtypes=[mx.uint32],
        grid=(groups * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
    )[0]
    nbytes = groups * THREADS * iters * VEC_BYTES
    return {"groups": groups, "iters": iters, **summarize(nbytes, time_op(op, trials))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-kib", type=int, default=32,
                   help="16 KiB is one chunk: start is always 0, loads are hoisted")
    p.add_argument("--max-mib", type=int, default=64)
    p.add_argument("--per-octave", type=int, default=2)
    p.add_argument("--groups", type=int, default=160, help="threadgroups in the shared-read sweep")
    p.add_argument("--target-gib", type=float, default=8.0, help="bytes delivered per sample")
    p.add_argument("--probe-groups", type=int, nargs="*", default=[10, 40],
                   help="extra sweeps up to --probe-max-kib with fewer groups (per-core cache)")
    p.add_argument("--probe-max-kib", type=int, default=1024)
    p.add_argument("--trials", type=int, default=7)
    p.add_argument("--out", default="results/raw")
    args = p.parse_args()

    load1 = os.getloadavg()[0]
    print(f"device: {mx.device_info()['device_name']}, load avg (1 min): {load1:.2f}")
    if load1 > 3:
        print("warning: system is busy; close other apps for cleaner numbers")

    target = int(args.target_gib * 2**30)
    sweep = []
    print(f"\n{'buffer KiB':>11}{'passes':>8}{'best GB/s':>11}{'median GB/s':>13}")
    for size in sweep_sizes(args.min_kib * 2**10, args.max_mib * 2**20, args.per_octave):
        r = shared_read(size, args.groups, target, args.trials)
        sweep.append(r)
        print(f"{r['buffer_KiB']:>11.0f}{r['passes']:>8}{r['best_GBps']:>11.1f}{r['median_GBps']:>13.1f}", flush=True)

    probes = {}
    for groups in args.probe_groups:
        rows = []
        print(f"\nprobe, {groups} groups\n{'buffer KiB':>11}{'best GB/s':>11}{'median GB/s':>13}")
        for size in sweep_sizes(args.min_kib * 2**10, args.probe_max_kib * 2**10, args.per_octave):
            r = shared_read(size, groups, target // 4, args.trials)
            rows.append(r)
            print(f"{r['buffer_KiB']:>11.0f}{r['best_GBps']:>11.1f}{r['median_GBps']:>13.1f}", flush=True)
        probes[str(groups)] = rows

    tg = []
    print(f"\n{'tg groups':>10}{'best GB/s':>11}{'median GB/s':>13}")
    for groups in (80, 160, 320, 640):
        r = tg_read(groups, iters=max(1, 4 * 2**30 // (groups * THREADS * VEC_BYTES)), trials=args.trials)
        tg.append(r)
        print(f"{groups:>10}{r['best_GBps']:>11.1f}{r['median_GBps']:>13.1f}", flush=True)

    record = {
        "benchmark": "gpu_hierarchy",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": mx.device_info()["device_name"],
        "mlx_version": mx.__version__,
        "macos": platform.mac_ver()[0],
        "load_avg_1min": load1,
        "args": vars(args),
        "results": {"shared_read_sweep": sweep, "probe_sweeps": probes, "threadgroup_read": tg},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"hierarchy-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
