"""DRAM read bandwidth: CPU alone, GPU alone, and both at once.

Answers two questions for the device spec (design notes 6.1):
  - what can the CPU reach on its own (memory.engines.cpu bandwidth)
  - does the total cap when engines share DRAM (memory.total_bandwidth)

Each reader is its own process with its own buffer, so there is no GIL and
no shared cache lines. All processes wait for a common start time, stream
reads for --seconds, and report bytes read inside the window.

  cpu    N processes, each summing a private int32 buffer on MLX's CPU stream
         (the reduction is multithreaded: one process alone reaches ~65 GB/s)
  gpu    one process running the read kernel from bandwidth.py

    python benchmarks/contention.py
    python benchmarks/contention.py --cpu-procs 4 6 10 --seconds 5
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx


def cpu_reader(nbytes: int, start_at: float, seconds: float, q) -> None:
    buf = mx.ones((nbytes // 4,), dtype=mx.int32, stream=mx.cpu)
    for _ in range(2):  # fault in pages and warm up
        mx.eval(mx.sum(buf, stream=mx.cpu))
    while time.time() < start_at:
        pass
    done = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        mx.eval(mx.sum(buf, stream=mx.cpu))
        done += nbytes
    q.put(("cpu", done, time.perf_counter() - t0))


def gpu_reader(nbytes: int, start_at: float, seconds: float, q) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from bandwidth import make_buffer, read_bytes, read_op

    op = read_op(make_buffer(nbytes))
    for _ in range(3):
        mx.eval(op())
    while time.time() < start_at:
        time.sleep(0.001)
    done = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        mx.eval([op() for _ in range(4)])
        done += 4 * read_bytes(nbytes)
    q.put(("gpu", done, time.perf_counter() - t0))


def run(cpu_procs: int, gpu: bool, args) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    start_at = time.time() + 3.0 + 0.3 * cpu_procs  # time for spawn + buffer setup
    cpu_bytes = args.cpu_mib * 2**20
    procs = [ctx.Process(target=cpu_reader, args=(cpu_bytes, start_at, args.seconds, q)) for _ in range(cpu_procs)]
    if gpu:
        procs.append(ctx.Process(target=gpu_reader, args=(args.gpu_mib * 2**20, start_at, args.seconds, q)))
    for p in procs:
        p.start()
    reports = [q.get() for _ in procs]
    for p in procs:
        p.join()
    gbps = {"cpu": 0.0, "gpu": 0.0}
    for kind, done, elapsed in reports:
        gbps[kind] += done / elapsed / 1e9
    return {"cpu_procs": cpu_procs, "gpu": gpu, "cpu_GBps": gbps["cpu"], "gpu_GBps": gbps["gpu"],
            "total_GBps": gbps["cpu"] + gbps["gpu"]}


def best_of(n: int, cpu_procs: int, gpu: bool, args) -> dict:
    runs = [run(cpu_procs, gpu, args) for _ in range(n)]
    best = max(runs, key=lambda r: r["total_GBps"])
    return {**best, "median_total_GBps": statistics.median(r["total_GBps"] for r in runs), "runs": len(runs)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-procs", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--cpu-mib", type=int, default=512, help="buffer per CPU process")
    p.add_argument("--gpu-mib", type=int, default=1024)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="results/raw")
    args = p.parse_args()

    load1 = os.getloadavg()[0]
    print(f"load avg (1 min): {load1:.2f}")
    if load1 > 3:
        print("warning: system is busy; close other apps for cleaner numbers")

    rows = [best_of(args.repeats, 0, True, args)]
    for n in args.cpu_procs:
        rows.append(best_of(args.repeats, n, False, args))
    for n in args.cpu_procs:
        rows.append(best_of(args.repeats, n, True, args))

    print(f"\n{'cpu procs':>9}{'gpu':>5}{'cpu GB/s':>10}{'gpu GB/s':>10}{'total':>8}{'median':>8}")
    for r in rows:
        print(f"{r['cpu_procs']:>9}{'yes' if r['gpu'] else 'no':>5}{r['cpu_GBps']:>10.1f}"
              f"{r['gpu_GBps']:>10.1f}{r['total_GBps']:>8.1f}{r['median_total_GBps']:>8.1f}")

    record = {
        "benchmark": "dram_contention",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "macos": platform.mac_ver()[0],
        "mlx_version": mx.__version__,
        "load_avg_1min": load1,
        "args": vars(args),
        "results": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"contention-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
