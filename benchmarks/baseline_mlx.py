"""mlx-lm baseline for Qwen3-0.6B BF16 (plan item 15).

For each prompt length L: stream_generate (the engine's own path: chunked
prefill, async pipelining, greedy sampling) on L random tokens, generating
N tokens. Records
  TTFT(L)            time until the first token is yielded
  decode per token   gaps between later yields (context L+1 .. L+N)
  prompt / generation tok/s as reported by mlx-lm

Also records the structure of one decode step and one prefill chunk from the
lazy graph (mx.export_to_dot): GPU primitives by type, view-only primitives
excluded, and the CPU time to build a decode step's graph.

    python benchmarks/baseline_mlx.py
    python benchmarks/baseline_mlx.py --lengths 16 512 4096 --gen 33
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import re
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

MODEL = "Qwen/Qwen3-0.6B"
# Primitives that only change the view (strides, shape) and dispatch nothing.
VIEWS = {"Transpose", "Flatten", "Unflatten", "Reshape", "Broadcast", "Slice", "Squeeze",
         "ExpandDims", "AsStrided", "StopGradient", "Depends"}


def graph_primitives(*arrays: mx.array) -> collections.Counter:
    with tempfile.NamedTemporaryFile("r", suffix=".dot") as f:
        mx.export_to_dot(f.name, *arrays)
        labels = re.findall(r'label\s*=\s*"([A-Za-z_]+)', Path(f.name).read_text())
    return collections.Counter(labels)


def kernel_count(c: collections.Counter) -> int:
    return sum(n for k, n in c.items() if k not in VIEWS)


def random_prompt(n: int, seed: int) -> list[int]:
    key = mx.random.key(seed)
    return mx.random.randint(1000, 150000, (n,), key=key).tolist()


def structure(model, context: int, chunk: int) -> dict:
    """Primitive counts of one decode step at `context` and one prefill chunk."""
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([random_prompt(context, 1)]), cache=cache))
    builds = []
    for _ in range(20):
        t0 = time.perf_counter()
        y = model(mx.array([[42]]), cache=cache)
        nxt = mx.argmax(y[:, -1, :], axis=-1)
        builds.append(time.perf_counter() - t0)
        decode = graph_primitives(nxt)
        mx.eval(nxt)
    cache = make_prompt_cache(model)
    y = model(mx.array([random_prompt(chunk, 2)]), cache=cache)
    prefill = graph_primitives(y)
    return {
        "decode_context": context,
        "decode_primitives": dict(decode), "decode_kernels": kernel_count(decode),
        "decode_graph_build_s": statistics.median(builds),
        "prefill_chunk": chunk,
        "prefill_primitives": dict(prefill), "prefill_kernels": kernel_count(prefill),
    }


def run(model, tok, L: int, gen: int, seed: int) -> dict:
    prompt = random_prompt(L, seed)
    stamps, last = [], None
    t0 = time.perf_counter()
    for r in stream_generate(model, tok, prompt=prompt, max_tokens=gen):
        stamps.append(time.perf_counter() - t0)
        last = r
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    steady = gaps[2:] if len(gaps) > 4 else gaps
    return {
        "L": L, "gen": len(stamps),
        "ttft_s": stamps[0],
        "decode_s_median": statistics.median(steady),
        "decode_s_best": min(steady),
        "decode_gaps_s": gaps,
        "prompt_tps": last.prompt_tps, "generation_tps": last.generation_tps,
        "peak_memory_gb": last.peak_memory,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=int, nargs="*", default=[16, 64, 512, 1024, 4096, 16384, 32768])
    p.add_argument("--gen", type=int, default=65)
    p.add_argument("--repeats", type=int, default=2, help="runs per length (best TTFT kept)")
    p.add_argument("--out", default="results/raw")
    args = p.parse_args()

    load1 = os.getloadavg()[0]
    print(f"device: {mx.device_info()['device_name']}, load avg (1 min): {load1:.2f}, mlx-lm {mlx_lm.__version__}")
    model, tok = load(MODEL)
    for r in (run(model, tok, 128, 16, 0), run(model, tok, 1024, 8, 0)):  # warm-up: compile kernels
        pass

    s = structure(model, context=1000, chunk=2048)
    print(f"decode step: {s['decode_kernels']} GPU kernels, graph build {s['decode_graph_build_s'] * 1e3:.2f} ms; "
          f"prefill chunk {s['prefill_chunk']}: {s['prefill_kernels']} kernels")

    runs = []
    print(f"\n{'L':>7}{'TTFT ms':>11}{'decode ms':>11}{'prompt tok/s':>14}{'gen tok/s':>11}")
    for L in args.lengths:
        best = None
        for rep in range(args.repeats if L <= 4096 else 1):
            r = run(model, tok, L, args.gen, seed=10 + rep)
            if best is None or r["ttft_s"] < best["ttft_s"]:
                best = r
        runs.append(best)
        print(f"{L:>7}{best['ttft_s'] * 1e3:>11.1f}{best['decode_s_median'] * 1e3:>11.2f}"
              f"{best['prompt_tps']:>14.0f}{best['generation_tps']:>11.1f}", flush=True)

    record = {
        "benchmark": "baseline_mlx_lm",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": mx.device_info()["device_name"],
        "mlx_version": mx.__version__, "mlx_lm_version": mlx_lm.__version__,
        "macos": platform.mac_ver()[0], "load_avg_1min": load1,
        "model": MODEL, "args": vars(args),
        "structure": s, "runs": runs,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"baseline_mlx-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(record, indent=1))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
