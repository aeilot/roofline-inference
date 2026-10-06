"""Paired, thermally gated comparison of the v2 engine against mlx-lm generate_step.

Both sides: same process and BF16 weights, same prompt tokens, greedy, same
number of generated tokens, one token of async lookahead, token IDs only (no
detokenizer). mlx-lm runs its stock generate_step (prefill_step_size 2048,
L-1 prefix + 1, clear_cache between chunks); v2 runs its own scheduler.

Gating: before every measured request, wait until NSProcessInfo.thermalState
is nominal (0), then put the GPU in a defined state:
  --regime warm : run a fixed ~150 ms GPU workload immediately before the request
  --regime idle : sleep --idle seconds (GPU clocks down, as for a fresh request)
Order alternates AB/BA. Thermal state is recorded before and after each run.
Generated sequences are compared; divergences are recorded (judge them with
check.py: only FP32 near-ties are acceptable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import generate_step

import thermal
from engine import Engine

HERE = Path(__file__).parent


def measure(it):
    stamps, tokens = [], []
    start = time.perf_counter()
    for token in it:
        stamps.append(time.perf_counter() - start)
        tokens.append(token)
    yielded = stamps[-1]
    mx.synchronize()
    completed = time.perf_counter() - start
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    steady = gaps[2:] if len(gaps) > 4 else gaps
    return dict(tokens=tokens, ttft_s=stamps[0], request_yield_s=yielded,
                request_completed_s=completed,
                decode_s_median=statistics.median(steady) if steady else None,
                decode_gaps_s=gaps)


_W = None


def warm_gpu(ms=150):
    global _W
    if _W is None:
        _W = mx.random.normal((2048, 2048)).astype(mx.bfloat16)
        mx.eval(_W)
    s = time.perf_counter()
    x = _W
    while (time.perf_counter() - s) * 1000 < ms:
        x = (x @ _W) * 0.01
        mx.eval(x)


def boot(a, b):
    """Paired bootstrap 95% CI of the median of per-pair ratios."""
    import random
    rng = random.Random(0)
    r = [x / y for x, y in zip(a, b)]
    if len(r) < 4:
        return [min(r), max(r)]
    meds = sorted(statistics.median(rng.choice(r) for _ in r) for _ in range(5000))
    return [meds[125], meds[4875]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    p.add_argument("--gen", type=int, default=129)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--regime", choices=["warm", "idle"], default="warm")
    p.add_argument("--idle", type=float, default=2.0)
    p.add_argument("--out", default=str(HERE / "results-warm.json"))
    args = p.parse_args()

    model, _ = load("Qwen/Qwen3-0.6B")
    engine = Engine(model)
    src = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
           for f in ["benchmark.py", "engine.py", "kernels.py", "attention.py", "skinny_table.json"]}
    record = dict(settings=vars(args), timestamp=datetime.now(timezone.utc).isoformat(),
                  device=mx.device_info(), mlx_version=mx.__version__, mlx_lm_version=mlx_lm.__version__,
                  macos=platform.mac_ver()[0], source_sha256=src,
                  baseline="mlx_lm.generate.generate_step defaults (prefill_step_size=2048)",
                  engine="v2 (chunk 4096)", runs=[], summary=[])
    out = Path(args.out)

    for L in args.lengths:
        prompt = mx.random.randint(1000, 150000, (L,), key=mx.random.key(10)).tolist()
        factories = {
            "mlx-lm": lambda: (t for t, _ in generate_step(mx.array(prompt), model, max_tokens=args.gen)),
            "v2": lambda: engine.generate(prompt, max_tokens=args.gen),
        }
        for name in factories:      # warm compilation/allocator for this shape
            measure(factories[name]())
        pairs = []
        for rep in range(args.repeats):
            order = ["mlx-lm", "v2"] if rep % 2 == 0 else ["v2", "mlx-lm"]
            pair = {}
            for name in order:
                mx.synchronize()
                gate = thermal.wait_cool(idle=1.0)
                if args.regime == "warm":
                    warm_gpu()
                else:
                    time.sleep(args.idle)
                m = measure(factories[name]())
                m["gate"] = gate
                m["thermal_after"] = thermal.thermal_state()
                m["load_average"] = os.getloadavg()
                pair[name] = m
            row = dict(length=L, repeat=rep, order=order,
                       tokens_equal=pair["mlx-lm"]["tokens"] == pair["v2"]["tokens"], measurements=pair)
            record["runs"].append(row)
            pairs.append(pair)
            out.write_text(json.dumps(record, indent=1))
            print(f"L={L} rep={rep} TTFT {pair['mlx-lm']['ttft_s']*1e3:.1f} -> {pair['v2']['ttft_s']*1e3:.1f} ms, "
                  f"decode {pair['mlx-lm']['decode_s_median']*1e3:.2f} -> {pair['v2']['decode_s_median']*1e3:.2f} ms, "
                  f"equal={row['tokens_equal']}", flush=True)
        summary = dict(length=L, pairs=len(pairs),
                       tokens_equal_pairs=sum(r["tokens_equal"] for r in record["runs"] if r["length"] == L))
        for metric in ("ttft_s", "decode_s_median", "request_completed_s"):
            a = [pr["mlx-lm"][metric] for pr in pairs]
            b = [pr["v2"][metric] for pr in pairs]
            summary[metric] = dict(mlx_lm_median=statistics.median(a), v2_median=statistics.median(b),
                                   paired_speedup_median=statistics.median(x / y for x, y in zip(a, b)),
                                   paired_speedup_ci95=boot(a, b))
        record["summary"].append(summary)
        out.write_text(json.dumps(record, indent=1))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
