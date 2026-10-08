"""Process-isolated timing of engine/engine.py against stock mlx-lm (generate_step).

Each measurement runs in a fresh process with one engine loaded: load -> warm-up with a
different prompt of the same length -> wait until the machine is cool (thermal state nominal,
on AC) -> 2 s idle -> timed request. Engine order is randomized per repeat.

Scenarios (the same ones the final evaluation uses):
  ttft16 ttft64 ttft256 ttft1024   prompt N tokens, 8 generated   -> TTFT
  ctx512 ctx4096 ctx32768          prompt N tokens, 129 generated -> TTFT and decode ms/token
  req256x256                       prompt 256, 256 generated      -> whole-request time

    python tools/bench.py --scenarios ttft16 ctx4096 --repeats 3
Prints per-scenario medians and the ratio mlx-lm / engine (> 1 means the engine is faster).
Heat matters a lot on this fanless laptop: compare engines only within one bench call.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCEN = {"ttft16": (16, 8), "ttft64": (64, 8), "ttft256": (256, 8), "ttft1024": (1024, 8),
        "ctx512": (512, 129), "ctx4096": (4096, 129), "ctx32768": (32768, 129), "req256x256": (256, 256)}


def worker(engine, length, gen, variant):
    import mlx.core as mx
    from mlx_lm import load
    sys.path.insert(0, str(HERE))
    from platform_state import ready, snapshot
    model, _ = load("Qwen/Qwen3-0.6B")
    if engine == "mlx-lm":
        from mlx_lm.generate import generate_step, generation_stream
        gen_fn = lambda p, n: (t for t, _ in generate_step(mx.array(p), model, max_tokens=n))  # noqa: E731
        stream = generation_stream
    else:
        sys.path.insert(0, str(HERE.parent / "engine"))
        from engine import Engine
        eng = Engine(model)
        gen_fn, stream = eng.generate, eng.stream
    prompt = mx.random.randint(1000, 150000, (length,), key=mx.random.key(10_000 + variant)).tolist()
    warm = mx.random.randint(1000, 150000, (length,), key=mx.random.key(90_000 + variant)).tolist()
    for _ in gen_fn(warm, min(gen, 8)):
        pass
    mx.synchronize(stream); mx.synchronize()
    t0 = time.time()
    while not ready(snapshot()) and time.time() - t0 < 300:
        time.sleep(5)
    time.sleep(2)
    stamps, start = [], time.perf_counter()
    for _ in gen_fn(prompt, gen):
        stamps.append(time.perf_counter() - start)
    mx.synchronize(stream); mx.synchronize()
    done = time.perf_counter() - start
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    print(json.dumps(dict(ttft=stamps[0], decode=statistics.median(gaps[2:]) if len(gaps) > 4 else None,
                          request=done, n=len(stamps))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(SCEN), choices=list(SCEN))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--worker", nargs=4, metavar=("ENGINE", "LEN", "GEN", "VARIANT"))
    a = ap.parse_args()
    if a.worker:
        e, L, g, v = a.worker
        return worker(e, int(L), int(g), int(v))
    rows = []
    for sid in a.scenarios:
        L, g = SCEN[sid]
        res = {"mlx-lm": [], "engine": []}
        for r in range(a.repeats):
            order = ["mlx-lm", "engine"]
            random.shuffle(order)
            for e in order:
                out = subprocess.run([sys.executable, __file__, "--worker", e, str(L), str(g), str(r)],
                                     capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
                if out.returncode:
                    print(out.stderr[-2000:]); sys.exit(f"{e} failed on {sid}")
                rec = json.loads(out.stdout.strip().splitlines()[-1])
                if rec["n"] != g:
                    sys.exit(f"{e} yielded {rec['n']} tokens on {sid}, expected {g}")
                res[e].append(rec)
        line = [sid]
        for metric in ("ttft", "decode", "request"):
            if res["engine"][0][metric] is None:
                continue
            m0 = statistics.median(x[metric] for x in res["mlx-lm"])
            m1 = statistics.median(x[metric] for x in res["engine"])
            line.append(f"{metric} mlx-lm {m0 * 1e3:.2f} ms engine {m1 * 1e3:.2f} ms ratio {m0 / m1:.3f}")
        print(" | ".join(line), flush=True)
        rows.append(dict(scenario=sid, results=res))
    Path(HERE.parent / "bench_last.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
