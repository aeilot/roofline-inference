"""One timed request in a fresh process (v3 protocol, PLAN.md sections 3-4).

load model -> warm-up request (same shape class, different content) -> thermal
gate -> GPU regime -> timed request -> synchronize the engine's stream ->
write one JSON record -> exit. Nothing is shared with other measurements.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

T_START = time.time()

import mlx.core as mx  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402  (also puts agent-test/v2 on sys.path)
import prompts  # noqa: E402
import thermal  # noqa: E402  (agent-test/v2/thermal.py)

WARM_MAX_LEN = 4096
# thermalState is the primary gate; the probe only catches gross slowdowns. With 5 %, a
# freshly loaded worker (probe ~3-4 % above the idle-process reference) waited ~1 min.
PROBE_TOL = 0.10


def warm_gpu(ms=150):
    a = mx.random.normal((2048, 2048)).astype(mx.bfloat16)
    mx.eval(a)
    x, s = a, time.perf_counter()
    while (time.perf_counter() - s) * 1000 < ms:
        x = (x @ a) * 0.01
        mx.eval(x)


def timed(adapter, prompt, gen):
    stamps, tokens = [], []
    start = time.perf_counter()
    for tok in adapter.generate(prompt, gen):
        stamps.append(time.perf_counter() - start)
        tokens.append(tok)
    yielded = stamps[-1]
    mx.synchronize(adapter.stream)      # the engine's own stream (v2 bug D2)
    mx.synchronize()
    completed = time.perf_counter() - start
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return dict(tokens=tokens, ttft_s=stamps[0], request_yield_s=yielded, request_completed_s=completed,
                decode_s_median=statistics.median(gaps[2:]) if len(gaps) > 4 else None, decode_gaps_s=gaps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, choices=adapters.ENGINES)
    p.add_argument("--kind", required=True, choices=["random", "text"])
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--variant", type=int, required=True)
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--regime", required=True, choices=["idle", "warm"])
    p.add_argument("--probe-ref", type=float, required=True)
    p.add_argument("--gate-timeout", type=float, default=600)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rec = dict(engine=a.engine, kind=a.kind, length=a.length, variant=a.variant, gen=a.gen,
               regime=a.regime, pid=os.getpid(), process_start=T_START)
    from mlx_lm import load
    model, _ = load("Qwen/Qwen3-0.6B")
    adapter = adapters.make(a.engine, model)
    prompt = prompts.make_prompt(a.kind, a.length, a.variant)
    rec["prompt_sha256"] = prompts.prompt_sha256(prompt)

    # warm-up: compile kernels and populate this process's allocator; different content
    warm = prompts.make_prompt(a.kind, min(a.length, WARM_MAX_LEN), a.variant + 1_000_000)
    for _ in adapter.generate(warm, min(a.gen, 8)):
        pass
    mx.synchronize(adapter.stream)
    mx.synchronize()

    gate = thermal.wait_cool(reference_ms=a.probe_ref, tol=PROBE_TOL, idle=1.0, timeout=a.gate_timeout)
    rec["gate"] = gate
    if gate.get("timeout"):
        rec["excluded"] = "gate_timeout"
    else:
        if a.regime == "idle":
            time.sleep(2.0)
        else:
            warm_gpu()
        mx.reset_peak_memory()
        rec["load_average_before"] = os.getloadavg()
        rec["t_measure_start"] = time.time()
        rec.update(timed(adapter, prompt, a.gen))
        rec["peak_memory_bytes"] = mx.get_peak_memory()
        rec["thermal_after"] = thermal.thermal_state()
        rec["load_average_after"] = os.getloadavg()
    tmp = Path(a.out + ".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(a.out)


if __name__ == "__main__":
    main()
