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


def power_state():
    """(on AC power, last sleep time, last wake time). Sleep/DarkWake cycles update the
    kernel timestamps, so a change across a measurement means the system slept during it
    (smoke 2026-10-07: lid closed on battery -> repeated DarkWake, 32k TTFT 3x slower,
    while thermalState stayed 0)."""
    import subprocess
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    times = subprocess.run(["sysctl", "-n", "kern.sleeptime", "kern.waketime"], capture_output=True, text=True).stdout
    return "AC Power" in batt, times.strip()


def synchronize(adapter):
    if hasattr(adapter, "synchronize"):
        adapter.synchronize()
    else:
        mx.synchronize(adapter.stream)
        mx.synchronize()


def timed(adapter, prompt, gen):
    stamps, tokens = [], []
    start = time.perf_counter()
    for tok in adapter.generate(prompt, gen):
        stamps.append(time.perf_counter() - start)
        tokens.append(tok)
    if len(tokens) != gen:
        raise RuntimeError(f"Expected {gen} tokens, received {len(tokens)}")
    yielded = stamps[-1]
    synchronize(adapter)
    completed = time.perf_counter() - start
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return dict(tokens=tokens, ttft_s=stamps[0], request_yield_s=yielded, request_completed_s=completed,
                decode_s_median=statistics.median(gaps[2:]) if len(gaps) > 4 else None, decode_gaps_s=gaps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, choices=adapters.ENGINES + ["V0"] + list(adapters.candidates()))
    p.add_argument("--kind", required=True, choices=["random", "text"])
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--variant", type=int, required=True)
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--regime", required=True, choices=["idle", "warm"])
    p.add_argument("--probe-ref", type=float, required=True)
    p.add_argument("--gate-timeout", type=float, default=600)
    p.add_argument("--no-gate", action="store_true", help="later unit of a gated group: skip the thermal wait")
    p.add_argument("--warm-full", action="store_true", help="Warm the exact prompt length before timing")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rec = dict(engine=a.engine, kind=a.kind, length=a.length, variant=a.variant, gen=a.gen,
               regime=a.regime, pid=os.getpid(), process_start=T_START)
    prompt = prompts.make_prompt(a.kind, a.length, a.variant)
    from huggingface_hub import snapshot_download
    model_path = snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
    if a.engine == "V0":
        from vllm_adapter import V0
        adapter = V0(model_path)
    else:
        from mlx_lm import load
        model, _ = load(model_path)
        adapter = adapters.make(a.engine, model)
    from importlib.metadata import version
    rec["model_snapshot"] = Path(model_path).name
    rec["implementation"] = getattr(adapter, "identity", {
        "engine": "mlx-lm.generate_step" if a.engine == "E0" else a.engine,
        "mlx_lm": version("mlx-lm"), "mlx": version("mlx")})
    if a.engine == "V0":
        rec["implementation"].update(vllm=version("vllm"), vllm_metal=version("vllm-metal"))
    rec["prompt_sha256"] = prompts.prompt_sha256(prompt)

    # warm-up: compile kernels and populate this process's allocator; different content
    warm = prompts.make_prompt(a.kind, (a.length if a.warm_full else min(a.length, WARM_MAX_LEN)), a.variant + 1_000_000)
    for _ in adapter.generate(warm, min(a.gen, 8)):
        pass
    synchronize(adapter)

    if a.no_gate:
        gate = dict(skipped=True, thermal_state=thermal.thermal_state(), waited_s=0.0)
    else:
        gate = thermal.wait_cool(reference_ms=a.probe_ref, tol=PROBE_TOL, idle=1.0, timeout=a.gate_timeout)
    rec["gate"] = gate
    if gate.get("timeout"):
        rec["excluded"] = "gate_timeout"
    else:
        if a.regime == "idle":
            time.sleep(2.0)
        else:
            warm_gpu()
        ac0, t0 = power_state()
        mx.reset_peak_memory()
        rec["load_average_before"] = os.getloadavg()
        rec["t_measure_start"] = time.time()
        rec.update(timed(adapter, prompt, a.gen))
        ac1, t1 = power_state()
        rec["power"] = dict(ac_before=ac0, ac_after=ac1, sleep_wake_before=t0, sleep_wake_after=t1)
        if not (ac0 and ac1):
            rec["excluded"] = "not_on_ac_power"
        elif t0 != t1:
            rec["excluded"] = "system_slept_during_measurement"
        rec["peak_memory_bytes"] = mx.get_peak_memory()
        rec["peak_memory_scope"] = "MLX allocator only; excludes PyTorch and process RSS"
        rec["thermal_after"] = thermal.thermal_state()
        rec["load_average_after"] = os.getloadavg()
    tmp = Path(a.out + ".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(a.out)


if __name__ == "__main__":
    main()
