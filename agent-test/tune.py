"""Diagnose forward latency for fusion ablations, with teacher-forced tokens."""
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from engine import Engine


def main():
    model, _ = load("Qwen/Qwen3-0.6B")
    engines = {f"pack={p},norm={n}": Engine(model, pack=p, fused_norm=n)
               for p, n in [(False, False), (True, False), (False, True), (True, True)]}
    prompt = mx.random.randint(1000, 150000, (1, 512), key=mx.random.key(10))
    x = mx.array([[42]])
    paths = {"mlx-lm-forward": (model, lambda c: model(x, cache=c))}
    for name, engine in engines.items():
        paths[name] = (engine, lambda c, e=engine: e.forward(x, c))
    results = {name: [] for name in paths}
    for rep in range(5):
        names = list(paths) if rep % 2 == 0 else list(reversed(paths))
        for name in names:
            obj, forward = paths[name]
            cache = make_prompt_cache(model) if name == "mlx-lm-forward" else obj.new_cache()
            y = model(prompt, cache=cache) if name == "mlx-lm-forward" else obj.forward(prompt, cache)
            mx.eval(y)
            for _ in range(3):
                mx.eval(forward(cache))
            times = []
            for _ in range(16):
                start = time.perf_counter()
                mx.eval(forward(cache))
                times.append(time.perf_counter() - start)
            results[name].append(statistics.median(times))
            print(rep, name, statistics.median(times) * 1000, flush=True)
    Path("agent-test/tuning.json").write_text(json.dumps(results, indent=2))
    print({name: statistics.median(values) * 1000 for name, values in results.items()})


if __name__ == "__main__":
    main()
