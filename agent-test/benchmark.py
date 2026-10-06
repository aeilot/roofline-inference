"""Controlled paired experiment against the installed mlx-lm generate_step.

Both sides return greedy token IDs, use fresh BF16 KVCache, the SAME input,
chunk size AND partition, stream, and number of lookahead steps. An explicit
--merge-prompt-tail tests the experimental alternate partition. Warm each shape, then
alternate AB/BA. Synchronize outstanding GPU work outside latency timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import generate_step, generation_stream
from mlx_lm.models.cache import make_prompt_cache
from mlx.utils import tree_flatten

from engine import Engine


def measure(factory):
    stamps, tokens = [], []
    start = time.perf_counter()
    iterator = factory()
    for token in iterator:
        stamps.append(time.perf_counter() - start)
        tokens.append(token)
    yielded = stamps[-1]
    mx.synchronize(generation_stream)
    completed = time.perf_counter() - start
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    # Same steady-state trimming as the repository baseline.
    steady = gaps[2:] if len(gaps) > 4 else gaps
    return dict(tokens=tokens, ttft_s=stamps[0], request_yield_s=yielded,
                request_completed_s=completed,
                decode_s_median=statistics.median(steady) if steady else None,
                decode_gaps_s=gaps)


def thermal_state():
    result = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                            text=True, timeout=5)
    return dict(returncode=result.returncode, output=result.stdout + result.stderr)


def check_forward(model, engine):
    """Teacher-forced all-position logits, chunk masks, and cache growth."""
    checks = []
    for chunks in ([16, 1, 1], [64, 17, 1], [255, 2, 1], [512, 17, 1]):
        ref_cache, opt_cache = make_prompt_cache(model), engine.new_cache()
        for n in chunks:
            x = mx.random.randint(1000, 150000, (1, n), key=mx.random.key(n + ref_cache[0].offset))
            a = model(x, cache=ref_cache)
            b = engine.forward(x, opt_cache, all_logits=True)
            mx.eval(a, b)
            diff = (a.astype(mx.float32) - b.astype(mx.float32))
            maximum = mx.max(mx.abs(diff)).item()
            rms = mx.sqrt(mx.mean(diff * diff)).item()
            equal = mx.all(mx.argmax(a, -1) == mx.argmax(b, -1)).item()
            check = dict(chunk=n, context=ref_cache[0].offset,
                         max_abs_logit_error=maximum, rms_logit_error=rms,
                         all_greedy_equal=equal)
            checks.append(check)
            # Packing only concatenates rows; do not accept silent quality loss.
            if maximum != 0 or not equal:
                raise AssertionError(f"Numerical correctness failed: {check}")
            assert [c.offset for c in ref_cache] == [c.offset for c in opt_cache]
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 64, 512, 4096])
    p.add_argument("--gen", type=int, default=65)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--cooldown", type=float, default=2.0)
    p.add_argument("--no-pack", action="store_true", help="projection fusion ablation")
    p.add_argument("--fused-norm", action="store_true", help="residual/norm fusion ablation")
    p.add_argument("--merge-prompt-tail", action="store_true", help="experimental alternate tail partition")
    p.add_argument("--out", default="agent-test/results.json")
    args = p.parse_args()
    if min(args.lengths) < 1 or args.gen < 6 or args.repeats < 2 or args.chunk < 1 or args.warmups < 1 or args.cooldown < 0:
        p.error("positive lengths/chunk, gen >= 6, repeats >= 2, warmups >= 1, cooldown >= 0 required")
    model, _ = load(args.model)
    params = tree_flatten(model.parameters())
    if any(w.dtype != mx.bfloat16 for _, w in params):
        raise ValueError("Every checkpoint parameter must be BF16")
    engine = Engine(model, pack=not args.no_pack, fused_norm=args.fused_norm)
    correctness = check_forward(model, engine)
    record = dict(settings=vars(args), timestamp=datetime.now(timezone.utc).isoformat(),
                  device=mx.device_info(),
                  mlx_version=mx.__version__, mlx_lm_version=mlx_lm.__version__,
                  macos=platform.mac_ver()[0], load_average=os.getloadavg(),
                  model_parameter_bytes=sum(w.nbytes for _, w in params),
                  weight_dtype="bfloat16", kv_dtype="bfloat16", sampling="greedy",
                  stream="shared generation_stream", correctness=correctness,
                  packed_weight_bytes=sum(w.nbytes for q, g, _, _ in engine.projections
                                          for w in (q, g) if w is not None),
                  prefill_partition=dict(baseline="chunks of L-1 prefix then 1",
                                        agent="chunks of L-1 prefix then 1" if not args.merge_prompt_tail
                                        else "chunks of full prompt, last chunk computes logits"),
                  note="Same process/resident models; thermal state observed, not externally locked",
                  source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__), Path(__file__).with_name("engine.py"),
                                              Path(__file__).with_name("kernels.py")]},
                  runs=[], summary=[])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for L in args.lengths:
        prompt = mx.random.randint(1000, 150000, (L,), key=mx.random.key(10)).tolist()
        digest = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        def baseline():
            for token, _ in generate_step(mx.array(prompt), model,
                                         max_tokens=args.gen, prefill_step_size=args.chunk):
                yield token
        def optimized():
            yield from engine.generate(prompt, max_tokens=args.gen,
                                       prefill_step_size=args.chunk,
                                       merge_prompt_tail=args.merge_prompt_tail)
        factories = {"mlx-lm": baseline, "agent": optimized}
        # Compilation and allocator state are warmed independently at each length.
        for _ in range(args.warmups):
            for name in factories:
                mx.synchronize(generation_stream)
                mx.clear_cache()
                measure(factories[name])
        pairs = []
        for rep in range(args.repeats):
            pair = {}
            order = ["mlx-lm", "agent"] if rep % 2 == 0 else ["agent", "mlx-lm"]
            for name in order:
                mx.synchronize(generation_stream)
                mx.clear_cache()
                time.sleep(args.cooldown)
                state = thermal_state()
                mx.reset_peak_memory()
                pair[name] = measure(factories[name])
                pair[name]["peak_memory_bytes"] = mx.get_peak_memory()
                pair[name]["thermal_before"] = state
                pair[name]["load_average_after"] = os.getloadavg()
            equal = pair["mlx-lm"]["tokens"] == pair["agent"]["tokens"]
            row = dict(length=L, repeat=rep, order=order, prompt_sha256=digest,
                       tokens_equal=equal, measurements=pair)
            record["runs"].append(row)
            out.write_text(json.dumps(record, indent=2))
            if not equal:
                raise AssertionError(f"Generated token mismatch at L={L}; recorded in {out}")
            pairs.append(pair)
            print(f"L={L} pair={rep} TTFT ms "
                  f"{pair['mlx-lm']['ttft_s']*1000:.2f} -> {pair['agent']['ttft_s']*1000:.2f}, "
                  f"decode ms {pair['mlx-lm']['decode_s_median']*1000:.2f} -> "
                  f"{pair['agent']['decode_s_median']*1000:.2f}", flush=True)
        summary = dict(length=L, tokens_equal=True)
        for metric in ("ttft_s", "decode_s_median", "request_yield_s", "request_completed_s"):
            a = statistics.median(pair["mlx-lm"][metric] for pair in pairs)
            b = statistics.median(pair["agent"][metric] for pair in pairs)
            ratios = [pair["mlx-lm"][metric] / pair["agent"][metric] for pair in pairs]
            summary[metric] = dict(baseline=a, agent=b, speedup=a / b,
                                   paired_speedup_median=statistics.median(ratios),
                                   paired_speedup_min=min(ratios), paired_speedup_max=max(ratios))
        record["summary"].append(summary)
        out.write_text(json.dumps(record, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
