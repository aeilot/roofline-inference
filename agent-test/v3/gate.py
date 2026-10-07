"""Correctness gate for every v3 engine (PLAN.md section 7).

Runs agent-test/v2's unit tests, then for each engine teacher-forces the FP32
model's own greedy continuation through the engine's real prefill/decode path
and compares logits with the FP32 reference. Criteria are v2 check.py's,
with E0 (mlx-lm) as the noise reference:
  median RMS, p99 max error, p99 top-2 margin error  <= 1.1x E0's
  top-1 disagreements not significantly more frequent than E0's (sign test p >= 0.05)
  every top-1 disagreement within E0's p99 margin noise
Writes gate.json: per-engine pass/fail and E0's p99 margin noise (the near-tie
threshold used later for generated-sequence divergences).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402
import prompts  # noqa: E402
from check import errs  # noqa: E402  (agent-test/v2/check.py)
from mlx_lm import load  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

# ~1500 positions: with the first draft's 624, the p99 statistics rested on the
# 8th-largest value and E1 (stock mlx-lm math) failed by 3% from noise alone.
GATE_PROMPTS = ([("random", L, v) for L in (16, 100, 512, 1024, 2048) for v in (0, 1, 2)]
                + [("text", L, v) for L in (64, 300, 1024, 2048) for v in (0, 1)])
GEN = 64


def p99(xs):
    xs = sorted(xs)
    return xs[int(0.99 * (len(xs) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--engines", nargs="+", default=adapters.ENGINES)
    a = ap.parse_args()
    result = dict(unit_tests=None, engines={})
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_engine.py"], cwd=adapters.V2,
                       capture_output=True, text=True)
    result["unit_tests"] = dict(returncode=t.returncode, tail=t.stdout[-600:])

    model, _ = load("Qwen/Qwen3-0.6B")
    model32, _ = load("Qwen/Qwen3-0.6B")
    model32.set_dtype(mx.float32)
    refs = []
    for kind, L, v in GATE_PROMPTS:
        prompt = prompts.make_prompt(kind, L, v)
        cont = [t for t, _ in generate_step(mx.array(prompt), model32, max_tokens=GEN)]
        full = mx.array(prompt + cont)[None]
        ref = model32(full, cache=make_prompt_cache(model32))[0, L - 1:L - 1 + GEN].astype(mx.float32)
        mx.eval(ref)
        refs.append((prompt, cont, ref))

    per = {}
    for name in a.engines:
        adapter = adapters.make(name, model)
        rows = []
        for prompt, cont, ref in refs:
            logits = adapter.teacher_forced(prompt, cont)
            rows += [errs(lg, ref[j]) for j, lg in enumerate(logits)]
        per[name] = rows
        del adapter

    base = per["E0"]
    noise = p99([x["margin_err"] for x in base])
    result["e0_margin_err_p99"] = noise
    for name, rows in per.items():
        only_x = sum((not o["top1"]) and b["top1"] for b, o in zip(base, rows))
        only_b = sum((not b["top1"]) and o["top1"] for b, o in zip(base, rows))
        n = only_x + only_b
        p_worse = sum(math.comb(n, k) for k in range(only_x, n + 1)) / 2 ** n if n else 1.0
        s = dict(positions=len(rows),
                 rms_median=statistics.median(x["rms"] for x in rows),
                 max_p99=p99([x["max"] for x in rows]),
                 margin_err_p99=p99([x["margin_err"] for x in rows]),
                 top1_mismatch=sum(not x["top1"] for x in rows),
                 max_fp32_regret=max(x["fp32_regret"] for x in rows),
                 sign_test_p_worse=p_worse)
        b = result["engines"].get("E0") or s
        s["pass"] = bool(s["rms_median"] <= 1.1 * b["rms_median"] and s["max_p99"] <= 1.1 * b["max_p99"]
                         and s["margin_err_p99"] <= 1.1 * b["margin_err_p99"] and p_worse >= 0.05
                         and s["max_fp32_regret"] < noise)
        result["engines"][name] = s
        print(name, json.dumps(s), flush=True)
    result["all_pass"] = result["unit_tests"]["returncode"] == 0 and all(
        s["pass"] for s in result["engines"].values())
    Path(a.out).write_text(json.dumps(result, indent=1))
    print("GATE", "PASS" if result["all_pass"] else "FAIL")


if __name__ == "__main__":
    main()
