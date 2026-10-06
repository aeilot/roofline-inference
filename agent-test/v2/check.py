"""Numerical acceptance test against an FP32 reference (level A: sum order may change).

Teacher-forced over the FP32 model's own greedy continuation (neutral to both
engines), each engine runs its real path: mlx-lm = L-1 prefix + single-token
steps; v2 = whole-prompt prefill + fused decode steps. Logits at every
generated position are compared with the same weights upcast to FP32.

Acceptance (noise scale taken from mlx-lm's own BF16 error):
  * median RMS error, p99 per-position max error, and p99 error of the FP32
    top-2 logit margin are each <= 1.1x mlx-lm's;
  * top-1 disagreements with FP32 are not significantly more frequent
    (paired sign test, p >= 0.05) and every disagreement is within mlx-lm's
    p99 margin noise;
  * free-running greedy divergences between the engines occur only where the
    two chosen tokens are FP32 near-ties (gap < mlx-lm's p99 margin noise).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

from engine import Engine

TEXT_PROMPTS = [
    "List five prime numbers and explain how you know they are prime.",
    "Translate to French: The cache line is 128 bytes and the GPU has ten cores.",
    "Explain the roofline model in two sentences.",
    "用三句话解释为什么 decode 阶段受内存带宽限制。",
    "Write a Python function that merges two sorted lists, with a docstring.",
]


def errs(logits, ref):
    d = logits.astype(mx.float32) - ref
    pick = mx.argmax(logits).item()
    a, b = mx.argsort(ref)[-2:].tolist()[::-1]
    lf = logits.astype(mx.float32)
    margin_err = abs(((lf[a] - lf[b]) - (ref[a] - ref[b])).item())
    return dict(margin_err=margin_err, max=mx.max(mx.abs(d)).item(), rms=mx.sqrt(mx.mean(d * d)).item(),
                top1=pick == mx.argmax(ref).item(),
                fp32_regret=(mx.max(ref) - ref[pick]).item())  # 0 when top-1 agrees


def run(model, model32, engine, prompt, gen):
    seq_mlx = [t for t, _ in generate_step(mx.array(prompt), model, max_tokens=gen)]
    seq_v2 = list(engine.generate(prompt, max_tokens=gen))
    # Neutral teacher-forcing sequence: FP32 greedy continuation.
    seq = [t for t, _ in generate_step(mx.array(prompt), model32, max_tokens=gen)]
    full = mx.array(prompt + seq)[None]
    L = len(prompt)
    ref = model32(full, cache=make_prompt_cache(model32))[0, L - 1:L - 1 + gen].astype(mx.float32)
    mx.eval(ref)
    # mlx-lm path: L-1 prefix, then single tokens
    c = make_prompt_cache(model)
    if L > 1:
        mx.eval(model(full[:, :L - 1], cache=c))
    base = []
    for j in range(gen):
        base.append(model(full[:, L - 1 + j:L + j], cache=c)[0, -1])
        mx.eval(base[-1])
    # v2 path: whole-prompt prefill, then decode steps
    ce = engine.new_cache(L + gen + 1)
    ours = [engine.prefill(full[:, :L], ce, last=True)[0, -1]]
    for j in range(1, gen):
        ours.append(engine.decode(full[0, L - 1 + j:L + j].astype(mx.int32), ce, return_logits=True))
        mx.eval(ours[-1])
    eb = [errs(b, ref[j]) for j, b in enumerate(base)]
    eo = [errs(o, ref[j]) for j, o in enumerate(ours)]
    div = next((j for j, (a, b) in enumerate(zip(seq_mlx, seq_v2)) if a != b), None)
    info = None
    if div is not None:
        pre = mx.array(prompt + seq_mlx[:div])[None]
        r = model32(pre, cache=make_prompt_cache(model32))[0, -1]
        info = dict(position=div, mlx_token=seq_mlx[div], v2_token=seq_v2[div],
                    fp32_logit_mlx=r[seq_mlx[div]].item(), fp32_logit_v2=r[seq_v2[div]].item(),
                    fp32_gap_between_choices=abs(r[seq_mlx[div]] - r[seq_v2[div]]).item())
    summ = lambda e, k, f: f(x[k] for x in e)
    return dict(
        prompt_len=L,
        mlx_err_median_rms=summ(eb, "rms", statistics.median), v2_err_median_rms=summ(eo, "rms", statistics.median),
        mlx_err_max=summ(eb, "max", max), v2_err_max=summ(eo, "max", max),
        mlx_top1_vs_fp32=sum(x["top1"] for x in eb), v2_top1_vs_fp32=sum(x["top1"] for x in eo),
        positions=gen, sequences_equal=seq_mlx == seq_v2, divergence=info,
        _eb=eb, _eo=eo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 100, 512, 1024, 2048, 4096])
    p.add_argument("--gen", type=int, default=65)
    p.add_argument("--out", default="check.json")
    a = p.parse_args()
    model, tok = load("Qwen/Qwen3-0.6B")
    model32, _ = load("Qwen/Qwen3-0.6B")
    model32.set_dtype(mx.float32)
    engine = Engine(model)
    prompts = [mx.random.randint(1000, 150000, (L,), key=mx.random.key(seed)).tolist()
               for L in a.lengths for seed in (10, 11)]
    prompts += [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False) for t in TEXT_PROMPTS]
    rows, EB, EO = [], [], []
    for pr in prompts:
        r = run(model, model32, engine, pr, a.gen)
        EB += r.pop("_eb"); EO += r.pop("_eo")
        rows.append(r)
        print(json.dumps(r), flush=True)
    agg = dict(
        positions=len(EB),
        mlx_rms_median=statistics.median(x["rms"] for x in EB), v2_rms_median=statistics.median(x["rms"] for x in EO),
        mlx_max=max(x["max"] for x in EB), v2_max=max(x["max"] for x in EO),
        mlx_top1_mismatch=sum(not x["top1"] for x in EB), v2_top1_mismatch=sum(not x["top1"] for x in EO))
    p99 = lambda e, k: sorted(x[k] for x in e)[int(0.99 * (len(e) - 1))]
    agg.update(mlx_max_p99=p99(EB, "max"), v2_max_p99=p99(EO, "max"),
               mlx_margin_err_p99=p99(EB, "margin_err"), v2_margin_err_p99=p99(EO, "margin_err"))
    # Divergent choices must be FP32 near-ties: within mlx-lm's own BF16 top-2 margin noise (p99).
    ties = all(r["divergence"] is None or r["divergence"]["fp32_gap_between_choices"] < agg["mlx_margin_err_p99"]
               for r in rows)
    # Paired sign test on positions where exactly one engine disagrees with FP32 top-1.
    only_v2 = sum((not o["top1"]) and b["top1"] for b, o in zip(EB, EO))
    only_mlx = sum((not b["top1"]) and o["top1"] for b, o in zip(EB, EO))
    n = only_v2 + only_mlx
    p_worse = sum(math.comb(n, k) for k in range(only_v2, n + 1)) / 2 ** n if n else 1.0
    agg.update(only_v2_wrong=only_v2, only_mlx_wrong=only_mlx, sign_test_p_v2_worse=p_worse,
               v2_max_fp32_regret=max(x["fp32_regret"] for x in EO),
               mlx_max_fp32_regret=max(x["fp32_regret"] for x in EB))
    ok = (agg["v2_rms_median"] <= 1.1 * agg["mlx_rms_median"]
          and agg["v2_max_p99"] <= 1.1 * agg["mlx_max_p99"]
          and agg["v2_margin_err_p99"] <= 1.1 * agg["mlx_margin_err_p99"]
          and p_worse >= 0.05 and ties
          and agg["v2_max_fp32_regret"] < agg["mlx_margin_err_p99"])
    agg["divergences_are_fp32_near_ties"] = ties
    print(json.dumps(agg))
    Path(a.out).write_text(json.dumps(dict(all_pass=ok, aggregate=agg, rows=rows), indent=2))
    print("ALL PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
