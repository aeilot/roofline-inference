"""Correctness check for engine/engine.py (the same check the final evaluation applies).

1. API contract: Engine(model).stream, new_cache(capacity), forward(tokens, cache) -> [vocab],
   generate(prompt, max_tokens) yields exactly max_tokens ints.
2. Numerics vs an FP32 reference (same weights upcast), teacher-forced over the FP32 model's
   own greedy continuation, through the engine's real path (forward of the whole prompt, then
   one token at a time) and through mlx-lm's path. The engine passes if its median RMS logit
   error, p99 per-position max error and p99 top-2-margin error are each <= 1.1x mlx-lm's, and
   its top-1 disagreements with FP32 are not significantly more frequent (sign test p >= 0.05).
   Summation order may change (that is allowed); this checks the error stays at BF16-noise level.

    python tools/check.py            # ~3-6 min; prints PASS or FAIL with the numbers
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from engine import Engine  # noqa: E402

GEN = 64
TEXTS = ["Explain the roofline model in two sentences.",
         "用三句话解释为什么 decode 阶段受内存带宽限制。",
         "Write a Python function that merges two sorted lists, with a docstring.",
         "Translate to French: The cache line is 128 bytes and the GPU has ten cores."]


def errs(logits, ref):
    lf = logits.astype(mx.float32)
    d = lf - ref
    pick = mx.argmax(lf).item()
    a, b = mx.argsort(ref)[-2:].tolist()[::-1]
    return dict(rms=mx.sqrt(mx.mean(d * d)).item(), max=mx.max(mx.abs(d)).item(),
                margin=abs(((lf[a] - lf[b]) - (ref[a] - ref[b])).item()), top1=pick == a)


def p99(xs):
    xs = sorted(xs)
    return xs[int(0.99 * (len(xs) - 1))]


def main():
    model, tok = load("Qwen/Qwen3-0.6B")
    model32, _ = load("Qwen/Qwen3-0.6B")
    model32.set_dtype(mx.float32)
    eng = Engine(model)

    # 1. contract
    assert isinstance(eng.stream, (mx.Stream,)) or hasattr(eng.stream, "device"), "Engine.stream missing"
    out = list(eng.generate([1000, 2000, 3000], 7))
    assert len(out) == 7 and all(isinstance(t, int) for t in out), f"generate must yield 7 ints, got {out}"
    c = eng.new_cache(16)
    lg = eng.forward(mx.array([[1000, 2000]]), c)
    assert lg.shape == (model.args.vocab_size,), f"forward must return [vocab], got {lg.shape}"

    prompts = [mx.random.randint(1000, 150000, (L,), key=mx.random.key(s)).tolist()
               for L in (16, 100, 512, 1024, 2048) for s in (0, 1, 2)]
    prompts += [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False) for t in TEXTS]
    E, B = [], []
    for prompt in prompts:
        L = len(prompt)
        cont = [t for t, _ in generate_step(mx.array(prompt), model32, max_tokens=GEN)]
        full = mx.array(prompt + cont)[None]
        ref = model32(full, cache=make_prompt_cache(model32))[0, L - 1:L - 1 + GEN].astype(mx.float32)
        # mlx-lm path
        cb = make_prompt_cache(model)
        if L > 1:
            mx.eval(model(full[:, :L - 1], cache=cb))
        for j in range(GEN):
            B.append(errs(model(full[:, L - 1 + j:L + j], cache=cb)[0, -1], ref[j]))
        # engine path
        ce = eng.new_cache(L + GEN + 1)
        E.append(errs(eng.forward(full[:, :L], ce), ref[0]))
        for j in range(1, GEN):
            E.append(errs(eng.forward(full[:, L - 1 + j:L + j], ce), ref[j]))
    only_e = sum((not e["top1"]) and b["top1"] for e, b in zip(E, B))
    only_b = sum((not b["top1"]) and e["top1"] for e, b in zip(E, B))
    n = only_e + only_b
    p_worse = sum(math.comb(n, k) for k in range(only_e, n + 1)) / 2 ** n if n else 1.0
    rows = [("median RMS", statistics.median(x["rms"] for x in E), statistics.median(x["rms"] for x in B)),
            ("p99 max", p99([x["max"] for x in E]), p99([x["max"] for x in B])),
            ("p99 top-2 margin", p99([x["margin"] for x in E]), p99([x["margin"] for x in B]))]
    ok = all(e <= 1.1 * b for _, e, b in rows) and p_worse >= 0.05
    for name, e, b in rows:
        print(f"{name:18s} engine {e:.4f}  mlx-lm {b:.4f}  ratio {e / b:.3f}")
    print(f"top-1 disagreements only-engine {only_e}, only-mlx-lm {only_b}, sign-test p(engine worse) {p_worse:.3f}")
    print(f"positions {len(E)}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
