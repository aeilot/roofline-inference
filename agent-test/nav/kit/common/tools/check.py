"""Correctness check for engine/engine.py (the same criteria the final evaluation applies).

1. API contract: Engine(model).stream, new_cache(capacity), forward(tokens, cache) -> [vocab],
   generate(prompt, max_tokens) yields exactly max_tokens ints.
2. Numerics vs an FP32 reference (same weights upcast), teacher-forced over the FP32 model's
   own greedy continuation, through the engine's forward path (whole prompt, then one token at a
   time) and through mlx-lm's path. "noise" = mlx-lm's p99 top-2-margin error. The engine passes if
     - its median RMS logit error, p99 per-position max error and p99 top-2-margin error are each
       <= 1.1x mlx-lm's,
     - its top-1 disagreements with FP32 are not significantly more frequent (sign test p >= 0.05),
     - every top-1 disagreement is a near-tie: FP32 rates the engine's pick at most `noise` below
       FP32's own top token.
3. Long prompts through generate() (chunked prefill, the evaluation's 4k/32k path): the engine's
   generated tokens are compared with mlx-lm's. At the first position where they differ, FP32
   must not rate the engine's token lower than mlx-lm's by more than `noise`. The evaluation
   applies this rule to every timed request (32k included); a scenario that breaks it does not count.
Summation order may change (that is allowed); this checks the error stays at BF16-noise level.
The evaluation uses other prompts with the same criteria: a result close to a limit may fail there.

    python tools/check.py            # ~2-5 min; prints PASS or FAIL with the numbers
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
LONG = (3000, 9000)     # > 2048: several prefill chunks
LONG_GEN = 32
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
                margin=abs(((lf[a] - lf[b]) - (ref[a] - ref[b])).item()), top1=pick == a,
                regret=(ref[a] - ref[pick]).item())


def p99(xs):
    xs = sorted(xs)
    return xs[int(0.99 * (len(xs) - 1))]


def fp32_last_logits(model32, seq):
    c = make_prompt_cache(model32)
    x = mx.array(seq)[None]
    for s in range(0, x.shape[1] - 1, 2048):
        model32(x[:, s:min(s + 2048, x.shape[1] - 1)], cache=c)
        mx.eval([e.state for e in c])
    return model32(x[:, -1:], cache=c)[0, -1]


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

    # 2. teacher-forced numerics
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
    noise = p99([x["margin"] for x in B])
    only_e = sum((not e["top1"]) and b["top1"] for e, b in zip(E, B))
    only_b = sum((not b["top1"]) and e["top1"] for e, b in zip(E, B))
    n = only_e + only_b
    p_worse = sum(math.comb(n, k) for k in range(only_e, n + 1)) / 2 ** n if n else 1.0
    worst = max(x["regret"] for x in E)
    rows = [("median RMS", statistics.median(x["rms"] for x in E), statistics.median(x["rms"] for x in B)),
            ("p99 max", p99([x["max"] for x in E]), p99([x["max"] for x in B])),
            ("p99 top-2 margin", p99([x["margin"] for x in E]), p99([x["margin"] for x in B]))]
    ok = all(e <= 1.1 * b for _, e, b in rows) and p_worse >= 0.05 and worst < noise
    for name, e, b in rows:
        print(f"{name:18s} engine {e:.4f}  mlx-lm {b:.4f}  ratio {e / b:.3f}")
    print(f"top-1 disagreements only-engine {only_e}, only-mlx-lm {only_b}, sign-test p(engine worse) {p_worse:.3f}")
    print(f"largest FP32 regret of an engine pick {worst:.4f} (must be < noise {noise:.4f})")
    print(f"positions {len(E)}")

    # 3. long prompts through generate()
    for L in LONG:
        prompt = mx.random.randint(1000, 150000, (L,), key=mx.random.key(50 + L)).tolist()
        ref_toks = [t for t, _ in generate_step(mx.array(prompt), model, max_tokens=LONG_GEN)]
        got = list(eng.generate(prompt, LONG_GEN))
        j = next((i for i, (x, y) in enumerate(zip(ref_toks, got)) if x != y), None)
        if len(got) != LONG_GEN:
            ok = False
            print(f"long prompt {L}: generate yielded {len(got)} tokens, expected {LONG_GEN}")
        elif j is None:
            print(f"long prompt {L}: generate matches mlx-lm on all {LONG_GEN} tokens")
        else:
            lg32 = fp32_last_logits(model32, prompt + ref_toks[:j])
            gap = (lg32[ref_toks[j]] - lg32[got[j]]).item()
            good = gap <= noise
            ok &= good
            print(f"long prompt {L}: first difference at token {j}; FP32 rates the engine's token {gap:+.4f} "
                  f"below mlx-lm's ({'near-tie, ok' if good else 'beyond noise: FAIL'})")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
