"""Correctness check for engine/engine.py (the same criteria and threshold the final evaluation applies).

NOISE = 0.2241 is the evaluation's BF16 noise threshold: stock mlx-lm's p99 top-2-margin error
against FP32 on the evaluation gate's prompts (identical in every gate run with this MLX version).

1. API contract: Engine(model).stream, new_cache(capacity), forward(tokens, cache) -> [vocab],
   generate(prompt, max_tokens) yields exactly max_tokens ints.
2. Numerics vs an FP32 reference (same weights upcast), teacher-forced over the FP32 model's
   own greedy continuation, through the engine's forward path (whole prompt, then one token at a
   time) and through mlx-lm's path. The engine passes if
     - its median RMS logit error, p99 per-position max error and p99 top-2-margin error are each
       <= 1.1x mlx-lm's,
     - its top-1 disagreements with FP32 are not significantly more frequent (sign test p >= 0.05),
     - every top-1 disagreement is a near-tie: FP32 rates the engine's pick less than NOISE below
       FP32's own top token.
3. Generated tokens through generate(), compared with mlx-lm's: at the first position where they
   differ, FP32 must not rate the engine's token lower than mlx-lm's by more than NOISE. The
   evaluation applies this rule to every timed request; a scenario that breaks it scores as no
   speed-up. Prompts: two long ones (3000 and 9000 tokens, several prefill chunks) and the
   evaluation's own random-token prompts for every scenario up to 4096 tokens (4 per scenario,
   with the evaluation's generation lengths). `--full` adds its four 32768-token prompts (~10 min).
   The evaluation also uses natural-text prompts that are not available here.
Summation order may change (that is allowed); this checks the error stays at BF16-noise level.

    python tools/check.py            # ~4-6 min; prints PASS or FAIL with the numbers
    python tools/check.py --full     # also the 32k prompts; run it before your final commit
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

NOISE = 0.2241
GEN = 64
TEXTS = ["Explain the roofline model in two sentences.",
         "用三句话解释为什么 decode 阶段受内存带宽限制。",
         "Write a Python function that merges two sorted lists, with a docstring.",
         "Translate to French: The cache line is 128 bytes and the GPU has ten cores."]
# (scenario, prompt length, generated tokens); variants 0, 2, 4, 6 are the evaluation's random prompts
EVAL_SCENARIOS = [("ttft16", 16, 8), ("ttft64", 64, 8), ("ttft256", 256, 8), ("ttft1024", 1024, 8),
                  ("ctx512", 512, 129), ("ctx4096", 4096, 129), ("req256x256", 256, 256)]
EVAL_LONG = [("ctx32768", 32768, 129)]
EVAL_VARIANTS = (0, 2, 4, 6)


def eval_prompt(length, variant):
    """The evaluation's random-token prompt (agent-test/v3/prompts.py, kind "random")."""
    return mx.random.randint(1000, 150000, (length,), key=mx.random.key(1_000_003 * length + variant)).tolist()


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
    full = "--full" in sys.argv[1:]
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
        full_seq = mx.array(prompt + cont)[None]
        ref = model32(full_seq, cache=make_prompt_cache(model32))[0, L - 1:L - 1 + GEN].astype(mx.float32)
        # mlx-lm path
        cb = make_prompt_cache(model)
        if L > 1:
            mx.eval(model(full_seq[:, :L - 1], cache=cb))
        for j in range(GEN):
            B.append(errs(model(full_seq[:, L - 1 + j:L + j], cache=cb)[0, -1], ref[j]))
        # engine path
        ce = eng.new_cache(L + GEN + 1)
        E.append(errs(eng.forward(full_seq[:, :L], ce), ref[0]))
        for j in range(1, GEN):
            E.append(errs(eng.forward(full_seq[:, L - 1 + j:L + j], ce), ref[j]))
    only_e = sum((not e["top1"]) and b["top1"] for e, b in zip(E, B))
    only_b = sum((not b["top1"]) and e["top1"] for e, b in zip(E, B))
    n = only_e + only_b
    p_worse = sum(math.comb(n, k) for k in range(only_e, n + 1)) / 2 ** n if n else 1.0
    worst = max(x["regret"] for x in E)
    rows = [("median RMS", statistics.median(x["rms"] for x in E), statistics.median(x["rms"] for x in B)),
            ("p99 max", p99([x["max"] for x in E]), p99([x["max"] for x in B])),
            ("p99 top-2 margin", p99([x["margin"] for x in E]), p99([x["margin"] for x in B]))]
    ok = all(e <= 1.1 * b for _, e, b in rows) and p_worse >= 0.05 and worst < NOISE
    for name, e, b in rows:
        print(f"{name:18s} engine {e:.4f}  mlx-lm {b:.4f}  ratio {e / b:.3f}")
    print(f"top-1 disagreements only-engine {only_e}, only-mlx-lm {only_b}, sign-test p(engine worse) {p_worse:.3f}")
    print(f"largest FP32 regret of an engine pick {worst:.4f} (must be < NOISE {NOISE})")
    print(f"positions {len(E)}")

    # 3. generated tokens vs mlx-lm (first-difference FP32 rule)
    cases = [(f"long{L}", mx.random.randint(1000, 150000, (L,), key=mx.random.key(50 + L)).tolist(), 32)
             for L in (3000, 9000)]
    cases += [(f"{sid} v{v}", eval_prompt(L, v), g) for sid, L, g in EVAL_SCENARIOS + (EVAL_LONG if full else [])
              for v in EVAL_VARIANTS]
    exact = 0
    for name, prompt, g in cases:
        ref_toks = [t for t, _ in generate_step(mx.array(prompt), model, max_tokens=g)]
        got = list(eng.generate(prompt, g))
        if len(got) != g:
            ok = False
            print(f"{name}: generate yielded {len(got)} tokens, expected {g}: FAIL")
            continue
        j = next((i for i, (x, y) in enumerate(zip(ref_toks, got)) if x != y), None)
        if j is None:
            exact += 1
            continue
        lg32 = fp32_last_logits(model32, prompt + ref_toks[:j])
        gap = (lg32[ref_toks[j]] - lg32[got[j]]).item()
        good = gap <= NOISE
        ok &= good
        print(f"{name}: first difference at token {j}; FP32 rates the engine's token {gap:+.4f} below "
              f"mlx-lm's ({'within NOISE, ok' if good else 'beyond NOISE: FAIL'})")
    print(f"generated-token check: {exact} of {len(cases)} requests identical to mlx-lm"
          + ("" if full else " (32k prompts not included; use --full)"))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
