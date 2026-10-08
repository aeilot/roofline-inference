"""Operation-level profile of the engine (in-process, synchronous timing).

Prefill at prompt length L: total engine.forward over the whole prompt, compared with the same
model's matrix multiplications alone (projections + MLP, 28 layers, MLX matmul) and causal
attention alone (MLX scaled_dot_product_attention). Decode at context C: one engine.forward
step, compared with reading every weight once (a GEMV per weight matrix) and attention alone.

    python tools/profile_ops.py --prefill 64 1024 4096 --decode 512 4096
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from engine import Engine  # noqa: E402


def t(f, n=5):
    f()
    ts = []
    for _ in range(n):
        mx.synchronize(); s = time.perf_counter(); f(); mx.synchronize()
        ts.append((time.perf_counter() - s) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", type=int, nargs="*", default=[64, 1024, 4096])
    ap.add_argument("--decode", type=int, nargs="*", default=[512, 4096])
    a = ap.parse_args()
    model, _ = load("Qwen/Qwen3-0.6B")
    eng = Engine(model)
    layers = model.layers
    mats = [(l.self_attn.q_proj.weight, l.self_attn.k_proj.weight, l.self_attn.v_proj.weight,
             l.self_attn.o_proj.weight, l.mlp.gate_proj.weight, l.mlp.up_proj.weight, l.mlp.down_proj.weight)
            for l in layers]
    H, KV, D = model.args.num_attention_heads, model.args.num_key_value_heads, model.args.head_dim
    for L in a.prefill:
        p = mx.random.randint(1000, 150000, (1, L), key=mx.random.key(1))
        xs = {w.shape[1]: mx.random.normal((L, w.shape[1])).astype(mx.bfloat16) for ws in mats[:1] for w in ws}
        mx.eval(p, *xs.values())
        total = t(lambda: mx.eval(eng.forward(p, eng.new_cache(L + 8))), n=3)
        gem = t(lambda: mx.eval([xs[w.shape[1]] @ w.T for ws in mats for w in ws]), n=3)
        q = mx.random.normal((1, H, L, D)).astype(mx.bfloat16); k = mx.random.normal((1, KV, L, D)).astype(mx.bfloat16)
        mx.eval(q, k)
        att = t(lambda: mx.eval([mx.fast.scaled_dot_product_attention(q, k, k, scale=D ** -0.5, mask="causal")
                                 for _ in layers]), n=3)
        print(f"prefill L={L}: engine {total:.1f} ms | matmuls alone {gem:.1f} | attention alone {att:.1f} "
              f"| remainder {total - gem - att:.1f}", flush=True)
    for C in a.decode:
        c = eng.new_cache(C + 16)
        mx.eval(eng.forward(mx.random.randint(1000, 150000, (1, C), key=mx.random.key(2)), c))
        y = mx.array([[5]])
        step = t(lambda: mx.eval(eng.forward(y, c)), n=8)
        x1 = {k: mx.random.normal((1, k)).astype(mx.bfloat16) for k in (1024, 2048, 3072)}
        mx.eval(*x1.values())
        emb = model.model.embed_tokens.weight
        gemv = t(lambda: mx.eval([x1[w.shape[1]] @ w.T for ws in mats for w in ws] + [x1[1024] @ emb.T]), n=5)
        kk = mx.random.normal((1, KV, C, D)).astype(mx.bfloat16); qq = mx.random.normal((1, H, 1, D)).astype(mx.bfloat16)
        mx.eval(kk, qq)
        att = t(lambda: mx.eval([mx.fast.scaled_dot_product_attention(qq, kk, kk, scale=D ** -0.5) for _ in layers]), n=5)
        wbytes = sum(w.nbytes for ws in mats for w in ws) + emb.nbytes
        print(f"decode ctx={C}: engine step {step:.2f} ms (synchronous) | all weights as GEMVs {gemv:.2f} "
              f"({wbytes / 1e9:.2f} GB) | attention alone {att:.2f}", flush=True)


if __name__ == "__main__":
    main()
