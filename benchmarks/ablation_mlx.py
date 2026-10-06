"""Ablation of mlx-lm on Qwen3-0.6B for the gap decomposition (plan item 16).

Telescoping variants on the real weights, each timed in a pipelined loop
(graph for step i+1 built while the GPU runs step i, one wait at the end), so
differences are measured, not modeled:

decode at context c (KV cache filled with random values, offset c):
  gemv        only the 197 matmuls, with the decode dependency chain
  gemv_attn   + scaled_dot_product_attention over the c cached keys
  forward     the full model step + argmax fed back as the next token
  generate    mlx-lm stream_generate from the same cache (sampling, sync, loop)

prefill of L tokens as mlx-lm does it: the first L - 1 tokens in chunks of at
most 2048 (cache state evaluated, allocator cache cleared after each), then
the last token alone, which produces the logits:
  gemm        only the matmuls, per chunk
  gemm_attn   + causal attention over the keys so far
  forward     the full model on each chunk, then logits and argmax of the last row
  generate    time to first token of stream_generate

    python benchmarks/ablation_mlx.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

MODEL = "Qwen/Qwen3-0.6B"
CHUNK = 2048  # mlx-lm prefill_step_size default
HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128


def prefill_chunks(L: int) -> list[tuple[int, int]]:
    """(start, size) as mlx-lm's generate_step: L - 1 tokens in <= CHUNK pieces, then 1."""
    out = [(s, min(CHUNK, L - 1 - s)) for s in range(0, L - 1, CHUNK)]
    return out + [(L - 1, 1)]


def weights(model) -> tuple[list[dict], mx.array]:
    layers = []
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        layers.append({"q": a.q_proj.weight, "k": a.k_proj.weight, "v": a.v_proj.weight, "o": a.o_proj.weight,
                       "gate": m.gate_proj.weight, "up": m.up_proj.weight, "down": m.down_proj.weight})
    return layers, model.model.embed_tokens.weight


def random_kv(ctx: int, extra: int, seed: int) -> list[tuple[mx.array, mx.array]]:
    n = ((ctx + extra) // 256 + 1) * 256
    keys = mx.random.split(mx.random.key(seed), 56)
    out = []
    for i in range(28):
        ka, kb = keys[2 * i], keys[2 * i + 1]
        out.append((mx.random.normal((1, KV_HEADS, n, HEAD_DIM), key=ka).astype(mx.bfloat16),
                    mx.random.normal((1, KV_HEADS, n, HEAD_DIM), key=kb).astype(mx.bfloat16)))
    mx.eval(*[a for kv in out for a in kv])
    return out


def filled_cache(model, kv, ctx: int):
    cache = make_prompt_cache(model)
    for c, (k, v) in zip(cache, kv):
        c.keys, c.values, c.offset = mx.array(k), mx.array(v), ctx
    return cache


def pipelined(step, x0, steps: int) -> float:
    """Per-step time of x_{i+1} = step(x_i): async_eval every step, wait once."""
    x = x0
    for _ in range(3):
        x = step(x)
    mx.eval(x)
    t0 = time.perf_counter()
    for _ in range(steps):
        x = step(x)
        mx.async_eval(x)
    mx.eval(x)
    return (time.perf_counter() - t0) / steps


def decode_variants(model, tok, ctx: int, steps: int, rounds: int) -> dict:
    layers, emb = weights(model)
    kv = random_kv(ctx, steps + 16, seed=ctx)
    scale = HEAD_DIM**-0.5

    def gemv_step(x, attn: bool):
        outs = []
        for i, w in enumerate(layers):
            q, k, v = x @ w["q"].T, x @ w["k"].T, x @ w["v"].T
            outs += [k, v]
            if attn:
                K, V = kv[i]
                a = mx.fast.scaled_dot_product_attention(q.reshape(1, HEADS, 1, HEAD_DIM), K[:, :, :ctx],
                                                         V[:, :, :ctx], scale=scale)
                q = a.reshape(1, HEADS * HEAD_DIM)
            h = q @ w["o"].T
            g, u = h @ w["gate"].T, h @ w["up"].T
            outs.append(u)
            x = g @ w["down"].T
        logits = x @ emb.T
        return logits[:, : x.shape[-1]], outs

    def make(attn: bool):
        def step(x):
            nx, outs = gemv_step(x, attn)
            mx.async_eval(outs)
            return nx
        return step

    x0 = mx.random.normal((1, 1024)).astype(mx.bfloat16) * 0.01

    def forward() -> float:
        cache = filled_cache(model, kv, ctx)

        def fwd(t):
            logits = model(t, cache=cache)
            return mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)

        return pipelined(fwd, mx.array([[42]]), steps)

    def generate() -> float:
        cache = filled_cache(model, kv, ctx)
        stamps = []
        t0 = time.perf_counter()
        for _ in stream_generate(model, tok, prompt=[42], max_tokens=steps + 3, prompt_cache=cache):
            stamps.append(time.perf_counter() - t0)
        gaps = [b - a for a, b in zip(stamps[2:], stamps[3:])]
        return sum(gaps) / len(gaps)

    variants = {"gemv_s": lambda: pipelined(make(False), x0, steps),
                "gemv_attn_s": lambda: pipelined(make(True), x0, steps),
                "forward_s": forward, "generate_s": generate}
    samples = {k: [] for k in variants}
    for _ in range(rounds):  # interleaved, so drift (thermal, background) hits all variants alike
        for k, f in variants.items():
            samples[k].append(f())
    return {"ctx": ctx, **{k: min(v) for k, v in samples.items()}, "samples": samples}


def prefill_variants(model, tok, L: int, rounds: int) -> dict:
    layers, emb = weights(model)
    chunks = prefill_chunks(L)
    scale = HEAD_DIM**-0.5
    kv = random_kv(L, 0, seed=7 + L)

    def skeleton(attn: bool) -> float:
        def run():
            x = None
            for s, c in chunks:
                x = mx.random.normal((c, 1024)).astype(mx.bfloat16) * 0.01
                outs = []
                for i, w in enumerate(layers):
                    q, k, v = x @ w["q"].T, x @ w["k"].T, x @ w["v"].T
                    outs += [k, v]
                    if attn:
                        K, V = kv[i]
                        a = mx.fast.scaled_dot_product_attention(
                            q.reshape(1, c, HEADS, HEAD_DIM).transpose(0, 2, 1, 3), K[:, :, : s + c], V[:, :, : s + c],
                            scale=scale, mask="causal")
                        q = a.transpose(0, 2, 1, 3).reshape(c, HEADS * HEAD_DIM)
                    h = q @ w["o"].T
                    g, u = h @ w["gate"].T, h @ w["up"].T
                    outs.append(u)
                    x = g @ w["down"].T
                if (s, c) != chunks[-1]:
                    mx.eval(*outs)  # mlx-lm evaluates the cache state after every chunk
                    mx.clear_cache()
            mx.eval(mx.argmax(x[-1:] @ emb.T, axis=-1))
        run()
        t0 = time.perf_counter()
        run()
        return time.perf_counter() - t0

    prompt = mx.random.randint(1000, 150000, (L,), key=mx.random.key(L)).tolist()

    def forward() -> float:
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        for s, c in chunks[:-1]:
            model(mx.array([prompt[s: s + c]]), cache=cache)
            mx.eval([cc.state for cc in cache])
            mx.clear_cache()
        s, c = chunks[-1]
        logits = model(mx.array([prompt[s: s + c]]), cache=cache)
        mx.eval(mx.argmax(logits[:, -1, :], axis=-1))
        return time.perf_counter() - t0

    def generate() -> float:
        t0 = time.perf_counter()
        for _ in stream_generate(model, tok, prompt=prompt, max_tokens=1):
            return time.perf_counter() - t0

    forward()
    variants = {"gemm_s": lambda: skeleton(False), "gemm_attn_s": lambda: skeleton(True),
                "forward_s": forward, "generate_s": generate}
    samples = {k: [] for k in variants}
    for _ in range(rounds):
        for k, f in variants.items():
            samples[k].append(f())
    return {"L": L, **{k: min(v) for k, v in samples.items()}, "samples": samples}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", type=int, nargs="*", default=[512, 4096, 16384, 32768])
    p.add_argument("--lengths", type=int, nargs="*", default=[16, 64, 512, 1024, 4096, 16384, 32768])
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--rounds", type=int, default=3, help="interleaved rounds; min kept (1 for L > 4096)")
    p.add_argument("--out", default="results/raw")
    args = p.parse_args()

    load1 = os.getloadavg()[0]
    print(f"device: {mx.device_info()['device_name']}, load avg (1 min): {load1:.2f}")
    model, tok = load(MODEL)
    for _ in stream_generate(model, tok, prompt=list(range(1000, 1300)), max_tokens=8):
        pass

    dec = []
    print(f"\n{'ctx':>7}{'gemv':>9}{'+attn':>9}{'forward':>9}{'generate':>10}   (ms per token)")
    for c in args.contexts:
        r = decode_variants(model, tok, c, args.steps, args.rounds)
        dec.append(r)
        print(f"{c:>7}{r['gemv_s'] * 1e3:>9.2f}{r['gemv_attn_s'] * 1e3:>9.2f}{r['forward_s'] * 1e3:>9.2f}"
              f"{r['generate_s'] * 1e3:>10.2f}", flush=True)

    pre = []
    print(f"\n{'L':>7}{'gemm':>10}{'+attn':>10}{'forward':>10}{'generate':>10}   (ms)")
    for L in args.lengths:
        r = prefill_variants(model, tok, L, args.rounds if L <= 4096 else 1)
        pre.append(r)
        print(f"{L:>7}{r['gemm_s'] * 1e3:>10.1f}{r['gemm_attn_s'] * 1e3:>10.1f}{r['forward_s'] * 1e3:>10.1f}"
              f"{r['generate_s'] * 1e3:>10.1f}", flush=True)

    record = {
        "benchmark": "ablation_mlx_lm",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": mx.device_info()["device_name"],
        "mlx_version": mx.__version__, "mlx_lm_version": mlx_lm.__version__,
        "macos": platform.mac_ver()[0], "load_avg_1min": load1,
        "args": vars(args), "decode": dec, "prefill": pre,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ablation_mlx-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(record, indent=1))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
