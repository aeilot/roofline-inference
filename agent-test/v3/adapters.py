"""Uniform interface over the engines compared in v3.

Every adapter exposes:
  stream                          the GPU stream its work runs on (synchronized at the end of timing)
  generate(prompt, max_tokens)    iterator of greedy token ids (one-token async lookahead)
  teacher_forced(prompt, cont)    logits at positions L-1 .. L-2+len(cont), through the engine's real
                                  prefill/decode path (used by the correctness gate)

E0  mlx-lm generate_step, default settings (the baseline users get)
E1  mlx-lm model forward with fair scheduling: whole-prompt prefill in 4096 chunks, last-row
    logits only, no clear_cache, KV preallocated, no logprobs  (separates scheduling from kernels)
E2  v2 engine (agent-test/v2, frozen)
E2-noskinny / E2-nogemv / E2-nogqa / E2-noprefuse   v2 with one optimization removed
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

V2 = Path(__file__).resolve().parents[1] / "v2"
sys.path.insert(0, str(V2))

import engine as v2engine  # noqa: E402  (agent-test/v2/engine.py)
from mlx_lm.generate import generate_step, generation_stream  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

ENGINES = ["E0", "E1", "E2", "E2-noskinny", "E2-nogemv", "E2-nogqa", "E2-noprefuse"]
CHUNK = 4096


class E0:
    """mlx-lm stock generate_step."""

    def __init__(self, model):
        self.model = model
        self.stream = generation_stream

    def generate(self, prompt, max_tokens):
        for tok, _ in generate_step(mx.array(prompt), self.model, max_tokens=max_tokens):
            yield tok

    def teacher_forced(self, prompt, cont):
        full = mx.array(prompt + cont)[None]
        L = len(prompt)
        c = make_prompt_cache(self.model)
        with mx.stream(self.stream):
            if L > 1:
                mx.eval(self.model(full[:, :L - 1], cache=c))
            out = []
            for j in range(len(cont)):
                out.append(self.model(full[:, L - 1 + j:L + j], cache=c)[0, -1])
                mx.eval(out[-1])
        return out


class E1:
    """mlx-lm model forward + the same scheduling choices as v2 (fair baseline)."""

    def __init__(self, model):
        self.model = model
        self.inner = model.model
        self.stream = mx.new_stream(mx.gpu)

    def _cache(self, capacity):
        cache = make_prompt_cache(self.model)
        for c in cache:
            c.step = (capacity + 255) // 256 * 256   # one allocation for the whole request
        return cache

    def _prefill(self, tokens, cache):
        """tokens [1, L]; returns logits [1, V] of the last row."""
        L = tokens.shape[1]
        for s in range(0, L, CHUNK):
            e = min(L, s + CHUNK)
            h = self.inner(tokens[:, s:e], cache=cache)          # final-normed hidden states
            if e < L:
                mx.async_eval([c.state for c in cache])
        return self.inner.embed_tokens.as_linear(h[:, -1:])[:, -1]

    def generate(self, prompt, max_tokens):
        if max_tokens == 0:
            return
        cache = self._cache(len(prompt) + max_tokens + 1)
        with mx.stream(self.stream):
            y = mx.argmax(self._prefill(mx.array(prompt)[None], cache), axis=-1)
            mx.async_eval(y)
            for _ in range(max_tokens):
                nxt = mx.argmax(self.model(y[None], cache=cache)[:, -1], axis=-1)
                mx.async_eval(nxt)
                yield y.item()
                y = nxt

    def teacher_forced(self, prompt, cont):
        full = mx.array(prompt + cont)[None]
        L = len(prompt)
        cache = self._cache(L + len(cont) + 1)
        with mx.stream(self.stream):
            out = [self._prefill(full[:, :L], cache)[0]]
            mx.eval(out[-1])
            for j in range(1, len(cont)):
                out.append(self.model(full[:, L - 1 + j:L + j], cache=cache)[0, -1])
                mx.eval(out[-1])
        return out


class _V2Adapter:
    def __init__(self, engine):
        self.engine = engine
        self.stream = v2engine.STREAM

    def generate(self, prompt, max_tokens):
        return self.engine.generate(prompt, max_tokens=max_tokens, chunk=CHUNK)

    def teacher_forced(self, prompt, cont):
        e = self.engine
        full = mx.array(prompt + cont)[None]
        L = len(prompt)
        cache = e.new_cache(L + len(cont) + 1)
        with mx.stream(self.stream):
            out = [e.prefill(full[:, :L], cache, last=True)[0, -1]]
            mx.eval(out[-1])
            for j in range(1, len(cont)):
                out.append(e.decode(full[0, L - 1 + j:L + j].astype(mx.int32), cache, return_logits=True))
                mx.eval(out[-1])
        return out


class _NoSkinny(v2engine.Engine):
    @staticmethod
    def matmul(x, W):
        return x @ W.T


class _NoGemv(v2engine.Engine):
    """Decode with stock MLX ops instead of the fused GEMV / qk-norm-RoPE kernels."""

    def decode(self, token, cache, *, return_logits=False):
        nq, nkv, hd = self.n_heads, self.n_kv, self.head_dim
        off = cache.offset
        h = self.embed[token]                                       # [1, hidden]
        for i, w in enumerate(self.layers):
            x = mx.fast.rms_norm(h, w["in_norm"], self.eps)
            q, k, v = mx.split(x @ w["qkv"].T, [nq * hd, (nq + nkv) * hd], axis=-1)
            q = mx.fast.rms_norm(q.reshape(1, 1, nq, hd), w["q_norm"], self.eps).transpose(0, 2, 1, 3)
            k = mx.fast.rms_norm(k.reshape(1, 1, nkv, hd), w["k_norm"], self.eps).transpose(0, 2, 1, 3)
            v = v.reshape(1, 1, nkv, hd).transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, hd, traditional=False, base=self.base, scale=1.0, offset=off)
            k = mx.fast.rope(k, hd, traditional=False, base=self.base, scale=1.0, offset=off)
            cache.update(i, mx.concatenate([k, v], axis=0))
            K, V = cache.views(i, off + 1)
            o = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.scale)
            h = h + o.reshape(1, -1) @ w["o"].T
            x = mx.fast.rms_norm(h, w["post_norm"], self.eps)
            gate, up = mx.split(x @ w["gu"].T, 2, axis=-1)
            h = h + v2engine.swiglu(gate, up) @ w["down"].T
        cache.offset += 1
        logits = (mx.fast.rms_norm(h, self.final_norm, self.eps) @ self.embed.T)[0]
        if return_logits:
            return logits
        return mx.argmax(logits, keepdims=True).astype(mx.int32)


class _NoPrefuse(v2engine.Engine):
    """Prefill with stock MLX ops (rms_norm, rope, causal SDPA, separate adds); keeps the
    single-pass schedule, last-row-only final layer and skinny GEMM."""

    def prefill(self, tokens, cache, *, last, all_logits=False):
        nq, nkv, hd = self.n_heads, self.n_kv, self.head_dim
        n = tokens.shape[1]
        off = cache.offset
        nl = len(self.layers)
        h = self.embed[tokens[0]]                                   # [n, hidden]
        for i, w in enumerate(self.layers):
            x = mx.fast.rms_norm(h, w["in_norm"], self.eps)
            q, k, v = mx.split(self.matmul(x, w["qkv"]), [nq * hd, (nq + nkv) * hd], axis=-1)
            q = mx.fast.rms_norm(q.reshape(n, nq, hd), w["q_norm"], self.eps).transpose(1, 0, 2)[None]
            k = mx.fast.rms_norm(k.reshape(n, nkv, hd), w["k_norm"], self.eps).transpose(1, 0, 2)[None]
            v = v.reshape(n, nkv, hd).transpose(1, 0, 2)[None]
            q = mx.fast.rope(q, hd, traditional=False, base=self.base, scale=1.0, offset=off)
            k = mx.fast.rope(k, hd, traditional=False, base=self.base, scale=1.0, offset=off)
            cache.update(i, mx.concatenate([k, v], axis=0))
            final = i == nl - 1
            if final and not last:
                break
            K, V = cache.views(i, off + n)
            mask = "causal"
            if final and not all_logits:
                q, h, mask = q[:, :, -1:], h[-1:], None
            o = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.scale, mask=mask)
            o = o[0].transpose(1, 0, 2).reshape(o.shape[2], -1)
            h = h + self.matmul(o, w["o"])
            x = mx.fast.rms_norm(h, w["post_norm"], self.eps)
            gate, up = mx.split(self.matmul(x, w["gu"]), 2, axis=-1)
            h = h + self.matmul(v2engine.swiglu(gate, up), w["down"])
        cache.offset += n
        if not last:
            return None
        h = mx.fast.rms_norm(h, self.final_norm, self.eps)
        if not all_logits:
            h = h[-1:]
        return (h @ self.embed.T)[None]


def candidates():
    """Navigation-experiment candidate engines: {name: directory with engine.py}, from the JSON
    file named by NAV_CANDIDATES (blinded names like C-1 ... C-4)."""
    import json
    import os
    f = os.environ.get("NAV_CANDIDATES")
    return json.loads(Path(f).read_text()) if f else {}


class Candidate:
    """An agent-built engine with the navigation-task API: Engine(model), .stream,
    new_cache(capacity), forward(tokens [1, n], cache) -> last-position logits [vocab],
    generate(prompt, max_tokens)."""

    def __init__(self, model, path):
        import importlib
        path = Path(path)
        # The candidate's own modules must win over agent-test/v2's (same names, e.g. engine, kernels).
        # Packages too: a candidate's kernels/ directory must not resolve to v2's kernels module.
        names = {p.stem for p in path.glob("*.py")} | {p.name for p in path.iterdir() if p.is_dir()} | {"engine"}
        for m in [m for m in sys.modules if m.split(".")[0] in names]:
            sys.modules.pop(m, None)
        sys.path.insert(0, str(path))
        self.engine = importlib.import_module("engine").Engine(model)
        self.stream = self.engine.stream

    def generate(self, prompt, max_tokens):
        return self.engine.generate(prompt, max_tokens)

    def teacher_forced(self, prompt, cont):
        e = self.engine
        full = mx.array(prompt + cont)[None]
        L = len(prompt)
        cache = e.new_cache(L + len(cont) + 1)
        out = [e.forward(full[:, :L], cache)]
        mx.eval(out[-1])
        for j in range(1, len(cont)):
            out.append(e.forward(full[:, L - 1 + j:L + j], cache))
            mx.eval(out[-1])
        return out


def make(name, model):
    if name == "E0":
        return E0(model)
    if name == "E1":
        return E1(model)
    if name in candidates():
        return Candidate(model, candidates()[name])
    cls = {"E2": v2engine.Engine, "E2-nogqa": v2engine.Engine, "E2-noskinny": _NoSkinny,
           "E2-nogemv": _NoGemv, "E2-noprefuse": _NoPrefuse}[name]
    kwargs = {"attn_impl": "mlx"} if name == "E2-nogqa" else {}
    return _V2Adapter(cls(model, **kwargs))
