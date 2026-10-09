"""Qwen3 BF16 single-stream inference engine (starting point).

Own transformer forward and greedy generation loop on MLX, using mlx-lm only for
weights, RoPE and the KV cache. Behaves like mlx-lm's default generate path.

API contract (checked by tools/check.py; the final evaluation uses exactly this):

    engine = Engine(model)              # model from mlx_lm.load(...)
    engine.stream                       # the mx.Stream all engine GPU work runs on
    cache = engine.new_cache(capacity)  # capacity = max tokens the request will hold
    logits = engine.forward(tokens, cache)   # tokens: mx.array [1, n] int; returns logits of the
                                             # LAST position, shape [vocab]; advances the cache by n
    for tok in engine.generate(prompt, max_tokens):   # greedy; yields exactly max_tokens ints; ignores EOS
        ...
"""
from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
from mlx_lm.generate import generation_stream
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
try:
    from .norm_rope import norm_rope
except ImportError:
    from norm_rope import norm_rope


class Engine:
    def __init__(self, model):
        if model.model_type != "qwen3":
            raise ValueError("Only dense Qwen3 is supported")
        self.model = model
        self.args = model.args
        self.stream = generation_stream
        with mx.stream(self.stream):
            self.qkv = [mx.concatenate([l.self_attn.q_proj.weight,
                                        l.self_attn.k_proj.weight,
                                        l.self_attn.v_proj.weight]) for l in model.layers]
            self.gu = [mx.concatenate([l.mlp.gate_proj.weight, l.mlp.up_proj.weight])
                       for l in model.layers]
            mx.eval(self.qkv, self.gu)

    def new_cache(self, capacity: int | None = None):
        return [KVCache() for _ in self.model.layers]

    def _hidden(self, tokens, cache, last_only=False):
        h = self.model.model.embed_tokens(tokens)
        mask = create_attention_mask(h, cache[0])
        B, L, _ = h.shape
        offset = mx.array(cache[0].offset, dtype=mx.int32)
        for i, (layer, c) in enumerate(zip(self.model.layers, cache)):
            a, m = layer.self_attn, layer.mlp
            x = layer.input_layernorm(h)
            qsize = a.n_heads * self.args.head_dim
            kvsize = a.n_kv_heads * self.args.head_dim
            final = i == len(self.model.layers) - 1
            if final and L > 1:
                # Every key/value is needed by future calls, but earlier final-layer
                # queries and feed-forward outputs cannot affect the last logit.
                kv = x @ self.qkv[i][qsize:].T
                k, v = mx.split(kv, 2, axis=-1)
                if last_only:
                    q = a.q_proj(x[:, -1:, :])
                    h = h[:, -1:, :]
                    qlength = 1
                else:
                    q = None
            else:
                qkv = x @ self.qkv[i].T
                if L <= 64:
                    q, k, v = norm_rope(qkv, a.q_norm.weight, a.k_norm.weight, offset, self.args)
                else:
                    q, k, v = mx.split(qkv, [qsize, qsize + kvsize], axis=-1)
                qlength = L
            fused = L <= 64 and not (final and L > 1)
            if q is not None and not fused:
                q = a.q_norm(q.reshape(B, qlength, a.n_heads, -1)).transpose(0, 2, 1, 3)
                q = a.rope(q, offset=c.offset + L - qlength)
            if not fused:
                k = a.k_norm(k.reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
                v = v.reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
                k = a.rope(k, offset=c.offset)
            k, v = c.update_and_fetch(k, v)
            if q is None:
                return h
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale,
                                                   mask=None if qlength == 1 else mask)
            h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(B, qlength, -1))
            x = layer.post_attention_layernorm(h)
            gate, up = mx.split(x @ self.gu[i].T, 2, axis=-1)
            h = h + m.down_proj(swiglu(gate, up))
        return self.model.model.norm(h[:, -1:, :] if last_only else h)

    def forward(self, tokens, cache):
        """Logits [vocab] of the last position; advances the cache."""
        with mx.stream(self.stream):
            h = self._hidden(tokens, cache, last_only=True)
            return self.model.model.embed_tokens.as_linear(h[:, -1:, :])[0, 0]

    def generate(self, prompt: list[int], max_tokens: int, prefill_step_size: int = 4096) -> Iterator[int]:
        if not prompt or max_tokens < 0:
            raise ValueError("Nonempty prompt and nonnegative max_tokens required")
        if max_tokens == 0:
            return
        cache = self.new_cache(len(prompt) + max_tokens)
        with mx.stream(self.stream):
            rest = prompt
            while len(rest) > prefill_step_size:
                n = prefill_step_size
                self._hidden(mx.array(rest[:n])[None], cache)
                mx.eval([c.state for c in cache])
                mx.clear_cache()
                rest = rest[n:]
            y = mx.argmax(self.forward(mx.array(rest)[None], cache))
            mx.async_eval(y)
            yield y.item()
            if max_tokens == 1:
                return
            y = mx.argmax(self.forward(y.reshape(1, 1), cache))
            mx.async_eval(y)
            for i in range(1, max_tokens):
                # Pipeline decode, without delaying the first yield or evaluating
                # a token beyond the requested count.
                if i + 1 < max_tokens:
                    nxt = mx.argmax(self.forward(y.reshape(1, 1), cache))
                    mx.async_eval(nxt)
                yield y.item()
                if i + 1 < max_tokens:
                    y = nxt
