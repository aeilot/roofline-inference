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


class Engine:
    def __init__(self, model):
        if model.model_type != "qwen3":
            raise ValueError("Only dense Qwen3 is supported")
        self.model = model
        self.args = model.args
        self.stream = generation_stream

    def new_cache(self, capacity: int | None = None):
        return [KVCache() for _ in self.model.layers]

    def _hidden(self, tokens, cache):
        h = self.model.model.embed_tokens(tokens)
        mask = create_attention_mask(h, cache[0])
        B, L, _ = h.shape
        for layer, c in zip(self.model.layers, cache):
            a, m = layer.self_attn, layer.mlp
            x = layer.input_layernorm(h)
            q, k, v = a.q_proj(x), a.k_proj(x), a.v_proj(x)
            q = a.q_norm(q.reshape(B, L, a.n_heads, -1)).transpose(0, 2, 1, 3)
            k = a.k_norm(k.reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q, k = a.rope(q, offset=c.offset), a.rope(k, offset=c.offset)
            k, v = c.update_and_fetch(k, v)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=mask)
            h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
            h = h + m.down_proj(swiglu(m.gate_proj(layer.post_attention_layernorm(h)),
                                       m.up_proj(layer.post_attention_layernorm(h))))
        return self.model.model.norm(h)

    def forward(self, tokens, cache):
        """Logits [vocab] of the last position; advances the cache."""
        with mx.stream(self.stream):
            h = self._hidden(tokens, cache)
            return self.model.model.embed_tokens.as_linear(h[:, -1:, :])[0, 0]

    def generate(self, prompt: list[int], max_tokens: int, prefill_step_size: int = 2048) -> Iterator[int]:
        if not prompt or max_tokens < 0:
            raise ValueError("Nonempty prompt and nonnegative max_tokens required")
        if max_tokens == 0:
            return
        cache = self.new_cache(len(prompt) + max_tokens)
        with mx.stream(self.stream):
            rest = prompt
            while len(rest) > 1:
                n = min(prefill_step_size, len(rest) - 1)
                self._hidden(mx.array(rest[:n])[None], cache)
                mx.eval([c.state for c in cache])
                mx.clear_cache()
                rest = rest[n:]
            y = mx.argmax(self.forward(mx.array(rest)[None], cache))
            mx.async_eval(y)
            for _ in range(max_tokens):
                nxt = mx.argmax(self.forward(y.reshape(1, 1), cache))
                mx.async_eval(nxt)
                yield y.item()
                y = nxt
