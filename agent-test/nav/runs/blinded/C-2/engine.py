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
from small_gemm import linear
from compiled import make_prefill


class Engine:
    def __init__(self, model):
        if model.model_type != "qwen3":
            raise ValueError("Only dense Qwen3 is supported")
        self.model = model
        self.args = model.args
        self.stream = generation_stream
        # Keep the original BF16 values, but issue one matrix multiplication
        # for projections sharing the same input.
        with mx.stream(self.stream):
            self.qkv = [mx.concatenate([l.self_attn.q_proj.weight,
                                        l.self_attn.k_proj.weight,
                                        l.self_attn.v_proj.weight], axis=0)
                        for l in model.layers]
            self.gate_up = [mx.concatenate([l.mlp.gate_proj.weight,
                                           l.mlp.up_proj.weight], axis=0)
                            for l in model.layers]
            mx.eval(self.qkv, self.gate_up)
        self._prefill = make_prefill(self)

    def new_cache(self, capacity: int | None = None):
        cache = [KVCache() for _ in self.model.layers]
        if capacity is not None and capacity >= 4096:
            length = ((capacity + 255) // 256) * 256
            shape = (1, self.args.num_key_value_heads, length, self.args.head_dim)
            with mx.stream(self.stream):
                for c in cache:
                    c.keys = mx.zeros(shape, self.model.model.embed_tokens.weight.dtype)
                    c.values = mx.zeros(shape, self.model.model.embed_tokens.weight.dtype)
        return cache

    def _hidden(self, tokens, cache, last=True):
        h = self.model.model.embed_tokens(tokens)
        mask = create_attention_mask(h, cache[0])
        B, L, _ = h.shape
        for i, (layer, c) in enumerate(zip(self.model.layers, cache)):
            a, m = layer.self_attn, layer.mlp
            x = layer.input_layernorm(h)
            qdim = a.n_heads * self.args.head_dim
            kdim = a.n_kv_heads * self.args.head_dim
            final = i == len(cache) - 1
            if final and L > 1:
                k, v = a.k_proj(x), a.v_proj(x)
                if last:
                    q = a.q_proj(x[:, -1:, :])
            else:
                raw_qkv = linear(x, self.qkv[i])
                q, k, v = mx.split(raw_qkv,
                                   [qdim, qdim + kdim], axis=-1)
            v = v.reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            offset = c.offset
            k = a.k_norm(k.reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            k = a.rope(k, offset=offset)
            k, v = c.update_and_fetch(k, v)
            if final and not last:
                return None
            if final and L > 1:
                h = h[:, -1:, :]
                qL = 1
                qoffset = offset + L - 1
                amask = None
            else:
                qL, qoffset, amask = L, offset, mask
            q = a.q_norm(q.reshape(B, qL, a.n_heads, -1)).transpose(0, 2, 1, 3)
            q = a.rope(q, offset=qoffset)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=amask)
            h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(B, qL, -1))
            gate, up = mx.split(linear(layer.post_attention_layernorm(h), self.gate_up[i]), 2, axis=-1)
            h = h + m.down_proj(swiglu(gate, up))
        return self.model.model.norm(h)

    def forward(self, tokens, cache):
        """Logits [vocab] of the last position; advances the cache."""
        with mx.stream(self.stream):
            n = tokens.shape[1]
            if cache[0].offset == 0:
                logits, states = self._prefill(tokens, mx.array(n - 1))
                for c, (k, v) in zip(cache, states):
                    c.keys, c.values, c.offset = k, v, n
                return logits
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
            while len(rest) > prefill_step_size:
                n = prefill_step_size
                self._hidden(mx.array(rest[:n])[None], cache, last=False)
                mx.eval([c.state for c in cache])
                rest = rest[n:]
            y = mx.argmax(self.forward(mx.array(rest)[None], cache))
            mx.async_eval(y)
            yield y.item()
            if max_tokens == 1:
                return
            y = mx.argmax(self.forward(y.reshape(1, 1), cache))
            mx.async_eval(y)
            for i in range(1, max_tokens):
                if i == max_tokens - 1:
                    yield y.item()
                    break
                nxt = mx.argmax(self.forward(y.reshape(1, 1), cache))
                mx.async_eval(nxt)
                yield y.item()
                y = nxt
