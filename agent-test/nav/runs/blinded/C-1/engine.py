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
if __package__:
    from .small_mm import matmul as small_matmul
else:
    from small_mm import matmul as small_matmul


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
            self.gate_up = [mx.concatenate([l.mlp.gate_proj.weight,
                                           l.mlp.up_proj.weight]) for l in model.layers]
            mx.eval(self.qkv, self.gate_up)
        self._compiled_prefill = mx.compile(self._fresh_prefill)

    def _fresh_prefill(self, tokens):
        # Only immutable weights are captured by the compiled graph. Each
        # invocation creates fresh request states and returns them explicitly.
        cache = self.new_cache()
        h = self._hidden(tokens, cache)
        logits = self.model.model.embed_tokens.as_linear(h[:, -1:, :])[0, 0]
        return logits, [(c.keys, c.values) for c in cache]

    def new_cache(self, capacity: int | None = None):
        return [KVCache() for _ in self.model.layers]

    def _hidden(self, tokens, cache, cache_only=False):
        h = self.model.model.embed_tokens(tokens)
        mask = create_attention_mask(h, cache[0])
        B, L, _ = h.shape
        for i, (layer, c) in enumerate(zip(self.model.layers, cache)):
            a, m = layer.self_attn, layer.mlp
            last = i == len(self.model.layers) - 1
            x = layer.input_layernorm(h)
            q_size = a.n_heads * self.args.head_dim
            if last and L > 64:
                kv = x @ self.qkv[i][q_size:].T
                k, v = mx.split(kv, 2, axis=-1)
                if not cache_only:
                    q = a.q_proj(x[:, -1:, :])
                    q = a.q_norm(q.reshape(B, 1, a.n_heads, -1)).transpose(0, 2, 1, 3)
                    q = a.rope(q, offset=c.offset + L - 1)
            else:
                if L >= 512:
                    q, k, v = a.q_proj(x), a.k_proj(x), a.v_proj(x)
                else:
                    qkv = self._project(x, self.qkv[i])
                    q, k, v = mx.split(qkv, [q_size, q_size + a.n_kv_heads * self.args.head_dim], axis=-1)
                q = a.q_norm(q.reshape(B, L, a.n_heads, -1)).transpose(0, 2, 1, 3)
                q = a.rope(q, offset=c.offset)
            k = a.k_norm(k.reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            k = a.rope(k, offset=c.offset)
            k, v = c.update_and_fetch(k, v)
            if last and cache_only:
                return None
            if last:
                q, h = q[:, :, -1:, :], h[:, -1:, :]
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale,
                                                    mask=None if last else mask)
            h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(B, 1 if last else L, -1))
            x = layer.post_attention_layernorm(h)
            if x.shape[1] >= 512:
                gate, up = m.gate_proj(x), m.up_proj(x)
            else:
                gu = self._project(x, self.gate_up[i])
                gate, up = mx.split(gu, 2, axis=-1)
            h = h + m.down_proj(swiglu(gate, up))
        return self.model.model.norm(h)

    @staticmethod
    def _project(x, weight):
        if (x.dtype == mx.bfloat16 and weight.dtype == mx.bfloat16
                and x.size // x.shape[-1] == 16 and weight.shape[1] == 1024
                and weight.shape[0] % 32 == 0):
            return small_matmul(x, weight)
        return x @ weight.T

    def forward(self, tokens, cache):
        """Logits [vocab] of the last position; advances the cache."""
        with mx.stream(self.stream):
            if cache[0].offset == 0 and tokens.shape[1] > 1:
                logits, states = self._compiled_prefill(tokens)
                for c, (k, v) in zip(cache, states):
                    c.keys, c.values, c.offset = k, v, tokens.shape[1]
                return logits
            h = self._hidden(tokens, cache)
            return self.model.model.embed_tokens.as_linear(h[:, -1:, :])[0, 0]

    def generate(self, prompt: list[int], max_tokens: int, prefill_step_size: int = 2048) -> Iterator[int]:
        if not prompt or max_tokens < 0:
            raise ValueError("Nonempty prompt and nonnegative max_tokens required")
        if max_tokens == 0:
            return
        cache = self.new_cache(len(prompt) + max_tokens)
        if len(prompt) > prefill_step_size:
            # Avoid repeatedly copying the growing prefix at each chunk.
            capacity = (len(prompt) + max_tokens + 255) // 256 * 256
            for c in cache:
                c.step = capacity
        with mx.stream(self.stream):
            rest = prompt
            while len(rest) > prefill_step_size:
                n = prefill_step_size
                self._hidden(mx.array(rest[:n])[None], cache, cache_only=True)
                mx.eval([c.state for c in cache])
                rest = rest[n:]
            y = mx.argmax(self.forward(mx.array(rest)[None], cache))
            mx.async_eval(y)
            for _ in range(max_tokens - 1):
                nxt = mx.argmax(self.forward(y.reshape(1, 1), cache))
                mx.async_eval(nxt)
                yield y.item()
                y = nxt
            yield y.item()
