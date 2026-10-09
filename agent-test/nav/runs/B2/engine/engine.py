"""Qwen3 BF16 single-stream inference engine.

Transformer forward and greedy generation on MLX, with packed projections,
fused head normalization/RoPE, and a short-prefill Metal GEMM. Uses mlx-lm's
model weights, attention conventions and KV cache growth.

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
from matmul import linear
from norm_rope import make_kernel


class PromptKVCache(KVCache):
    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys, self.values = keys, values
            self.offset = keys.shape[2]
            return keys, values
        return super().update_and_fetch(keys, values)


class Engine:
    def __init__(self, model):
        if model.model_type != "qwen3":
            raise ValueError("Only dense Qwen3 is supported")
        self.model = model
        self.args = model.args
        self.stream = generation_stream
        self.projections = []
        with mx.stream(self.stream):
            for layer in model.layers:
                a, m = layer.self_attn, layer.mlp
                self.projections.append((
                    mx.concatenate([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]),
                    mx.concatenate([a.k_proj.weight, a.v_proj.weight]),
                    mx.concatenate([m.gate_proj.weight, m.up_proj.weight]),
                ))
            mx.eval(self.projections)

        args = self.args
        head_dim = args.head_dim
        qwidth = args.num_attention_heads * head_dim
        kwidth = args.num_key_value_heads * head_dim
        eps = args.rms_norm_eps

        norm_rope = make_kernel(args.num_attention_heads, args.num_key_value_heads, eps, args.rope_theta)

        def prepare(h, nw, qw, qnw, knw, offset):
            x = mx.fast.rms_norm(h, nw, eps)
            return norm_rope(linear(x, qw), qnw, knw, offset)

        def finish(h, o, ow, nw, guw, dw):
            h = h + linear(o, ow)
            x = mx.fast.rms_norm(h, nw, eps)
            packed = linear(x, guw)
            gate = packed[..., :args.intermediate_size]
            up = packed[..., args.intermediate_size:2 * args.intermediate_size]
            return h + linear(swiglu(gate, up), dw)

        self.prepare = mx.compile(prepare)
        self.finish = mx.compile(finish)

    def new_cache(self, capacity: int | None = None):
        return [PromptKVCache() for _ in self.model.layers]

    def _hidden(self, tokens, cache, output=True):
        h = self.model.model.embed_tokens(tokens)
        mask = create_attention_mask(h, cache[0])
        B, L, _ = h.shape
        position = mx.array(cache[0].offset)
        for i, (layer, c) in enumerate(zip(self.model.layers, cache)):
            a, m = layer.self_attn, layer.mlp
            last = i == len(self.model.layers) - 1
            wqkv, wkv, wgu = self.projections[i]
            offset = c.offset
            if last:
                x = layer.input_layernorm(h)
                k, v = mx.split(linear(x, wkv), 2, axis=-1)
                k = a.k_norm(k.reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
                v = v.reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
                k = a.rope(k, offset=offset)
            else:
                q, k, v = self.prepare(h, layer.input_layernorm.weight, wqkv,
                                      a.q_norm.weight, a.k_norm.weight, position)
            k, v = c.update_and_fetch(k, v)
            if last and not output:
                return None
            qlen = 1 if last else L
            if last:
                q = a.q_proj(x[:, -1:, :])
                q = a.q_norm(q.reshape(B, qlen, a.n_heads, -1)).transpose(0, 2, 1, 3)
                q = a.rope(q, offset=offset + L - qlen)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=None if last else mask)
            o = o.transpose(0, 2, 1, 3).reshape(B, qlen, -1)
            h = self.finish(h[:, -1:, :] if last else h, o, a.o_proj.weight,
                            layer.post_attention_layernorm.weight, wgu, m.down_proj.weight)
            if i == 0 and L > 1:
                mx.async_eval(h)
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
            while len(rest) > prefill_step_size:
                n = prefill_step_size
                self._hidden(mx.array(rest[:n])[None], cache, output=False)
                mx.eval([c.state for c in cache])
                mx.clear_cache()
                rest = rest[n:]
            y = mx.argmax(self.forward(mx.array(rest)[None], cache))
            mx.async_eval(y)
            # Deliver the first token before encoding the decode lookahead.
            yield y.item()
            if max_tokens == 1:
                return
            y = mx.argmax(self.forward(y.reshape(1, 1), cache))
            mx.async_eval(y)
            for i in range(1, max_tokens):
                if i + 1 < max_tokens:
                    nxt = mx.argmax(self.forward(y.reshape(1, 1), cache))
                    mx.async_eval(nxt)
                yield y.item()
                if i + 1 < max_tokens:
                    y = nxt
