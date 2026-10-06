"""Qwen3 BF16 single-stream engine, v2.

Level-A only: BF16 weights, BF16 KV, exact model math; the implementation may
reorder float sums (checked against an FP32 reference, see check.py).

Prefill: whole prompt in one pass (chunked only for memory), no allocator
flush, fused qk-norm+RoPE writing attention-native layouts, GQA-paired flash
attention writing the o_proj layout, fused residual+RMSNorm, compiled SwiGLU;
the last layer runs attention/MLP for the final row only; vocabulary projection
for the final row only.
Decode: 7 kernels per layer (norm+QKV GEMV, qk-norm+RoPE, one KV write,
SDPA, O GEMV+residual, norm+gate/up GEMV+SwiGLU, down GEMV+residual),
preallocated KV, greedy argmax on GPU, one token of asynchronous lookahead.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator

import mlx.core as mx

from attention import attention
from kernels import add_rms, gemv, qk_norm_rope, qk_norm_rope_prefill, skinny_matmul, use_skinny

STREAM = mx.new_stream(mx.gpu)


@mx.compile
def swiglu(gate, up):
    return gate * mx.sigmoid(gate) * up


class Cache:
    """Preallocated per-layer KV buffer [2, n_kv, capacity, head_dim]."""

    def __init__(self, n_layers, n_kv, head_dim, capacity, dtype=mx.bfloat16):
        cap = (capacity + 255) // 256 * 256
        self.kv = [mx.zeros((2, n_kv, cap, head_dim), dtype) for _ in range(n_layers)]
        self.offset = 0
        self.capacity = cap

    def update(self, i, kv):
        """Write kv [2, n_kv, n, d] at [offset, offset+n)."""
        a, b = self.offset, self.offset + kv.shape[2]
        if b > self.capacity:
            raise ValueError("KV capacity exceeded")
        self.kv[i][:, :, a:b, :] = kv
        return self.kv[i]

    def views(self, i, n):
        buf = self.kv[i]
        return buf[0:1, :, :n, :], buf[1:2, :, :n, :]

    def state(self):
        return self.kv


class Engine:
    def __init__(self, reference, *, attn_impl="gqa"):
        if reference.model_type != "qwen3":
            raise ValueError("Only dense Qwen3 is supported")
        args = reference.args
        if not args.tie_word_embeddings:
            raise ValueError("Expected tied embeddings (Qwen3-0.6B)")
        self.args = args
        self.n_heads = args.num_attention_heads
        self.n_kv = args.num_key_value_heads
        self.head_dim = args.head_dim
        if self.head_dim != 128 or args.hidden_size % 256 or self.n_heads != 2 * self.n_kv:
            raise ValueError("Kernels assume head_dim 128, hidden % 256 == 0, GQA factor 2")
        self.scale = self.head_dim ** -0.5
        self.base = args.rope_theta
        self.eps = args.rms_norm_eps
        self.attn_impl = attn_impl
        m = reference.model
        self.embed = m.embed_tokens.weight
        self.final_norm = m.norm.weight
        self.layers = []
        for layer in reference.layers:
            a, f = layer.self_attn, layer.mlp
            if a.q_proj.weight.dtype != mx.bfloat16:
                raise ValueError("BF16 weights required")
            qkv = mx.concatenate([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], axis=0)
            gu = mx.concatenate([f.gate_proj.weight, f.up_proj.weight], axis=0)
            mx.eval(qkv, gu)  # checkpoint preparation, outside request timing
            self.layers.append(dict(
                qkv=qkv, o=a.o_proj.weight, gu=gu, down=f.down_proj.weight,
                in_norm=layer.input_layernorm.weight, post_norm=layer.post_attention_layernorm.weight,
                q_norm=a.q_norm.weight, k_norm=a.k_norm.weight))

    def new_cache(self, capacity):
        return Cache(len(self.layers), self.n_kv, self.head_dim, capacity)

    @staticmethod
    def matmul(x, W):
        # Short prompts: MLX GEMM under-fills the GPU for wide W; use a skinny kernel.
        if use_skinny(x.shape[0], *W.shape):
            return skinny_matmul(x, W)
        return x @ W.T

    # ---------------- prefill (M > 1) ----------------
    def prefill(self, tokens, cache: Cache, *, last: bool, all_logits=False):
        """tokens: [1, n]. If `last`, return logits [1, rows, vocab] for the final row (or all rows)."""
        n = tokens.shape[1]
        off = cache.offset
        Lp = (n + 31) // 32 * 32
        nl = len(self.layers)
        h = self.embed[tokens[0]]                                   # [n, hidden]
        x = mx.fast.rms_norm(h, self.layers[0]["in_norm"], self.eps)
        for i, w in enumerate(self.layers):
            q, kv = qk_norm_rope_prefill(self.matmul(x, w["qkv"]), w["q_norm"], w["k_norm"], off,
                                         self.n_heads, self.n_kv, self.base, Lp)
            cache.update(i, kv)
            final = i == nl - 1
            if final and not last:
                break        # last layer's K/V written; its outputs are dead for a middle chunk
            if final and not all_logits:
                # Only the final row feeds the logits: attention/MLP for one row.
                K, V = cache.views(i, off + n)
                o = mx.fast.scaled_dot_product_attention(q[None, :, n - 1:n], K, V, scale=self.scale)
                o, h = o.reshape(1, -1), h[n - 1:]
            elif self.attn_impl == "gqa" and n >= 128:
                buf = cache.kv[i]
                o = attention(q, buf[0:1], buf[1:2], n, off + n, self.scale)
            else:
                K, V = cache.views(i, off + n)
                o = mx.fast.scaled_dot_product_attention(q[None, :, :n], K, V, scale=self.scale, mask="causal")
                o = o[0].transpose(1, 0, 2).reshape(n, -1)
            h, x = add_rms(h, self.matmul(o, w["o"]), w["post_norm"])
            gate, up = mx.split(self.matmul(x, w["gu"]), 2, axis=-1)
            r = self.matmul(swiglu(gate, up), w["down"])
            if final:
                h = h + r
            else:
                h, x = add_rms(h, r, self.layers[i + 1]["in_norm"])
        cache.offset += n
        if not last:
            return None
        if all_logits:
            h = mx.fast.rms_norm(h, self.final_norm, self.eps)
            return (h @ self.embed.T)[None]
        return gemv(self.embed, h.reshape(-1), nw=self.final_norm, out_float=True)[None, None]

    # ---------------- decode (M = 1) ----------------
    def decode(self, token, cache: Cache, *, return_logits=False):
        """token: int32 array of shape (1,). Returns next greedy token, shape (1,)."""
        h = self.embed[token].reshape(-1)
        pos = mx.array([cache.offset], dtype=mx.int32)
        n = cache.offset + 1
        for i, w in enumerate(self.layers):
            qkv = gemv(w["qkv"], h, nw=w["in_norm"])
            q, kv = qk_norm_rope(qkv, w["q_norm"], w["k_norm"], pos, self.n_heads, self.n_kv, self.base)
            cache.update(i, kv)
            K, V = cache.views(i, n)
            o = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.scale)
            h = gemv(w["o"], o.reshape(-1), res=h)
            a = gemv(w["gu"], h, nw=w["post_norm"], swiglu=True)
            h = gemv(w["down"], a, res=h)
        cache.offset += 1
        logits = gemv(self.embed, h, nw=self.final_norm, out_float=True)
        if return_logits:
            return logits
        return mx.argmax(logits, keepdims=True).astype(mx.int32)

    def generate(self, prompt: list[int], *, max_tokens=65, chunk=4096, eos_ids=()) -> Iterator[int]:
        if not prompt or max_tokens < 0 or chunk < 1:
            raise ValueError("Nonempty prompt, nonnegative max_tokens, positive chunk required")
        if min(prompt) < 0 or max(prompt) >= self.args.vocab_size:
            raise ValueError("Token outside vocabulary")
        if max_tokens == 0:
            return
        eos_ids = set(eos_ids)
        cache = self.new_cache(len(prompt) + max_tokens + 1)
        with mx.stream(STREAM):
            tokens = mx.array(prompt, dtype=mx.int32)[None]
            for start in range(0, len(prompt), chunk):
                end = min(len(prompt), start + chunk)
                if end < len(prompt):
                    self.prefill(tokens[:, start:end], cache, last=False)
                    mx.async_eval(cache.state())  # bound graph size; no CPU sync
                else:
                    logits = self.prefill(tokens[:, start:end], cache, last=True)
            y = mx.argmax(logits[0, -1:], axis=-1).astype(mx.int32)
            mx.async_eval(y)
            for _ in range(max_tokens):
                nxt = self.decode(y, cache)
                mx.async_eval(nxt)
                token = y.item()
                yield token
                if token in eos_ids:
                    break
                y = nxt


def main():
    from mlx_lm import load
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="Explain the roofline model in two sentences.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--chunk", type=int, default=4096)
    args = p.parse_args()
    model, tokenizer = load(args.model)
    engine = Engine(model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    detok = tokenizer.detokenizer
    detok.reset()
    for token in engine.generate(prompt, max_tokens=args.max_tokens, chunk=args.chunk,
                                 eos_ids=tokenizer.eos_token_ids):
        if token in tokenizer.eos_token_ids:
            break
        detok.add_token(token)
        print(detok.last_segment, end="", flush=True)
    detok.finalize()
    print(detok.last_segment)


if __name__ == "__main__":
    main()
