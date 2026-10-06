"""HISTORICAL SNAPSHOT (not runnable against current kernels.py): the CPU+GPU
split-GEMM prefill tried and rejected in README section 4. Kept for reference.
"""
"""Qwen3 BF16 single-stream engine, v2.

Level-A only: BF16 weights, BF16 KV, exact model math; the implementation may
reorder float sums (checked against an FP32 reference, see check.py).

Prefill: whole prompt in one pass (chunked only for memory), no allocator
flush, last layer computes attention/MLP only for the final row, vocabulary
projection only for the final row. Decode: 7 kernels per layer (fused
norm-GEMV, fused qk-norm+RoPE, two KV writes, SDPA, GEMV+residual,
norm-GEMV+SwiGLU, GEMV+residual), preallocated KV, greedy argmax on GPU, one
token of asynchronous lookahead.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator

import mlx.core as mx
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from kernels import gemv, qk_norm_rope

STREAM = mx.new_stream(mx.gpu)


class Cache:
    """Preallocated per-layer K/V; capacity fixed for the request."""

    def __init__(self, n_layers, n_kv, head_dim, capacity, dtype=mx.bfloat16):
        cap = (capacity + 255) // 256 * 256
        self.keys = [mx.zeros((1, n_kv, cap, head_dim), dtype) for _ in range(n_layers)]
        self.values = [mx.zeros((1, n_kv, cap, head_dim), dtype) for _ in range(n_layers)]
        self.offset = 0
        self.capacity = cap

    def update(self, i, k, v):
        """Write k/v at [offset, offset+n); returns views up to offset+n."""
        n = k.shape[2]
        a, b = self.offset, self.offset + n
        if b > self.capacity:
            raise ValueError("KV capacity exceeded")
        self.keys[i][..., a:b, :] = k
        self.values[i][..., a:b, :] = v
        return self.keys[i][..., :b, :], self.values[i][..., :b, :]

    def state(self):
        return self.keys + self.values


def _round(n, m=64):
    return max(m, int(round(n / m)) * m)


class Engine:
    def __init__(self, reference, *, cpu_frac=0.0, cpu_min_tokens=256, cpu_mode="numpy"):
        if reference.model_type != "qwen3":
            raise ValueError("Only dense Qwen3 is supported")
        args = reference.args
        if not args.tie_word_embeddings:
            raise ValueError("Expected tied embeddings (Qwen3-0.6B)")
        self.args = args
        self.n_heads = args.num_attention_heads
        self.n_kv = args.num_key_value_heads
        self.head_dim = args.head_dim
        if self.head_dim != 128 or args.hidden_size % 256:
            raise ValueError("Kernels assume head_dim 128, hidden % 256 == 0")
        self.scale = self.head_dim ** -0.5
        self.base = args.rope_theta
        self.eps = args.rms_norm_eps
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
        # Heterogeneous prefill: GPU computes output rows [:Ng] of each weight in
        # BF16; the CPU matrix unit (Accelerate SGEMM) computes rows [Ng:] from an
        # exact FP32 upcast of the same BF16 weights, FP32 accumulation, BF16 output.
        self.cpu_frac = cpu_frac
        self.cpu_min_tokens = cpu_min_tokens
        self.cpu_weights = []
        if cpu_frac > 0:
            for w in self.layers:
                part = {}
                for name in ("qkv", "o", "gu", "down"):
                    W = w[name]
                    ng = W.shape[0] - _round(W.shape[0] * cpu_frac)
                    Wc = W[ng:].astype(mx.float32)
                    mx.eval(Wc)
                    part[name] = (ng, Wc, np.asarray(Wc).T)  # (K, Nc) view for numpy
                self.cpu_weights.append(part)
        self.cpu_mode = cpu_mode
        self._pool = ThreadPoolExecutor(1)
        self._bufs = {}
        self.q_size = self.n_heads * self.head_dim
        self.k_size = self.n_kv * self.head_dim
        self.ffn = self.layers[0]["down"].shape[1]

    def new_cache(self, capacity):
        return Cache(len(self.layers), self.n_kv, self.head_dim, capacity)

    def linear(self, x, i, name):
        W = self.layers[i][name]
        if not self.cpu_weights or x.shape[1] < self.cpu_min_tokens:
            return x @ W.T
        ng, Wc, WcT = self.cpu_weights[i][name]
        if self.cpu_mode == "numpy":
            return self._linear_numpy(x, W, ng, WcT)
        x32 = x.astype(mx.float32)          # conversions on GPU; CPU runs SGEMM only
        with mx.stream(mx.cpu):
            yc = x32 @ Wc.T
        yg = x @ W[:ng].T
        return mx.concatenate([yg, yc.astype(x.dtype)], axis=-1)

    def _linear_numpy(self, x, W, ng, WcT):
        """CPU SGEMM (Accelerate via numpy, GIL released) into a unified-memory
        MLX buffer, concurrent with the GPU GEMM. The GPU stream is drained up to
        x before the CPU reads it, so earlier readers of the buffer are done."""
        lead, K = x.shape[:-1], x.shape[-1]
        M = x.size // K
        key = (M, WcT.shape[1])
        if key not in self._bufs:
            Y = mx.zeros(key, mx.float32)
            mx.eval(Y)
            self._bufs[key] = (Y, np.asarray(Y))
        Y, ynp = self._bufs[key]
        x32 = x.astype(mx.float32)
        mx.eval(x32)
        fut = self._pool.submit(np.matmul, np.asarray(x32).reshape(M, K), WcT, out=ynp)
        yg = x @ W[:ng].T
        mx.async_eval(yg)
        fut.result()
        yc = (Y + 0).astype(x.dtype).reshape(*lead, -1)   # +0: fresh node reading the buffer now
        return mx.concatenate([yg, yc], axis=-1)

    # ---------------- prefill (M > 1) ----------------
    def prefill(self, tokens, cache: Cache, *, last: bool, all_logits=False):
        """tokens: [1, n]. If `last`, return logits of the final row (or all rows)."""
        B, n = tokens.shape
        h = self.embed[tokens]
        mask = "causal" if n > 1 else None
        nl = len(self.layers)
        for i, w in enumerate(self.layers):
            x = mx.fast.rms_norm(h, w["in_norm"], self.eps)
            qkv = self.linear(x, i, "qkv")
            q, k, v = mx.split(qkv, [self.q_size, self.q_size + self.k_size], axis=-1)
            q = mx.fast.rms_norm(q.reshape(B, n, self.n_heads, -1), w["q_norm"], self.eps).transpose(0, 2, 1, 3)
            k = mx.fast.rms_norm(k.reshape(B, n, self.n_kv, -1), w["k_norm"], self.eps).transpose(0, 2, 1, 3)
            v = v.reshape(B, n, self.n_kv, -1).transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.base, scale=1.0, offset=cache.offset)
            k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.base, scale=1.0, offset=cache.offset)
            K, V = cache.update(i, k, v)
            if i == nl - 1:
                if not last:
                    break  # last layer's K/V written; its outputs are dead for a middle chunk
                if not all_logits:
                    # Only the final row feeds the logits: attention/MLP for one row.
                    q, h, mask = q[:, :, -1:, :], h[:, -1:, :], None
            o = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.scale, mask=mask)
            o = o.transpose(0, 2, 1, 3).reshape(B, o.shape[2], -1)
            h = h + self.linear(o, i, "o")
            x = mx.fast.rms_norm(h, w["post_norm"], self.eps)
            gate, up = mx.split(self.linear(x, i, "gu"), 2, axis=-1)
            h = h + self.linear(nn_silu(gate) * up, i, "down")
        cache.offset += n
        if not last:
            return None
        h = mx.fast.rms_norm(h, self.final_norm, self.eps)
        return h @ self.embed.T

    # ---------------- decode (M = 1) ----------------
    def decode(self, token, cache: Cache, *, return_logits=False):
        """token: int32 array of shape (1,). Returns next greedy token, shape (1,)."""
        h = self.embed[token].reshape(-1)
        pos = mx.array([cache.offset], dtype=mx.int32)
        for i, w in enumerate(self.layers):
            qkv = gemv(w["qkv"], h, nw=w["in_norm"])
            q, k, v = qk_norm_rope(qkv, w["q_norm"], w["k_norm"], pos, self.n_heads, self.n_kv, self.base)
            K, V = cache.update(i, k, v)
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


def nn_silu(x):
    return x * mx.sigmoid(x)


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
