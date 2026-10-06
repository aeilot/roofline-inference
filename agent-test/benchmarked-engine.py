"""BF16 Qwen3 single-stream engine; decode projections + residual/norm fusion.

Uses MLX GPU primitives and mlx-lm's checkpoint loader/RoPE/KVCache, but
implements its own transformer forward and generation scheduler. No quantization,
speculation, approximate attention, or precision conversion.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generation_stream
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from kernels import add_rms


class Engine:
    def __init__(self, reference, *, pack=True, fused_norm=False):
        if reference.model_type != "qwen3":
            raise ValueError("Only dense Qwen3 is supported")
        if any(w.dtype != mx.bfloat16 for w in
               [reference.model.embed_tokens.weight] +
               [layer.self_attn.q_proj.weight for layer in reference.layers]):
            raise ValueError("This experiment requires unquantized BF16 weights")
        self.reference = reference
        self.args = reference.args
        self.pack = pack
        self.fused_norm = fused_norm
        self.projections = []
        for layer in reference.layers:
            a, m = layer.self_attn, layer.mlp
            q_size, k_size = a.q_proj.weight.shape[0], a.k_proj.weight.shape[0]
            if pack:
                qkv = mx.concatenate([a.q_proj.weight, a.k_proj.weight,
                                      a.v_proj.weight], axis=0)
                gu = mx.concatenate([m.gate_proj.weight, m.up_proj.weight], axis=0)
                mx.eval(qkv, gu)  # checkpoint preparation, outside request timing
            else:
                qkv = gu = None
            self.projections.append((qkv, gu, q_size, k_size))

    def new_cache(self):
        return [KVCache() for _ in self.reference.layers]

    def forward(self, tokens, cache, *, logits=True, all_logits=False):
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
            raise ValueError("Expected nonempty token input of shape (1, length)")
        if len(cache) != len(self.reference.layers):
            raise ValueError("One cache per layer is required")
        offsets = {c.offset for c in cache}
        if len(offsets) != 1:
            raise ValueError("Layer cache offsets must agree")
        h = self.reference.model.embed_tokens(tokens)
        mask = create_attention_mask(h, cache[0])
        B, L, _ = h.shape
        x = self.reference.layers[0].input_layernorm(h)
        for index, (layer, c, (qkv, gu, q_size, k_size)) in enumerate(zip(
            self.reference.layers, cache, self.projections
        )):
            a, m = layer.self_attn, layer.mlp
            if self.pack and L == 1:
                projected = x @ qkv.T
                q, k, v = mx.split(projected, [q_size, q_size + k_size], axis=-1)
            else:
                q, k, v = a.q_proj(x), a.k_proj(x), a.v_proj(x)
            q = a.q_norm(q.reshape(B, L, a.n_heads, -1)).transpose(0, 2, 1, 3)
            k = a.k_norm(k.reshape(B, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q, k = a.rope(q, offset=c.offset), a.rope(k, offset=c.offset)
            k, v = c.update_and_fetch(k, v)
            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=mask)
            r = a.o_proj(x.transpose(0, 2, 1, 3).reshape(B, L, -1))
            if self.fused_norm:
                h, x = add_rms(h, r, layer.post_attention_layernorm)
            else:
                h = h + r
                x = layer.post_attention_layernorm(h)
            if self.pack and L == 1:
                gate, up = mx.split(x @ gu.T, 2, axis=-1)
            else:
                gate, up = m.gate_proj(x), m.up_proj(x)
            r = m.down_proj(swiglu(gate, up))
            if index + 1 < len(self.reference.layers):
                norm = self.reference.layers[index + 1].input_layernorm
                if self.fused_norm:
                    h, x = add_rms(h, r, norm)
                else:
                    h = h + r
                    x = norm(h)
            else:
                h = h + r
        if not logits:
            return h  # prefill only needs KV, not vocabulary projections
        h = self.reference.model.norm(h if all_logits else h[:, -1:, :])
        if self.args.tie_word_embeddings:
            return self.reference.model.embed_tokens.as_linear(h)
        return self.reference.lm_head(h)

    def generate(self, prompt: list[int], *, max_tokens=65,
                 prefill_step_size=2048, eos_ids=(), merge_prompt_tail=False) -> Iterator[int]:
        if not prompt or max_tokens < 0 or prefill_step_size < 1:
            raise ValueError("Nonempty prompt, nonnegative max_tokens, positive chunk required")
        if min(prompt) < 0 or max(prompt) >= self.args.vocab_size:
            raise ValueError("Token outside vocabulary")
        if max_tokens == 0:
            return
        cache = self.new_cache()
        eos_ids = set(eos_ids)
        with mx.stream(generation_stream):
            # Tail merge is an engine scheduling optimization. The alternative
            # preserves mlx-lm's EXACT L-1 then 1 partition for a strict ablation.
            prefix = len(prompt) if merge_prompt_tail else len(prompt) - 1
            for start in range(0, prefix, prefill_step_size):
                end = min(prefix, start + prefill_step_size)
                final = merge_prompt_tail and end == len(prompt)
                if final:
                    logits = self.forward(mx.array([prompt[start:end]]), cache)
                else:
                    # Discard unused hidden graph BEFORE cache evaluation and
                    # subsequent updates; retaining it pins old KV versions and
                    # forces copy-on-write as the next chunk appends tokens.
                    self.forward(mx.array([prompt[start:end]]), cache, logits=False)
                    mx.eval([c.state for c in cache])
            if not merge_prompt_tail:
                logits = self.forward(mx.array([[prompt[-1]]]), cache)
            y = mx.argmax(logits[:, -1], axis=-1)
            mx.async_eval(y)
            for i in range(max_tokens):
                # Match mlx-lm's one-token lookahead, including final in-flight
                # token, so both benchmark paths perform the same decode steps.
                nxt = mx.argmax(self.forward(y[None], cache)[:, -1], axis=-1)
                mx.async_eval(nxt)
                token = y.item()
                yield token
                if token in eos_ids:
                    break
                y = nxt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="Explain the roofline model in two sentences.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--no-pack", action="store_true")
    p.add_argument("--fused-norm", action="store_true")
    p.add_argument("--merge-prompt-tail", action="store_true", help="experimental: may change BF16 greedy tokens")
    args = p.parse_args()
    model, tokenizer = load(args.model)
    engine = Engine(model, pack=not args.no_pack, fused_norm=args.fused_norm)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    tokenizer.detokenizer.reset()
    for token in engine.generate(prompt, max_tokens=args.max_tokens,
                                 prefill_step_size=args.chunk,
                                 eos_ids=tokenizer.eos_token_ids,
                                 merge_prompt_tail=args.merge_prompt_tail):
        if token in tokenizer.eos_token_ids:
            break
        tokenizer.detokenizer.add_token(token)
        print(tokenizer.detokenizer.last_segment, end="", flush=True)
    tokenizer.detokenizer.finalize()
    print(tokenizer.detokenizer.last_segment)
    mx.synchronize(generation_stream)


if __name__ == "__main__":
    main()
