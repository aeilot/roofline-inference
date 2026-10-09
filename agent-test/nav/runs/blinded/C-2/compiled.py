"""Pure transformer graphs; request cache arrays are explicit inputs and outputs."""
import mlx.core as mx
from mlx_lm.models.activations import swiglu
from small_gemm import linear


def make_prefill(engine):
    model, args = engine.model, engine.args
    qdim = args.num_attention_heads * args.head_dim
    kdim = args.num_key_value_heads * args.head_dim

    def prefill(tokens, lastoffset):
        h = model.model.embed_tokens(tokens)
        states = []
        for i, layer in enumerate(model.layers):
            a, m = layer.self_attn, layer.mlp
            final = i == len(model.layers) - 1
            x = layer.input_layernorm(h)
            if final:
                q = a.q_proj(x[:, -1:, :])
                k, v = a.k_proj(x), a.v_proj(x)
            else:
                q, k, v = mx.split(linear(x, engine.qkv[i]), [qdim, qdim + kdim], axis=-1)
            q = a.q_norm(q.reshape(1, -1, a.n_heads, args.head_dim)).transpose(0, 2, 1, 3)
            k = a.k_norm(k.reshape(1, -1, a.n_kv_heads, args.head_dim)).transpose(0, 2, 1, 3)
            v = v.reshape(1, -1, a.n_kv_heads, args.head_dim).transpose(0, 2, 1, 3)
            q = a.rope(q, offset=lastoffset if final else 0)
            k = a.rope(k, offset=0)
            k, v = mx.contiguous(k), mx.contiguous(v)
            states.append((k, v))
            if final:
                h = h[:, -1:, :]
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale,
                                                    mask=None if final else 'causal')
            h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(1, -1, qdim))
            gate, up = mx.split(linear(layer.post_attention_layernorm(h), engine.gate_up[i]), 2, axis=-1)
            h = h + m.down_proj(swiglu(gate, up))
        h = model.model.norm(h)
        return model.model.embed_tokens.as_linear(h)[0, 0], states

    return mx.compile(prefill)

