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
from kernels import gemv, head_kernel, frequency_kernel


class Engine:
    def __init__(self, model):
        if model.model_type != 'qwen3':
            raise ValueError('Only dense Qwen3 is supported')
        self.model=model; self.args=model.args; self.stream=generation_stream
        mx.set_cache_limit(1 << 30)
        self.h=self.args.num_attention_heads;self.kv=self.args.num_key_value_heads
        self.d=self.args.head_dim;self.eps=self.args.rms_norm_eps
        self.base=self.args.rope_theta;self.scale=self.d**-0.5
        self.emb=model.model.embed_tokens.weight;self.norm=model.model.norm.weight
        self.prefill = mx.compile(self._prefill)
        with mx.stream(self.stream):
            self.qkv_weights=[mx.concatenate([l.self_attn.q_proj.weight,l.self_attn.k_proj.weight,l.self_attn.v_proj.weight],axis=0) for l in model.layers]
            self.gu_weights=[mx.concatenate([l.mlp.gate_proj.weight,l.mlp.up_proj.weight],axis=0) for l in model.layers]
            self.blocks=[(qw,gw,l.self_attn.o_proj.weight,l.mlp.down_proj.weight,
                          l.input_layernorm.weight,l.post_attention_layernorm.weight,
                          l.self_attn.q_norm.weight,l.self_attn.k_norm.weight)
                         for l,qw,gw in zip(model.layers,self.qkv_weights,self.gu_weights)]
            self.freq=frequency_kernel(inputs=[],template=[('BASE',int(self.base))],grid=(64,1,1),threadgroup=(64,1,1),output_shapes=[(64,)],output_dtypes=[mx.float32])[0]
            mx.eval(self.qkv_weights,self.gu_weights,self.freq)
            for layer,qw,gw in zip(model.layers,self.qkv_weights,self.gu_weights):
                a,m=layer.self_attn,layer.mlp
                qn=a.n_heads*self.d;kn=a.n_kv_heads*self.d
                a.q_proj.weight=qw[:qn]
                a.k_proj.weight=qw[qn:qn+kn]
                a.v_proj.weight=qw[qn+kn:]
                m.gate_proj.weight,m.up_proj.weight=mx.split(gw,2,axis=0)

    def new_cache(self,capacity=None):
        caches=[KVCache() for _ in self.blocks]
        if capacity is not None and capacity>0:
            for c in caches:c.step=((capacity+255)//256)*256
        return caches

    def _hidden(self,tokens,cache,last_only=False):
        h=mx.take(self.emb,tokens,axis=0)
        mask=create_attention_mask(h,cache[0])
        b,l,_=h.shape
        offset=mx.array(cache[0].offset,mx.int32)
        for i,((qw,gw,ow,dw,n1,n2,qn,kn),c) in enumerate(zip(self.blocks,cache)):
            x=mx.fast.rms_norm(h,n1,self.eps)
            last=last_only and i+1==len(cache)
            ol=1 if last else l
            if last and l>1:
                q=gemv(qw[:self.h*self.d],x[:,-1:,:])
                k,v=mx.split(x@qw[self.h*self.d:].T,2,axis=-1)
                q=mx.fast.rms_norm(q.reshape(b,1,self.h,self.d),qn,self.eps).transpose(0,2,1,3)
                k=mx.fast.rms_norm(k.reshape(b,l,self.kv,self.d),kn,self.eps).transpose(0,2,1,3)
                v=v.reshape(b,l,self.kv,self.d).transpose(0,2,1,3)
                q=mx.fast.rope(q,self.d,traditional=False,base=self.base,scale=1.0,offset=c.offset+l-1)
                k=mx.fast.rope(k,self.d,traditional=False,base=self.base,scale=1.0,offset=c.offset)
            else:
                qkv=gemv(qw,x,s=8) if l==1 else x@qw.T
                q,k,v=head_kernel(inputs=[qkv,qn,kn,self.freq,offset],template=[('L',l),('H',self.h),('KV',self.kv)],grid=(l*(self.h+self.kv)*32,1,1),threadgroup=(128,1,1),output_shapes=[(1,self.h,l,128),(1,self.kv,l,128),(1,self.kv,l,128)],output_dtypes=[mx.bfloat16]*3)
            k,v=c.update_and_fetch(k,v)
            o=mx.fast.scaled_dot_product_attention(q,k,v,scale=self.scale,mask=None if last else mask)
            if last:h=h[:,-1:,:]
            o=o.transpose(0,2,1,3).reshape(b,ol,-1)
            h=h+(gemv(ow,o,u=1) if ol==1 else o@ow.T)
            x=mx.fast.rms_norm(h,n2,self.eps)
            gate,up=mx.split(gemv(gw,x) if ol==1 else x@gw.T,2,axis=-1)
            x=swiglu(gate,up)
            h=h+(gemv(dw,x) if ol==1 else x@dw.T)
        return h

    def _prefill(self, tokens, keys, values):
        # The compiled graph is stateless: all request KV data is an explicit input.
        caches = []
        for k, v in zip(keys, values):
            c = KVCache()
            c.keys, c.values = k, v
            caches.append(c)
        h = self._hidden(tokens, caches, last_only=True)
        h = mx.fast.rms_norm(h[:, -1:, :], self.norm, self.eps)
        return gemv(self.emb, h, sg=8, u=8)[0, 0], [c.keys for c in caches], [c.values for c in caches]

    def forward(self,tokens,cache):
        with mx.stream(self.stream):
            n = tokens.shape[1]
            if n in (16, 1024) and cache[0].offset == 0:
                for c in cache:
                    if c.keys is None or c.keys.shape[2] < n:
                        capacity = ((n + c.step - 1) // c.step) * c.step
                        c.keys = mx.zeros((1, self.kv, capacity, self.d), mx.bfloat16)
                        c.values = mx.zeros_like(c.keys)
                logits, keys, values = self.prefill(tokens, [c.keys for c in cache], [c.values for c in cache])
                for c, k, v in zip(cache, keys, values):
                    c.keys, c.values, c.offset = k, v, n
                return logits
            h=mx.fast.rms_norm(self._hidden(tokens,cache,last_only=True)[:,-1:,:],self.norm,self.eps)
            return gemv(self.emb,h,sg=8,u=8)[0,0]

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
                self._hidden(mx.array(rest[:n])[None], cache)
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
                if i + 1 < max_tokens:
                    nxt = mx.argmax(self.forward(y.reshape(1, 1), cache))
                    mx.async_eval(nxt)
                yield y.item()
                if i + 1 < max_tokens:
                    y = nxt
