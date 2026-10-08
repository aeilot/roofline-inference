"""Sequential microbenchmark of row-contiguous BF16 GEMVs."""

import mlx.core as mx

src = '''
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint r = threadgroup_position_in_grid.x * (32 / S) * SG + sg * (32 / S) + lane / S;
uint kl = lane % S;
if (r >= N) return;
const device bfloat4 *wp = (const device bfloat4*)(w + r * K);
const device bfloat4 *xp = (const device bfloat4*)x;
float acc = 0;
for (uint base = 0; base < K / 4; base += S * U) {
    float4 wf[U];
    #pragma unroll
    for (uint i = 0; i < U; ++i) wf[i] = float4(wp[base + i * S + kl]);
    #pragma unroll
    for (uint i = 0; i < U; ++i) acc += dot(wf[i], float4(xp[base + i * S + kl]));
}
#pragma unroll
for (uint d = S / 2; d > 0; d /= 2) acc += simd_shuffle_down(acc, d);
if (kl == 0) y[r] = bfloat(acc);
'''
kern=mx.fast.metal_kernel(name='row_gemv',input_names=['w','x'],output_names=['y'],source=src)
def gemv(w,x,s=32,sg=4,u=2):
    n,k=w.shape
    return kern(inputs=[w,x],template=[('N',n),('K',k),('S',s),('SG',sg),('U',u)],grid=(((n+(32//s)*sg-1)//((32//s)*sg))*sg*32,1,1),threadgroup=(sg*32,1,1),output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.bfloat16])[0]


head_source = r'''
uint lane = thread_index_in_simdgroup;
uint idx = thread_position_in_grid.x / 32;
uint head = idx % (H + KV);
uint t = idx / (H + KV);
if (head >= H + KV) return;
uint inbase = t * (H + 2 * KV) * 128 + head * 128;
float x0 = float(qkv[inbase + lane]);
float x1 = float(qkv[inbase + lane + 32]);
float x2 = float(qkv[inbase + lane + 64]);
float x3 = float(qkv[inbase + lane + 96]);
float ss = simd_sum(x0*x0 + x1*x1 + x2*x2 + x3*x3);
float inv = metal::rsqrt(ss / 128.0f + 1e-6f);
const device bfloat* w = head < H ? qw : kw;
// Keep the same BF16 boundary between RMSNorm and rotary encoding.
float n0 = float(bfloat(float(bfloat(x0 * inv)) * float(w[lane])));
float n1 = float(bfloat(float(bfloat(x1 * inv)) * float(w[lane + 32])));
float n2 = float(bfloat(float(bfloat(x2 * inv)) * float(w[lane + 64])));
float n3 = float(bfloat(float(bfloat(x3 * inv)) * float(w[lane + 96])));
float pos = float(offset + int(t));
float a0 = pos * freq[lane];
float a1 = pos * freq[lane + 32];
float c0 = metal::cos(a0), s0 = metal::sin(a0);
float c1 = metal::cos(a1), s1 = metal::sin(a1);
const uint outbase = ((head < H ? head : head - H) * L + t) * 128;
if (head < H) {
    q[outbase + lane] = bfloat(n0*c0 - n2*s0);
    q[outbase + lane + 32] = bfloat(n1*c1 - n3*s1);
    q[outbase + lane + 64] = bfloat(n0*s0 + n2*c0);
    q[outbase + lane + 96] = bfloat(n1*s1 + n3*c1);
} else {
    k[outbase + lane] = bfloat(n0*c0 - n2*s0);
    k[outbase + lane + 32] = bfloat(n1*c1 - n3*s1);
    k[outbase + lane + 64] = bfloat(n0*s0 + n2*c0);
    k[outbase + lane + 96] = bfloat(n1*s1 + n3*c1);
    uint vbase = inbase + KV * 128;
    v[outbase + lane] = qkv[vbase + lane];
    v[outbase + lane + 32] = qkv[vbase + lane + 32];
    v[outbase + lane + 64] = qkv[vbase + lane + 64];
    v[outbase + lane + 96] = qkv[vbase + lane + 96];
}
'''
head_kernel = mx.fast.metal_kernel(name='qkv_norm_rope',input_names=['qkv','qw','kw','freq','offset'],output_names=['q','k','v'],source=head_source)


frequency_kernel = mx.fast.metal_kernel(name='rope_frequencies',input_names=[],output_names=['freq'],source='''
uint d = thread_position_in_grid.x;
freq[d] = metal::exp2(-float(d) / 64.0f * metal::log2(float(BASE)));
''')
