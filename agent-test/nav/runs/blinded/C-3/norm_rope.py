import mlx.core as mx
from functools import lru_cache

# Preserve MLX RMSNorm's BF16 rounding between normalization and learned
# scaling, and between scaling and rotary embedding. All reductions are FP32.
SOURCE = r'''
uint lane = thread_position_in_threadgroup.x;
uint pos = threadgroup_position_in_grid.y;
uint head = threadgroup_position_in_grid.z;
uint row = pos * ((NH + 2 * NK) * D) + head * D;
float ss = 0.0f;
for (uint j = 0; j < D; j += 32) {
    float x = float(qkv[row + lane + j]);
    ss += x * x;
}
float inv = metal::rsqrt(simd_sum(ss) / float(D) + EPS);
uint oh = head < NH ? head : head - NH;
for (uint j = 0; j < D; j += 32) {
    uint d = lane + j;
    uint other = (d + D / 2) % D;
    float w = head < NH ? float(qw[d]) : float(kw[d]);
    float wother = head < NH ? float(qw[other]) : float(kw[other]);
    T xn = T(float(qkv[row + d]) * inv);
    T yn = T(float(qkv[row + other]) * inv);
    T x = T(float(xn) * w);
    T y = T(float(yn) * wother);
    float freq = metal::pow(BASE, -2.0f * float(d % (D / 2)) / float(D));
    float theta = float(offset + int(pos)) * freq;
    float out = float(x) * metal::cos(theta) + (d < D / 2 ? -float(y) : float(y)) * metal::sin(theta);
    uint dst = (oh * L + pos) * D + d;
    if (head < NH) q[dst] = T(out);
    else {
        k[dst] = T(out);
        v[dst] = qkv[pos * ((NH + 2 * NK) * D) + (NH + NK + oh) * D + d];
    }
}
'''

@lru_cache(None)
def get_kernel(eps, base):
    return mx.fast.metal_kernel(name='qk_norm_rope', input_names=['qkv','qw','kw','offset'],
                               output_names=['q','k','v'], source=SOURCE,
                               header=f'constant float EPS = {eps}f; constant float BASE = {base}f;')

def norm_rope(qkv, qw, kw, offset, args):
    length = qkv.shape[1]
    nh, nk, d = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    return get_kernel(args.rms_norm_eps, float(args.rope_theta))(inputs=[qkv,qw,kw,offset],
                  template=[('T',qkv.dtype),('L',length),('NH',nh),('NK',nk),('D',d)],
                  grid=(32,length,nh+nk), threadgroup=(32,1,1),
                  output_shapes=[(1,nh,length,d),(1,nk,length,d),(1,nk,length,d)],
                  output_dtypes=[qkv.dtype]*3)
