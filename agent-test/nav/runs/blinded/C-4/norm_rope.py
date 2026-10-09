"""BF16 head normalization, RoPE and Q/K/V layout conversion in one kernel."""
import math
import mlx.core as mx

def make_kernel(heads, kvheads, eps, base):
    source = f'''constexpr float EPS = {eps:.18g}f;
constexpr float LOG_BASE = {math.log(base):.18g}f;
''' + '''
    const uint head = threadgroup_position_in_grid.x;
    const uint token = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    constexpr uint WIDTH = (H + 2 * HK) * 128;
    const bool is_k = head >= H;
    const device T* weight = is_k ? kw : qw;
    const uint input = token * WIDTH + head * 128 + lane;
    float x0 = float(packed[input]);
    float x1 = float(packed[input + 32]);
    float x2 = float(packed[input + 64]);
    float x3 = float(packed[input + 96]);
    float sum = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    float scale = metal::rsqrt(simd_sum(sum) / 128.0f + EPS);
    // Preserve the BF16 rounding between head RMS normalization and RoPE.
    x0 = float(T(float(T(x0 * scale)) * float(weight[lane])));
    x1 = float(T(float(T(x1 * scale)) * float(weight[lane + 32])));
    x2 = float(T(float(T(x2 * scale)) * float(weight[lane + 64])));
    x3 = float(T(float(T(x3 * scale)) * float(weight[lane + 96])));
    const float position = float(pos + token);
    float angle0 = position * metal::exp(-float(lane) * LOG_BASE / 64.0f);
    float angle1 = position * metal::exp(-float(lane + 32) * LOG_BASE / 64.0f);
    float cs0 = metal::cos(angle0), sn0 = metal::sin(angle0);
    float cs1 = metal::cos(angle1), sn1 = metal::sin(angle1);
    device T* output = is_k ? kout : qout;
    const uint target = ((is_k ? head - H : head) * L + token) * 128 + lane;
    output[target] = T(x0 * cs0 - x2 * sn0);
    output[target + 64] = T(x0 * sn0 + x2 * cs0);
    output[target + 32] = T(x1 * cs1 - x3 * sn1);
    output[target + 96] = T(x1 * sn1 + x3 * cs1);
    if (is_k) {
        const uint input_v = token * WIDTH + (H + HK + head - H) * 128 + lane;
        vout[target] = packed[input_v];
        vout[target + 32] = packed[input_v + 32];
        vout[target + 64] = packed[input_v + 64];
        vout[target + 96] = packed[input_v + 96];
    }
    '''
    kernel = mx.fast.metal_kernel(name='fused_qkv_norm_rope',
        input_names=['packed', 'qw', 'kw', 'pos'],
        output_names=['qout', 'kout', 'vout'], source=source)
    def apply(packed, qw, kw, pos):
        length = packed.shape[1]
        return kernel(inputs=[packed, qw, kw, pos],
            template=[('T', packed.dtype), ('H', heads), ('HK', kvheads), ('L', length)],
            grid=((heads + kvheads) * 32, length, 1), threadgroup=(32, 1, 1),
            output_shapes=[(1, heads, length, 128), (1, kvheads, length, 128), (1, kvheads, length, 128)],
            output_dtypes=[packed.dtype] * 3)
    return apply
