"""Fused BF16 residual addition + RMSNorm for Qwen3's hidden dimension.

Round the residual to BF16 BEFORE accumulating squares, just like the separate
BF16 Add and RMSNorm. Reduction uses float32. No weight/KV precision change.
"""
import mlx.core as mx


_add_rms = mx.fast.metal_kernel(
    name="agent_add_rms",
    input_names=["a", "b", "weight", "eps"],
    output_names=["residual", "normalized"],
    source="""
        uint row = threadgroup_position_in_grid.x;
        uint t = thread_position_in_threadgroup.x;
        float vals[4];
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            uint col = t * 4 + i;
            uint idx = row * 1024 + col;
            T r = T(float(a[idx]) + float(b[idx]));
            residual[idx] = r;
            vals[i] = float(r);
            sum += vals[i] * vals[i];
        }
        sum = simd_sum(sum);
        threadgroup float partial[8];
        uint lane = thread_index_in_simdgroup;
        uint group = simdgroup_index_in_threadgroup;
        if (lane == 0) partial[group] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
        float scale = metal::precise::rsqrt(total / 1024.0f + eps);
        for (uint i = 0; i < 4; ++i) {
            uint col = t * 4 + i;
            normalized[row * 1024 + col] = T(float(T(vals[i] * scale)) * float(weight[col]));
        }
    """,
)


def add_rms(a, b, norm):
    if a.shape[-1] != 1024:
        residual = a + b
        return residual, norm(residual)
    rows = a.size // 1024
    return _add_rms(inputs=[a, b, norm.weight, mx.array(norm.eps, dtype=mx.float32)],
                    template=[("T", a.dtype)],
                    output_shapes=[a.shape, a.shape],
                    output_dtypes=[a.dtype, a.dtype],
                    grid=(rows * 256, 1, 1), threadgroup=(256, 1, 1))
