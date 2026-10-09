"""A BF16-input, FP32-accumulating GEMM for sixteen prompt rows.

The 16 x 64 output tile uses one SIMD group, with no padded rows or
threadgroup-memory staging. All products and intermediate BF16 rounding
points are retained.
"""
import mlx.core as mx

_kernel = mx.fast.metal_kernel(
    name="qwen_prompt16_gemm",
    input_names=["x", "w"],
    output_names=["y"],
    header="#include <metal_simdgroup_matrix>\n",
    source=r"""
    using namespace metal;
    uint row = threadgroup_position_in_grid.y * 16;
    uint col = threadgroup_position_in_grid.x * 64;
    ushort lane = thread_index_in_simdgroup;
    ushort qid = lane / 4;
    ushort ry = (qid & 4) + ((lane / 2) % 4);
    ushort rx = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_matrix<float, 8, 8> accum[2][8];
    simdgroup_matrix<float, 8, 8> aa[2], bb[8];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < 8; ++j) {
            accum[i][j].thread_elements()[0] = 0.0f;
            accum[i][j].thread_elements()[1] = 0.0f;
        }
    }
    #pragma clang loop unroll(disable)
    for (int kk = 0; kk < K; kk += 8) {
        #pragma clang loop unroll(full)
        for (short i = 0; i < 2; ++i) {
            uint pos = (row + i * 8 + ry) * K + kk + rx;
            aa[i].thread_elements()[0] = float(x[pos]);
            aa[i].thread_elements()[1] = float(x[pos + 1]);
        }
        #pragma clang loop unroll(full)
        for (short j = 0; j < 8; ++j) {
            uint pos = (col + j * 8 + rx) * K + kk + ry;
            bb[j].thread_elements()[0] = float(w[pos]);
            bb[j].thread_elements()[1] = float(w[pos + K]);
        }
        #pragma clang loop unroll(full)
        for (short i = 0; i < 2; ++i) {
            #pragma clang loop unroll(full)
            for (short j = 0; j < 8; ++j) {
                simdgroup_multiply_accumulate(accum[i][j], aa[i], bb[j], accum[i][j]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (short i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < 8; ++j) {
            uint pos = (row + i * 8 + ry) * N + col + j * 8 + rx;
            y[pos] = T(accum[i][j].thread_elements()[0]);
            y[pos + 1] = T(accum[i][j].thread_elements()[1]);
        }
    }
    """,
)


def linear(x, weight):
    if x.size // x.shape[-1] != 16:
        return x @ weight.T
    n, k = weight.shape
    if n % 64 or k % 8 or x.dtype != mx.bfloat16:
        return x @ weight.T
    shape = x.shape[:-1] + (n,)
    return _kernel(
        inputs=[x, weight],
        template=[("T", mx.bfloat16), ("N", n), ("K", k)],
        grid=(n // 64 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.bfloat16],
    )[0]
