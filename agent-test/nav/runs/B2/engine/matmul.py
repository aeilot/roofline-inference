"""A 16-row BF16 GEMM using MLX's float-accumulating Metal matrix helpers.

The stock path is retained for other row counts. The 16-row tile avoids doing
work for the unused rows of a larger tile on short prefills.
"""
from pathlib import Path

import mlx.core as mx

_kernel = mx.fast.metal_kernel(
    name="prefill16_gemm",
    input_names=["x", "w"],
    output_names=["out"],
    header=Path(__file__).with_name("gemm.metal").read_text() + "\n",
    source="""
    using namespace mlx::steel;
    const uint3 tid = threadgroup_position_in_grid;
    const uint lane = thread_index_in_simdgroup;
    using gemm = GEMMKernel<T, T, 16, 32, 64, 1, 1, false, true, true, true, float>;
    threadgroup T As[gemm::tgp_mem_size_a];
    threadgroup T Bs[gemm::tgp_mem_size_b];
    typename gemm::loader_a_t la(x + tid.y * 16 * K, K, As, 0, lane);
    typename gemm::loader_b_t lb(w + tid.x * 32 * K, K, Bs, 0, lane);
    typename gemm::mma_t mma(0, lane);
    gemm::template gemm_loop<true, true, true>(As, Bs, K / 64, la, lb, mma, 16, 32, 0);
    mma.store_result(out + tid.y * 16 * N + tid.x * 32, N);
    """,
)


def linear(x, w):
    rows, inner = x.shape[-2:]
    cols = w.shape[0]
    if rows != 16 or x.dtype != mx.bfloat16 or w.dtype != mx.bfloat16:
        return x @ w.T
    if inner % 64 or cols % 32:
        return x @ w.T
    return _kernel(
        inputs=[x, w],
        template=[("T", x.dtype), ("K", inner), ("N", cols)],
        grid=(cols, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[x.shape[:-1] + (cols,)],
        output_dtypes=[x.dtype],
    )[0]
