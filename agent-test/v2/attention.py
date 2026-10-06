"""GQA-paired causal flash attention for prefill (BF16 in/out, FP32 softmax/accumulate).

One simdgroup owns 8 query positions for BOTH query heads that share a KV head,
so every K/V fragment read from threadgroup memory feeds two MMAs. Q stays in
registers as BF16 (BF16 x BF16 products are exact in FP32, so S = QK^T matches
an FP32-input MMA); P.V uses FP32 P and FP32-converted V like MLX's kernel.
Causal mask is bottom-right aligned (query i sees keys <= kL - qL + i).

Layouts: Q [H, qLp, D] (qLp = qL padded to BQ), K/V cache [NKV, cap, D],
output O [qL, H * D] (directly the o_proj input).
"""
from functools import lru_cache

import mlx.core as mx

D = 128

_src = """
    constexpr int D = 128;
    constexpr int LD = D + 8;
    constexpr int BQ = NSG * 8;
    const int qL = P[0], kL = P[1], qLp = P[2], cap = P[3];
    const int H = P[4];
    const int off = kL - qL;
    const uint nqb = threadgroup_position_in_grid.x;
    const uint qb = (qLp / BQ) - 1 - nqb;            // heavy (late) blocks first
    const uint g = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const int q0 = qb * BQ + sg * 8;

#if KT
    constexpr int LDK = BK + 8;
    threadgroup bfloat16_t Ks[D * LDK];             // K^T: [d][key]
#else
    threadgroup bfloat16_t Ks[BK * LD];
#endif
#if VF32
    constexpr int LDV = D + 4;
    threadgroup float Vs[BK * LDV];
#else
    threadgroup bfloat16_t Vs[BK * LD];
#endif

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    simdgroup_matrix<bfloat16_t, 8, 8> Qf[2][16];
    _Pragma("clang loop unroll(full)") for (int h = 0; h < 2; ++h)
        _Pragma("clang loop unroll(full)") for (int d = 0; d < 16; ++d)
            simdgroup_load(Qf[h][d], Q + (size_t((2 * g + h) * qLp + q0)) * D + d * 8, D);
    simdgroup_matrix<float, 8, 8> Of[2][16];
    _Pragma("clang loop unroll(full)") for (int h = 0; h < 2; ++h)
        _Pragma("clang loop unroll(full)") for (int d = 0; d < 16; ++d) Of[h][d] = simdgroup_matrix<float, 8, 8>(0);
    float mrow[2] = {-INFINITY, -INFINITY};
    float lrow[2] = {0.0f, 0.0f};
    const float sc = SCALE * 1.4426950408889634f;
    const int kend = min(kL, off + int(qb) * BQ + BQ);
    const int my_last = off + q0 + fm;
    const device bfloat16_t* Kg = K + size_t(g) * cap * D;
    const device bfloat16_t* Vg = V + size_t(g) * cap * D;

    for (int k0 = 0; k0 < kend; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < BK * 16; i += NSG * 32) {
            int r = i / 16, c = (i % 16) * 8;
#if KT
            {
                const device bfloat16_t* src = Kg + size_t(k0 + r) * D + c;
                _Pragma("clang loop unroll(full)") for (int j = 0; j < 8; ++j) Ks[(c + j) * LDK + r] = src[j];
            }
#else
            *(threadgroup uint4*)&Ks[r * LD + c] = *(const device uint4*)(Kg + size_t(k0 + r) * D + c);
#endif
#if VF32
            {
                const device bfloat16_t* src = Vg + size_t(k0 + r) * D + c;
                _Pragma("clang loop unroll(full)") for (int j = 0; j < 8; ++j) Vs[r * LDV + c + j] = float(src[j]);
            }
#else
            *(threadgroup uint4*)&Vs[r * LD + c] = *(const device uint4*)(Vg + size_t(k0 + r) * D + c);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 > off + q0 + 7) continue;              // whole block masked for this simdgroup

        simdgroup_matrix<float, 8, 8> S[2][BK / 8];
        _Pragma("clang loop unroll(full)") for (int h = 0; h < 2; ++h)
            _Pragma("clang loop unroll(full)") for (int kf = 0; kf < BK / 8; ++kf) S[h][kf] = simdgroup_matrix<float, 8, 8>(0);
        _Pragma("clang loop unroll(full)") for (int d = 0; d < 16; ++d) {
            _Pragma("clang loop unroll(full)") for (int kf = 0; kf < BK / 8; ++kf) {
                simdgroup_matrix<bfloat16_t, 8, 8> Kt;
#if KT
                simdgroup_load(Kt, &Ks[d * 8 * LDK + kf * 8], LDK);
#else
                simdgroup_load(Kt, &Ks[kf * 8 * LD + d * 8], LD, ulong2(0, 0), true);
#endif
                simdgroup_multiply_accumulate(S[0][kf], Qf[0][d], Kt, S[0][kf]);
                simdgroup_multiply_accumulate(S[1][kf], Qf[1][d], Kt, S[1][kf]);
            }
        }
        const bool need_mask = (k0 + BK - 1 > off + q0) || (k0 + BK > kL);
        _Pragma("clang loop unroll(full)") for (int h = 0; h < 2; ++h) {
            float rmax = -INFINITY;
            _Pragma("clang loop unroll(full)") for (int kf = 0; kf < BK / 8; ++kf) {
                _Pragma("clang loop unroll(full)") for (int j = 0; j < 2; ++j) {
                    float s = S[h][kf].thread_elements()[j] * sc;
                    int key = k0 + kf * 8 + fn + j;
                    if (need_mask && (key > my_last || key >= kL)) s = -INFINITY;
                    S[h][kf].thread_elements()[j] = s;
                    rmax = max(rmax, s);
                }
            }
            rmax = max(rmax, simd_shuffle_xor(rmax, ushort(1)));
            rmax = max(rmax, simd_shuffle_xor(rmax, ushort(8)));
            float mnew = max(mrow[h], rmax);
            float factor = (mnew == -INFINITY) ? 1.0f : fast::exp2(mrow[h] - mnew);
            float rsum = 0.0f;
            _Pragma("clang loop unroll(full)") for (int kf = 0; kf < BK / 8; ++kf) {
                _Pragma("clang loop unroll(full)") for (int j = 0; j < 2; ++j) {
                    float s = S[h][kf].thread_elements()[j];
                    float p = (mnew == -INFINITY) ? 0.0f : fast::exp2(s - mnew);
                    S[h][kf].thread_elements()[j] = p;
                    rsum += p;
                }
            }
            rsum += simd_shuffle_xor(rsum, ushort(1));
            rsum += simd_shuffle_xor(rsum, ushort(8));
            lrow[h] = lrow[h] * factor + rsum;
            mrow[h] = mnew;
#if SKIP
            if (simd_all(factor == 1.0f)) continue;
#endif
            _Pragma("clang loop unroll(full)") for (int d = 0; d < 16; ++d) {
                Of[h][d].thread_elements()[0] *= factor;
                Of[h][d].thread_elements()[1] *= factor;
            }
        }
        _Pragma("clang loop unroll(full)") for (int kf = 0; kf < BK / 8; ++kf) {
            _Pragma("clang loop unroll(full)") for (int d = 0; d < 16; ++d) {
                simdgroup_matrix<float, 8, 8> Vf;
#if VF32
                simdgroup_load(Vf, &Vs[kf * 8 * LDV + d * 8], LDV);
#else
                Vf.thread_elements()[0] = float(Vs[(kf * 8 + fm) * LD + d * 8 + fn]);
                Vf.thread_elements()[1] = float(Vs[(kf * 8 + fm) * LD + d * 8 + fn + 1]);
#endif
                simdgroup_multiply_accumulate(Of[0][d], S[0][kf], Vf, Of[0][d]);
                simdgroup_multiply_accumulate(Of[1][d], S[1][kf], Vf, Of[1][d]);
            }
        }
    }
    const int row = q0 + fm;
    if (row < qL) {
        _Pragma("clang loop unroll(full)") for (int h = 0; h < 2; ++h) {
            float inv = 1.0f / lrow[h];
            device bfloat16_t* dst = O + size_t(row) * (H * D) + (2 * g + h) * D;
            _Pragma("clang loop unroll(full)") for (int d = 0; d < 16; ++d) {
                dst[d * 8 + fn] = bfloat16_t(Of[h][d].thread_elements()[0] * inv);
                dst[d * 8 + fn + 1] = bfloat16_t(Of[h][d].thread_elements()[1] * inv);
            }
        }
    }
"""


@lru_cache(maxsize=None)
def _kernel(nsg, bk, scale, kt, vf32, skip):
    src = _src.replace("NSG", str(nsg)).replace("BK", str(bk)).replace("SCALE", f"{scale!r}f")
    src = src.replace("#if KT", f"#if {int(kt)}").replace("#if VF32", f"#if {int(vf32)}").replace("#if SKIP", f"#if {int(skip)}")
    return mx.fast.metal_kernel(name=f"gqa_flash_{nsg}_{bk}_{int(kt)}{int(vf32)}{int(skip)}",
                                input_names=["Q", "K", "V", "P"], output_names=["O"], source=src)


def attention(Q, K, V, qL, kL, scale, nsg=4, bk=16, kt=False, vf32=False, skip=True):
    """Q [H, qLp, D] BF16; K/V full cache buffers [1, NKV, cap, D]; returns [qL, H*D]."""
    H, qLp, _ = Q.shape
    nkv, cap = K.shape[1], K.shape[2]
    assert H == 2 * nkv and qLp % (nsg * 8) == 0 and cap % bk == 0
    P = mx.array([qL, kL, qLp, cap, H], dtype=mx.int32)
    return _kernel(nsg, bk, float(scale), kt, vf32, skip)(
        inputs=[Q, K, V, P], output_shapes=[(qL, H * D)], output_dtypes=[Q.dtype],
        grid=(qLp // (nsg * 8) * nsg * 32, nkv, 1), threadgroup=(nsg * 32, 1, 1))[0]
