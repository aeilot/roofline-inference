"""Fused BF16 Metal kernels for single-token (M=1) Qwen3 decode.

Every kernel keeps BF16 weights/activations/KV and accumulates in float32.
Intermediate BF16 rounding points follow the MLX ops they replace (GEMV output,
RMSNorm `w * T(x * inv)`, residual add, SwiGLU), so the result differs from the
unfused graph only by float summation order inside the dot products.
"""
from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx

EPS = 1e-6


@lru_cache(maxsize=None)
def _gemv_kernel(K: int, R: int, NSG: int, pre: bool, epi: str, out_float: bool):
    """y = epi(W @ pre(x)); W is [N, K] row-major BF16.

    pre: RMSNorm of x with weight `nw` computed redundantly per simdgroup.
    epi: "none" | "res" (y = res + T(Wx)) | "swiglu" (rows j and j+F of W are
         gate/up; y_j = silu(T(g)) * T(u), output has F = N/2 rows).
    Each simdgroup owns R output rows; each lane covers K/32 columns in
    16-byte chunks so a row is read with fully coalesced vector loads.
    """
    assert K % 256 == 0
    C = K // 256
    names = ["W", "x"] + (["nw"] if pre else []) + (["res"] if epi == "res" else [])
    T = "bfloat16_t"
    OT = "float" if out_float else T
    lines = [f"""
    constexpr int C = {C};
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint j0 = (threadgroup_position_in_grid.x * {NSG} + sg) * {R};
    float xv[C * 8];
    float ss = 0.0f;
    for (int c = 0; c < C; ++c) {{
        int base = (c * 32 + lane) * 8;
        for (int j = 0; j < 8; ++j) {{
            float v = float(x[base + j]);
            xv[c * 8 + j] = v;
            ss += v * v;
        }}
    }}"""]
    if pre:
        lines.append(f"""
    ss = simd_sum(ss);
    float inv = metal::precise::rsqrt(ss / {K}.0f + {EPS!r}f);
    for (int c = 0; c < C; ++c) {{
        int base = (c * 32 + lane) * 8;
        for (int j = 0; j < 8; ++j)
            xv[c * 8 + j] = float({T}(float(nw[base + j]) * float({T}(xv[c * 8 + j] * inv))));
    }}""")
    rows = 2 * R if epi == "swiglu" else R
    if epi == "swiglu":
        row_of = f"(r < {R} ? j0 + r : j0 + (r - {R}) + F)"
    else:
        row_of = "j0 + r"
    lines.append(f"""
    constexpr int RR = {rows};
    {"const uint F = W_shape[0] / 2;" if epi == "swiglu" else ""}
    float acc[RR];
    for (int r = 0; r < RR; ++r) acc[r] = 0.0f;
    for (int c = 0; c < C; ++c) {{
        int base = (c * 32 + lane) * 8;
        float4 xa = float4(xv[c*8], xv[c*8+1], xv[c*8+2], xv[c*8+3]);
        float4 xb = float4(xv[c*8+4], xv[c*8+5], xv[c*8+6], xv[c*8+7]);
        for (int r = 0; r < RR; ++r) {{
            uint row = {row_of};
            const device packed_ushort4* p = (const device packed_ushort4*)(W + size_t(row) * {K} + base);
            ushort4 a = p[0];
            ushort4 b = p[1];
            acc[r] += dot(float4(as_type<bfloat4>(a)), xa) + dot(float4(as_type<bfloat4>(b)), xb);
        }}
    }}
    for (int r = 0; r < RR; ++r) acc[r] = simd_sum(acc[r]);
    if (lane == 0) {{""")
    if epi == "none":
        lines.append(f"        for (int r = 0; r < RR; ++r) y[j0 + r] = {OT}(acc[r]);")
    elif epi == "res":
        lines.append(f"        for (int r = 0; r < RR; ++r) y[j0 + r] = {T}(float(res[j0 + r]) + float({T}(acc[r])));")
    else:
        lines.append(f"""        for (int r = 0; r < {R}; ++r) {{
            float g = float({T}(acc[r]));
            float u = float({T}(acc[r + {R}]));
            float s = float({T}(1.0f / (1.0f + metal::exp(-g))));
            float a = float({T}(g * s));
            y[j0 + r] = {T}(a * u);
        }}""")
    lines.append("    }")
    return mx.fast.metal_kernel(
        name=f"gemv_K{K}_R{R}_S{NSG}_{int(pre)}_{epi}_{int(out_float)}",
        input_names=names, output_names=["y"], source="\n".join(lines))


# (R, NSG) tuned per K on M4 (see tune_gemv.py)
GEMV_CFG = {1024: (2, 8), 2048: (2, 4), 3072: (1, 4)}


def gemv(W, x, *, nw=None, res=None, swiglu=False, out_float=False, cfg=None):
    N, K = W.shape
    R, NSG = cfg or GEMV_CFG.get(K, (1, 4))
    epi = "swiglu" if swiglu else ("res" if res is not None else "none")
    n_out = N // 2 if swiglu else N
    assert n_out % (R * NSG) == 0, (N, R, NSG)
    kern = _gemv_kernel(K, R, NSG, nw is not None, epi, out_float)
    inputs = [W, x] + ([nw] if nw is not None else []) + ([res] if res is not None else [])
    return kern(inputs=inputs, output_shapes=[(n_out,)],
                output_dtypes=[mx.float32 if out_float else W.dtype],
                grid=(n_out // R * 32, 1, 1), threadgroup=(32 * NSG, 1, 1))[0]


_qk_src = """
    // qkv: [ (NQ + 2*NKV) * 128 ] BF16 projection output of one token.
    uint head = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    const device bfloat16_t* src = qkv + head * 128;
    float v[4];
    for (int i = 0; i < 4; ++i) v[i] = float(src[lane + 32 * i]);
    if (head >= NQ + NKV) {               // V: kv[1, h] == kv + (head - NQ) * 128
        uint o = (head - NQ) * 128;
        for (int i = 0; i < 4; ++i) kvo[o + lane + 32 * i] = bfloat16_t(v[i]);
        return;
    }
    const device bfloat16_t* w = head < NQ ? qw : kw;
    float ss = simd_sum(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]);
    float inv = metal::precise::rsqrt(ss / 128.0f + EPSV);
    float n[4];
    for (int i = 0; i < 4; ++i)
        n[i] = float(bfloat16_t(float(w[lane + 32 * i]) * float(bfloat16_t(v[i] * inv))));
    float L = float(pos[0]);
    float o[4];
    for (int p = 0; p < 2; ++p) {           // pair (d, d + 64), d = lane + 32p
        float d = float(lane + 32 * p) / 64.0f;
        float theta = L * metal::exp2(-d * LOG2BASE);
        float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
        float x1 = n[p], x2 = n[p + 2];
        o[p] = x1 * c - x2 * s;
        o[p + 2] = x1 * s + x2 * c;
    }
    device bfloat16_t* dst = head < NQ ? qo + head * 128 : kvo + (head - NQ) * 128;
    for (int i = 0; i < 4; ++i) dst[lane + 32 * i] = bfloat16_t(o[i]);
"""


@lru_cache(maxsize=None)
def _qk_kernel(nq, nkv, base):
    src = (_qk_src.replace("NQ", str(nq)).replace("NKV", str(nkv))
           .replace("EPSV", f"{EPS!r}f").replace("LOG2BASE", f"{math.log2(base)!r}f"))
    return mx.fast.metal_kernel(name=f"qk_norm_rope_{nq}_{nkv}", input_names=["qkv", "qw", "kw", "pos"],
                                output_names=["qo", "kvo"], source=src)


def qk_norm_rope(qkv, qw, kw, pos, nq, nkv, base):
    """Per-head RMSNorm on q/k, RoPE (non-traditional) at position `pos`; v copied.
    Returns q [1, nq, 1, 128] and kv [2, nkv, 1, 128] (K and V stacked)."""
    kern = _qk_kernel(nq, nkv, base)
    return kern(inputs=[qkv, qw, kw, pos],
                output_shapes=[(1, nq, 1, 128), (2, nkv, 1, 128)],
                output_dtypes=[qkv.dtype] * 2,
                grid=((nq + 2 * nkv) * 32, 1, 1), threadgroup=(32, 1, 1))


# ---------------- prefill (M > 1) helpers ----------------

_qk_prefill_src = """
    // One simdgroup per (token t, head). qkv: [L, (NQ + 2*NKV) * 128].
    // Outputs: Q [NQ, Lp, 128] (rows >= L untouched/padded), KV [2, NKV, L, 128].
    uint sgid = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    const int L = P[0], Lp = P[1], off = P[2];
    const uint NH = NQ + 2 * NKV;
    uint t = sgid / NH, head = sgid % NH;
    if (t >= uint(L)) return;
    const device bfloat16_t* src = qkv + size_t(t) * NH * 128 + head * 128;
    float v[4];
    for (int i = 0; i < 4; ++i) v[i] = float(src[lane + 32 * i]);
    if (head >= NQ + NKV) {
        device bfloat16_t* dst = kv + (size_t(NKV + head - NQ - NKV) * L + t) * 128;
        for (int i = 0; i < 4; ++i) dst[lane + 32 * i] = bfloat16_t(v[i]);
        return;
    }
    const device bfloat16_t* w = head < NQ ? qw : kw;
    float ss = simd_sum(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]);
    float inv = metal::precise::rsqrt(ss / 128.0f + EPSV);
    float n[4];
    for (int i = 0; i < 4; ++i)
        n[i] = float(bfloat16_t(float(w[lane + 32 * i]) * float(bfloat16_t(v[i] * inv))));
    float pos = float(off + int(t));
    float o[4];
    for (int p = 0; p < 2; ++p) {
        float d = float(lane + 32 * p) / 64.0f;
        float theta = pos * metal::exp2(-d * LOG2BASE);
        float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
        o[p] = n[p] * c - n[p + 2] * s;
        o[p + 2] = n[p] * s + n[p + 2] * c;
    }
    device bfloat16_t* dst = head < NQ ? q + (size_t(head) * Lp + t) * 128
                                       : kv + (size_t(head - NQ) * L + t) * 128;
    for (int i = 0; i < 4; ++i) dst[lane + 32 * i] = bfloat16_t(o[i]);
"""


@lru_cache(maxsize=None)
def _qk_prefill_kernel(nq, nkv, base):
    src = (_qk_prefill_src.replace("NQ", str(nq)).replace("NKV", str(nkv))
           .replace("EPSV", f"{EPS!r}f").replace("LOG2BASE", f"{math.log2(base)!r}f"))
    return mx.fast.metal_kernel(name=f"qk_norm_rope_prefill_{nq}_{nkv}",
                                input_names=["qkv", "qw", "kw", "P"], output_names=["q", "kv"], source=src)


def qk_norm_rope_prefill(qkv, qw, kw, offset, nq, nkv, base, Lp):
    """qkv [L, (nq+2nkv)*128] -> Q [nq, Lp, 128] (pad rows zero), KV [2, nkv, L, 128]."""
    L = qkv.shape[0]
    P = mx.array([L, Lp, offset], dtype=mx.int32)
    nsg = 4
    n_sg = L * (nq + 2 * nkv)
    return _qk_prefill_kernel(nq, nkv, base)(
        inputs=[qkv, qw, kw, P], output_shapes=[(nq, Lp, 128), (2, nkv, L, 128)],
        output_dtypes=[qkv.dtype] * 2, init_value=0,
        grid=((n_sg + nsg - 1) // nsg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1))


_add_rms_src = """
    // rows of width W (multiple of 1024 handled as W/256 per thread, 256 threads)
    uint row = threadgroup_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    constexpr int PER = W / 256;
    float vals[PER];
    float sum = 0.0f;
    for (int i = 0; i < PER; ++i) {
        uint idx = row * W + t * PER + i;
        bfloat16_t r = bfloat16_t(float(a[idx]) + float(b[idx]));
        residual[idx] = r;
        vals[i] = float(r);
        sum += vals[i] * vals[i];
    }
    sum = simd_sum(sum);
    threadgroup float partial[8];
    if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = simd_sum(thread_index_in_simdgroup < 8 ? partial[thread_index_in_simdgroup] : 0.0f);
    float scale = metal::precise::rsqrt(total / float(W) + EPSV);
    for (int i = 0; i < PER; ++i) {
        uint col = t * PER + i;
        normalized[row * W + col] = bfloat16_t(float(weight[col]) * float(bfloat16_t(vals[i] * scale)));
    }
"""


@lru_cache(maxsize=None)
def _add_rms_kernel(W):
    return mx.fast.metal_kernel(name=f"add_rms_{W}", input_names=["a", "b", "weight"],
                                output_names=["residual", "normalized"],
                                source=_add_rms_src.replace("EPSV", f"{EPS!r}f").replace("W)", f"{W})")
                                .replace("W /", f"{W} /").replace("* W", f"* {W}"))


def add_rms(a, b, weight):
    """(a + b, rms_norm(a + b) * weight) with MLX's BF16 rounding points."""
    W = a.shape[-1]
    rows = a.size // W
    return _add_rms_kernel(W)(inputs=[a, b, weight], output_shapes=[a.shape, a.shape],
                              output_dtypes=[a.dtype, a.dtype],
                              grid=(rows * 256, 1, 1), threadgroup=(256, 1, 1))


# ---------------- skinny GEMM for short prompts (8 < M <= 32) ----------------

_skinny_src = """
    // Y[M, N] = X[Mp, K] @ W[N, K]^T, Mp = MT*8 (X zero-padded); BF16 out, FP32 acc.
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint n0 = (threadgroup_position_in_grid.x * NSG + sg) * (NT * 8);
    simdgroup_matrix<float, 8, 8> acc[MT][NT];
    _Pragma("clang loop unroll(full)") for (int m = 0; m < MT; ++m)
        _Pragma("clang loop unroll(full)") for (int n = 0; n < NT; ++n) acc[m][n] = simdgroup_matrix<float, 8, 8>(0);
    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix<bfloat16_t, 8, 8> xm[MT], wm[NT];
        _Pragma("clang loop unroll(full)") for (int n = 0; n < NT; ++n)
            simdgroup_load(wm[n], W + size_t(n0 + n * 8) * K + k, K, ulong2(0, 0), true);
        _Pragma("clang loop unroll(full)") for (int m = 0; m < MT; ++m)
            simdgroup_load(xm[m], X + size_t(m * 8) * K + k, K);
        _Pragma("clang loop unroll(full)") for (int m = 0; m < MT; ++m)
            _Pragma("clang loop unroll(full)") for (int n = 0; n < NT; ++n)
                simdgroup_multiply_accumulate(acc[m][n], xm[m], wm[n], acc[m][n]);
    }
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    _Pragma("clang loop unroll(full)") for (int m = 0; m < MT; ++m) {
        int row = m * 8 + fm;
        if (row >= M) continue;
        _Pragma("clang loop unroll(full)") for (int n = 0; n < NT; ++n) {
            Y[size_t(row) * N + n0 + n * 8 + fn] = bfloat16_t(acc[m][n].thread_elements()[0]);
            Y[size_t(row) * N + n0 + n * 8 + fn + 1] = bfloat16_t(acc[m][n].thread_elements()[1]);
        }
    }
"""

_skinny = mx.fast.metal_kernel(name="skinny_gemm", input_names=["X", "W"], output_names=["Y"],
                               source=_skinny_src)
SKINNY_CFG = {}
try:  # produced by tune_skinny.py on this device
    import json as _json
    from pathlib import Path as _Path
    SKINNY_TABLE = {tuple(int(v) for v in k.split(",")): v for k, v in
                    _json.loads((_Path(__file__).with_name("skinny_table.json")).read_text()).items()}
except FileNotFoundError:
    SKINNY_TABLE = {}
for (_mt, _n, _k), _v in SKINNY_TABLE.items():
    SKINNY_CFG[(_mt, _n)] = (_v["NT"], _v["NSG"])


def use_skinny(M, N, K):
    e = SKINNY_TABLE.get(((M + 7) // 8, N, K))
    return bool(e and e["use"])


def skinny_matmul(x, W):
    """x [M, K] (8 < M <= 32) @ W[N, K]^T, bandwidth-oriented for short prompts."""
    M, K = x.shape
    N = W.shape[0]
    MT = (M + 7) // 8
    e = SKINNY_TABLE.get((MT, N, K))
    NT, NSG = (e["NT"], e["NSG"]) if e else SKINNY_CFG.get((MT, N), (1, 4))
    if M % 8:
        x = mx.pad(x, [(0, MT * 8 - M), (0, 0)])
    return _skinny(inputs=[x, W], template=[("K", K), ("N", N), ("M", M), ("MT", MT), ("NT", NT), ("NSG", NSG)],
                   output_shapes=[(M, N)], output_dtypes=[W.dtype],
                   grid=(N // (NT * 8) // NSG * 32 * NSG, 1, 1), threadgroup=(32 * NSG, 1, 1))[0]
