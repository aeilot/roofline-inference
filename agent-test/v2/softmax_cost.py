"""Is the long-prompt prefill gap explained by softmax ALU work missing from the bound?

Three variants of the SAME GQA flash-attention kernel (attention.py), identical
loads, MMAs, control flow and launch geometry; only the softmax differs:
  full    : production kernel
  no_exp  : per-score exp2(s - m) replaced by (s - m); scale/mask/max/sum/rescale kept
  no_soft : softmax removed; S fed straight into P.V (output meaningless)
Plus: register-only MMA peak, MLX GEMM, MLX SDPA for reference, and a sustained
60 s run of the full kernel and of the MMA peak to separate thermal throttling.
Writes softmax_cost.json.
"""
import json
import statistics
import time
from functools import lru_cache

import mlx.core as mx

import attention as A
import thermal

H, NKV, D = 16, 8, 128
SC = D ** -0.5

SOFT_START = "        const bool need_mask"


@lru_cache(maxsize=None)
def kernel(variant):
    src = A._src
    if variant == "no_exp":
        old = "fast::exp2(s - mnew)"
        assert src.count(old) == 1
        src = src.replace(old, "(s - mnew)")
    elif variant == "no_soft":
        # cut on the template source (before BK is substituted) so the markers match
        a = src.index(SOFT_START)
        vf = src.index("simdgroup_matrix<float, 8, 8> Vf;")
        pv = src.rfind("for (int kf = 0; kf < BK / 8; ++kf)", 0, vf)
        assert a < pv < vf
        src = src[:a] + src[src.rfind("\n", 0, pv) + 1:]
        # keep the normalizer finite: 1/0 under fast-math is UB
        old = "float lrow[2] = {0.0f, 0.0f};"
        assert src.count(old) == 1
        src = src.replace(old, "float lrow[2] = {1.0f, 1.0f};")
    src = (src.replace("NSG", "4").replace("BK", "16").replace("SCALE", f"{SC!r}f")
           .replace("#if KT", "#if 0").replace("#if VF32", "#if 0").replace("#if SKIP", "#if 1"))
    return mx.fast.metal_kernel(name=f"attn_variant_{variant}", input_names=["Q", "K", "V", "P"],
                                output_names=["O"], source=src)


def run(variant, Q, buf, qL, kL):
    P = mx.array([qL, kL, Q.shape[1], buf.shape[2], H], dtype=mx.int32)
    return kernel(variant)(inputs=[Q, buf[0:1], buf[1:2], P], output_shapes=[(qL, H * D)],
                           output_dtypes=[Q.dtype], grid=(Q.shape[1] // 32 * 128, NKV, 1),
                           threadgroup=(128, 1, 1))[0]


def timed(f, n=7):
    mx.eval(f())
    ts = []
    for _ in range(n):
        mx.synchronize(); s = time.perf_counter(); mx.eval(f()); mx.synchronize()
        ts.append(time.perf_counter() - s)
    return statistics.median(ts)


_mma_src = """
    simdgroup_matrix<bfloat16_t,8,8> a, b;
    simdgroup_load(a, A + (thread_position_in_grid.x / 32 % 4) * 64, 8);
    simdgroup_load(b, A + 64, 8);
    simdgroup_matrix<float,8,8> c[8];
    _Pragma("clang loop unroll(full)") for (int i = 0; i < 8; ++i) c[i] = simdgroup_matrix<float,8,8>(0);
    for (int it = 0; it < ITERS; ++it) {
        _Pragma("clang loop unroll(full)") for (int i = 0; i < 8; ++i) simdgroup_multiply_accumulate(c[i], a, b, c[i]);
    }
    float s = 0; _Pragma("clang loop unroll(full)") for (int i = 0; i < 8; ++i) s += c[i].thread_elements()[0];
    out[thread_position_in_grid.x] = s;
"""
_mma = mx.fast.metal_kernel(name="mma_peak_bf16", input_names=["A"], output_names=["out"], source=_mma_src)
_MA = mx.random.normal((256,)).astype(mx.bfloat16)
MMA_THREADS, MMA_ITERS = 10 * 1024 * 8, 4096
MMA_FLOP = MMA_THREADS / 32 * MMA_ITERS * 8 * 512 * 2


def mma_peak():
    return _mma(inputs=[_MA], template=[("ITERS", MMA_ITERS)], output_shapes=[(MMA_THREADS,)],
                output_dtypes=[mx.float32], grid=(MMA_THREADS, 1, 1), threadgroup=(256, 1, 1))[0]


def sustained(label, f, flop, seconds=60, window=5):
    thermal.wait_cool(idle=2, timeout=900)
    s = time.perf_counter(); last = s; n = 0; rates = []
    while time.perf_counter() - s < seconds:
        mx.eval(f()); n += 1
        now = time.perf_counter()
        if now - last >= window:
            rates.append(round(n * flop / (now - last) / 1e12, 3)); n = 0; last = now
    st = thermal.thermal_state()
    print(f"sustained {label}: TFLOP/s per {window}s {rates} thermal_after={st}", flush=True)
    return dict(window_s=window, tflops=rates, thermal_after=st)


def check_no_soft():
    """no_soft must really compute O = S.V over the visited key blocks (unmasked)."""
    qL = kL = 64
    buf = mx.random.normal((2, NKV, 256, D)).astype(mx.bfloat16)
    Q = mx.random.normal((H, qL, D)).astype(mx.bfloat16)
    o = run("no_soft", Q, buf, qL, kL).astype(mx.float32).reshape(qL, H, D)
    q = Q.astype(mx.float32); k = buf[0, :, :kL].astype(mx.float32); v = buf[1, :, :kL].astype(mx.float32)
    kk = mx.repeat(k, 2, axis=0); vv = mx.repeat(v, 2, axis=0)
    s = q @ kk.transpose(0, 2, 1)                     # [H, qL, kL], no scale, no mask
    r = mx.arange(qL)[:, None]
    q0 = r // 8 * 8                                   # simdgroup's first row
    lim = mx.minimum(((q0 + 7) // 16 + 1) * 16, r // 32 * 32 + 32)   # BK=16 blocks visited
    blk = lim > mx.arange(kL)[None]
    ref = ((s * blk) @ vv).transpose(1, 0, 2)
    rel = (mx.max(mx.abs(o - ref)) / mx.max(mx.abs(ref))).item()
    assert rel < 0.02, rel
    return rel


def main():
    out = dict(note=__doc__, shapes=[], sustained={})
    out["no_soft_check_rel_err"] = check_no_soft()
    print("no_soft computes S.V, rel err", out["no_soft_check_rel_err"], flush=True)
    thermal.wait_cool(idle=2)
    out["mma_peak_tflops"] = [round(MMA_FLOP / timed(mma_peak) / 1e12, 3) for _ in range(3)]
    print("MMA peak", out["mma_peak_tflops"], flush=True)
    x = mx.random.normal((2048, 1024)).astype(mx.bfloat16)
    w = mx.random.normal((6144, 1024)).astype(mx.bfloat16)
    mx.eval(x, w)
    out["mlx_gemm_tflops"] = round(2 * 2048 * 1024 * 6144 * 8 / timed(lambda: [x @ w.T for _ in range(8)]) / 1e12, 3)
    print("MLX GEMM", out["mlx_gemm_tflops"], flush=True)

    for qL, kL in [(2048, 2048), (2048, 8192), (2048, 32768)]:
        cap = (kL + 255) // 256 * 256
        buf = mx.zeros((2, NKV, cap, D), mx.bfloat16)
        k = mx.random.normal((1, NKV, kL, D)).astype(mx.bfloat16)
        v = mx.random.normal((1, NKV, kL, D)).astype(mx.bfloat16)
        buf[0:1, :, :kL] = k
        buf[1:2, :, :kL] = v
        q = (1.5 * mx.random.normal((1, H, qL, D))).astype(mx.bfloat16)
        Q = q[0]
        mx.eval(buf, k, v, Q)
        flop = 4 * D * H * sum(min(kL, kL - qL + i + 1) for i in range(qL))
        scores = flop // (4 * D)
        fns = {"full": lambda: run("full", Q, buf, qL, kL),
               "no_exp": lambda: run("no_exp", Q, buf, qL, kL),
               "no_soft": lambda: run("no_soft", Q, buf, qL, kL),
               "mlx_sdpa": lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=SC, mask="causal")}
        times = {name: [] for name in fns}
        thermal.wait_cool(idle=2)
        for rep in range(5):                      # interleave variants, rotate order
            names = list(fns)[rep % 4:] + list(fns)[:rep % 4]
            for name in names:
                times[name].append(timed(fns[name], n=3))
        t = {name: statistics.median(v) for name, v in times.items()}
        row = dict(qL=qL, kL=kL, flop=flop, scores=scores,
                   time_ms={n: round(v * 1e3, 3) for n, v in t.items()},
                   tflops={n: round(flop / v / 1e12, 3) for n, v in t.items()},
                   softmax_share_of_full=round((t["full"] - t["no_soft"]) / t["full"], 4),
                   exp_share_of_full=round((t["full"] - t["no_exp"]) / t["full"], 4),
                   ns_per_score_softmax=round((t["full"] - t["no_soft"]) / scores * 1e9, 5),
                   thermal_after=thermal.thermal_state())
        out["shapes"].append(row)
        print(json.dumps(row), flush=True)
        if kL == 32768:
            out["sustained"]["full_attention_2048x32768"] = sustained(
                "attention full", lambda: run("full", Q, buf, qL, kL), flop)
    out["sustained"]["mma_peak"] = sustained("MMA peak", mma_peak, MMA_FLOP)
    json.dump(out, open("softmax_cost.json", "w"), indent=1)


if __name__ == "__main__":
    main()
