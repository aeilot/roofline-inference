"""Per-kernel cost of the operations an engine actually runs (plan item 13).

P(shape) for prefill and per-kernel time for decode, using MLX's own kernels
(the ones mlx-lm dispatches) at Qwen3-0.6B shapes:

  gemm         mx.matmul (M x K) @ (K x N), bf16, for every projection and M
  sdpa         mx.fast.scaled_dot_product_attention: decode (1 query, Lk keys)
               and prefill chunks (Lq queries, causal, Lk keys)
  small        RMSNorm, RoPE, add, SwiGLU, KV-cache slice update at M rows
  exp          peak exp throughput (custom Metal kernel), the softmax resource

Each sample evaluates R copies of the operation on distinct operands (a pool
>= 128 MiB, so decode-sized reads come from DRAM as decode weights do) in one
mx.eval, and divides by R. Copies are independent: decode-sized kernels of
>= 1 MiB lose < 5% to dependency barriers (kernel_chain.py), smaller ones are
reported both ways (chain mode: each op consumes the previous result).

    python benchmarks/compute_shapes.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.activations import swiglu

HIDDEN, INTER, Q_DIM, KV_DIM, HEADS, KV_HEADS, HEAD_DIM, VOCAB = 1024, 3072, 2048, 1024, 16, 8, 128, 151936
# (name, K, N) as mlx-lm runs them: separate q/k/v and gate/up projections.
PROJ = [("q_proj", HIDDEN, Q_DIM), ("k_proj", HIDDEN, KV_DIM), ("v_proj", HIDDEN, KV_DIM),
        ("o_proj", Q_DIM, HIDDEN), ("gate_proj", HIDDEN, INTER), ("up_proj", HIDDEN, INTER),
        ("down_proj", INTER, HIDDEN), ("lm_head", HIDDEN, VOCAB)]
POOL = 160 * 2**20


def timeit(fn, trials: int) -> float:
    for _ in range(2):
        mx.eval(fn())
    s = []
    for _ in range(trials):
        out = fn()
        t0 = time.perf_counter()
        mx.eval(out)
        s.append(time.perf_counter() - t0)
    return min(s)


def copies(nbytes: int, cap: int = 64) -> int:
    return max(4, min(cap, POOL // max(nbytes, 1)))


def bench_gemm(M: int, K: int, N: int, trials: int) -> dict:
    w_bytes = K * N * 2
    R = copies(w_bytes) if M <= 64 else max(2, min(16, int(4e11 / (2 * M * K * N))))
    Ws = [mx.random.normal((N, K)).astype(mx.bfloat16) for _ in range(R)]  # mlx-lm stores [out, in]
    x = mx.random.normal((M, K)).astype(mx.bfloat16)
    mx.eval(x, *Ws)
    t = timeit(lambda: [x @ W.T for W in Ws], trials) / R
    flops = 2 * M * K * N
    return {"M": M, "K": K, "N": N, "time_s": t, "tflops": flops / t / 1e12,
            "weight_GBps": w_bytes / t / 1e9, "copies": R}


def bench_sdpa(Lq: int, Lk: int, trials: int) -> dict:
    kv_bytes = 2 * KV_HEADS * Lk * HEAD_DIM * 2
    R = copies(kv_bytes, cap=16) if Lq == 1 else 2
    q = mx.random.normal((1, HEADS, Lq, HEAD_DIM)).astype(mx.bfloat16)
    kvs = [(mx.random.normal((1, KV_HEADS, Lk, HEAD_DIM)).astype(mx.bfloat16),
            mx.random.normal((1, KV_HEADS, Lk, HEAD_DIM)).astype(mx.bfloat16)) for _ in range(R)]
    mx.eval(q, *[a for kv in kvs for a in kv])
    mask = "causal" if Lq > 1 else None
    t = timeit(lambda: [mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5, mask=mask)
                        for k, v in kvs], trials) / R
    # Causal chunk: query i (of Lq, aligned to the end of Lk) sees Lk - Lq + i + 1 keys.
    pairs = Lq * Lk - Lq * (Lq - 1) // 2 if Lq > 1 else Lk
    flops = 4 * HEADS * HEAD_DIM * pairs
    return {"Lq": Lq, "Lk": Lk, "time_s": t, "tflops": flops / t / 1e12, "kv_GBps": kv_bytes / t / 1e9,
            "copies": R}


def bench_small(M: int, trials: int) -> dict:
    """Per-op time of the small decode/prefill kernels at M rows, independent
    copies and dependent chain."""
    R = 64 if M <= 64 else 8
    x = mx.random.normal((M, HIDDEN)).astype(mx.bfloat16)
    g = mx.random.normal((M, INTER)).astype(mx.bfloat16)
    w = mx.ones((HIDDEN,), dtype=mx.bfloat16)
    q = mx.random.normal((1, HEADS, M, HEAD_DIM)).astype(mx.bfloat16)
    cache = mx.zeros((1, KV_HEADS, 4096 + M, HEAD_DIM), dtype=mx.bfloat16)
    kv = mx.random.normal((1, KV_HEADS, M, HEAD_DIM)).astype(mx.bfloat16)
    mx.eval(x, g, w, q, cache, kv)
    ops = {
        "rms_norm": (lambda a: mx.fast.rms_norm(a, w, 1e-6), x),
        "rope": (lambda a: mx.fast.rope(a, HEAD_DIM, traditional=False, base=1e6, scale=1.0, offset=7), q),
        "add": (lambda a: a + x, x),
        "swiglu": (lambda a: swiglu(a, g), g),
        "slice_update": (lambda c: mx.slice_update(c, kv, mx.array([0, 0, 4000, 0]), (0, 1, 2, 3)), cache),
    }
    out = {}
    for name, (f, a0) in ops.items():
        indep = timeit(lambda: [f(a0 + 0) if name == "slice_update" else f(a0) for _ in range(R)], trials) / R

        def chain():
            a = a0
            for _ in range(R):
                a = f(a)
            return a

        out[name] = {"independent_s": indep, "chain_s": timeit(chain, trials) / R}
    return {"M": M, **out}


_exp = mx.fast.metal_kernel(
    name="exp_peak",
    input_names=["seed"],
    output_names=["out"],
    source="""
        uint tid = thread_position_in_grid.x;
        float a0 = seed[0] + tid * 1e-7f, a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
        float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;
        for (uint i = 0; i < ITERS; ++i) {
            a0 = metal::fast::exp(a0) * -0.5f; a1 = metal::fast::exp(a1) * -0.5f;
            a2 = metal::fast::exp(a2) * -0.5f; a3 = metal::fast::exp(a3) * -0.5f;
            a4 = metal::fast::exp(a4) * -0.5f; a5 = metal::fast::exp(a5) * -0.5f;
            a6 = metal::fast::exp(a6) * -0.5f; a7 = metal::fast::exp(a7) * -0.5f;
        }
        out[tid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    """,
)


def bench_exp(trials: int) -> dict:
    """exp/s with 8 independent chains per thread; each step is one exp and
    one multiply, so the multiply is counted against the result too."""
    threads, iters = 4096 * 256, 2048
    seed = mx.array([0.01], dtype=mx.float32)
    f = lambda: _exp(inputs=[seed], template=[("ITERS", iters)], output_shapes=[(threads,)],  # noqa: E731
                     output_dtypes=[mx.float32], grid=(threads, 1, 1), threadgroup=(256, 1, 1))[0]
    t = timeit(f, trials)
    return {"exp_per_s": threads * iters * 8 / t, "time_s": t, "note": "fast::exp + fmul per step"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, nargs="*", default=[1, 16, 64, 256, 512, 1024, 2048, 4096])
    p.add_argument("--sdpa-lk", type=int, nargs="*", default=[512, 1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--out", default="results/raw")
    args = p.parse_args()

    load1 = os.getloadavg()[0]
    print(f"device: {mx.device_info()['device_name']}, load avg (1 min): {load1:.2f}")
    res: dict = {"gemm": {}, "sdpa": [], "small": [], "exp": None}

    print(f"\n{'gemm':<10}{'M':>6}{'us':>11}{'TFLOP/s':>9}{'W GB/s':>8}")
    for name, K, N in PROJ:
        rows = []
        for M in args.M:
            if name == "lm_head" and M > 64:
                continue
            r = bench_gemm(M, K, N, args.trials)
            rows.append(r)
            print(f"{name:<10}{M:>6}{r['time_s'] * 1e6:>11.1f}{r['tflops']:>9.2f}{r['weight_GBps']:>8.1f}", flush=True)
        res["gemm"][name] = rows

    print(f"\n{'sdpa Lq':>8}{'Lk':>7}{'ms':>10}{'TFLOP/s':>9}{'KV GB/s':>9}")
    pairs = [(1, lk) for lk in args.sdpa_lk]
    pairs += [(min(lk, 2048), lk) for lk in args.sdpa_lk] + [(lq, lq) for lq in (16, 64, 256)]
    for lq, lk in pairs:
        r = bench_sdpa(lq, lk, args.trials)
        res["sdpa"].append(r)
        print(f"{lq:>8}{lk:>7}{r['time_s'] * 1e3:>10.3f}{r['tflops']:>9.2f}{r['kv_GBps']:>9.1f}", flush=True)

    print(f"\n{'small M':>8}" + "".join(f"{k:>22}" for k in ("rms_norm", "rope", "add", "swiglu", "slice_update")))
    for M in (1, 16, 64, 512, 2048):
        r = bench_small(M, args.trials)
        res["small"].append(r)
        print(f"{M:>8}" + "".join(f"{r[k]['independent_s'] * 1e6:>10.1f} / {r[k]['chain_s'] * 1e6:>7.1f}us"
                                  for k in ("rms_norm", "rope", "add", "swiglu", "slice_update")), flush=True)

    res["exp"] = bench_exp(args.trials)
    print(f"\nexp: {res['exp']['exp_per_s'] / 1e12:.3f} T exp/s")

    record = {
        "benchmark": "compute_shapes",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": mx.device_info()["device_name"],
        "mlx_version": mx.__version__,
        "macos": platform.mac_ver()[0], "load_avg_1min": load1,
        "args": vars(args), "results": res,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"compute_shapes-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(record, indent=1))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
