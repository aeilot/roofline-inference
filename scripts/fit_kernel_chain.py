"""Fit the kernel-chain cost model (design notes 5.3, plan items 11-12).

Input: results/raw/kernel_chain-*.json from benchmarks/kernel_chain.py.

    CPU   submit(N, K) = K * (N * e + c_commit)            least squares on submit times
    GPU   g(b) = t0 + b / B_eff                             per-b slope of (T - submit) vs N, K = 1
          c_sync                                            intercept of the same fit
    T(N, b, K) = pipeline: buffer k is committed at k * (N e + c_commit) and runs
                 N g(b) on the GPU after the previous buffer; + c_sync

Checks (item 12): the residual of g(b) against the straight line t0 + b/B_eff,
and the prediction error of the pipeline on all K > 1 configurations.

    python scripts/fit_kernel_chain.py [results/raw/kernel_chain-....json]
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np


def load(path: str | None) -> dict:
    path = path or sorted(glob.glob("results/raw/kernel_chain-*.json"))[-1]
    return {"path": path, **json.loads(Path(path).read_text())}


def pipeline_time(N: int, K: int, e: float, c_commit: float, g: float, c_sync: float) -> float:
    cpu, gpu = 0.0, 0.0
    for _ in range(K):
        cpu += N * e + c_commit
        gpu = max(cpu, gpu) + N * g
    return gpu + c_sync


def fit(record: dict) -> dict:
    rows = [r for r in record["results"] if r["mode"] == "chain"]
    # CPU: median submit time per configuration.
    X = np.array([[r["N"] * r["K"], r["K"]] for r in rows], dtype=float)
    y = np.array([statistics.median(r["submit_s"]) for r in rows])
    (e, c_commit), *_ = np.linalg.lstsq(X, y, rcond=None)

    # GPU: for K = 1, T - submit = N g(b) + c_sync.
    g_of_b, sync_of_b = {}, {}
    for b in sorted({r["b"] for r in rows}):
        k1 = [r for r in rows if r["b"] == b and r["K"] == 1]
        A = np.array([[r["N"], 1.0] for r in k1])
        t = np.array([r["best_s"] - statistics.median(r["submit_s"]) for r in k1])
        (g, cs), *_ = np.linalg.lstsq(A, t, rcond=None)
        g_of_b[b], sync_of_b[b] = g, cs
    bs = np.array(sorted(g_of_b))
    gs = np.array([g_of_b[b] for b in bs])
    # Relative least squares: weight each b by 1/g so small kernels count.
    W = 1 / gs
    (t0, inv_bw), *_ = np.linalg.lstsq(np.c_[np.ones_like(bs), bs] * W[:, None], gs * W, rcond=None)
    c_sync = float(np.median(list(sync_of_b.values())))

    per_b = []
    for b in bs:
        line = t0 + b * inv_bw
        per_b.append({"b": int(b), "g_s": float(g_of_b[b]), "line_s": float(line),
                      "residual": float(g_of_b[b] / line - 1), "B_eff_marginal": float(b / (g_of_b[b] - t0))})

    pred = []
    for r in rows:
        p = pipeline_time(r["N"], r["K"], e, c_commit, g_of_b[r["b"]], c_sync)
        pred.append({"b": r["b"], "N": r["N"], "K": r["K"], "measured_s": r["best_s"], "predicted_s": p,
                     "error": p / r["best_s"] - 1})

    indep = {}
    for r in record["results"]:
        if r["mode"] == "indep" and r["N"] >= 16:
            chain = next(c for c in rows if (c["b"], c["N"], c["K"]) == (r["b"], r["N"], 1))
            indep.setdefault(r["b"], []).append(r["best_s"] / chain["best_s"])
    return {
        "source": record["path"],
        "e_s": float(e), "c_commit_s": float(c_commit), "t0_s": float(t0), "B_eff": float(1 / inv_bw),
        "c_sync_s": c_sync, "per_b": per_b, "pipeline": pred,
        "independent_over_chain": {int(b): float(np.mean(v)) for b, v in indep.items()},
    }


def main() -> None:
    f = fit(load(sys.argv[1] if len(sys.argv) > 1 else None))
    print(f"source: {f['source']}")
    print(f"CPU  e = {f['e_s'] * 1e6:.2f} us/kernel, c_commit = {f['c_commit_s'] * 1e6:.1f} us/buffer")
    print(f"GPU  t0 = {f['t0_s'] * 1e6:.2f} us/kernel, B_eff = {f['B_eff'] / 1e9:.1f} GB/s, "
          f"c_sync = {f['c_sync_s'] * 1e6:.1f} us")
    print(f"\n{'b KiB':>8}{'g(b) us':>10}{'line us':>10}{'resid':>8}{'B marginal GB/s':>17}")
    for r in f["per_b"]:
        print(f"{r['b'] // 1024:>8}{r['g_s'] * 1e6:>10.2f}{r['line_s'] * 1e6:>10.2f}{r['residual']:>8.1%}"
              f"{r['B_eff_marginal'] / 1e9:>17.1f}")
    errs = [abs(p["error"]) for p in f["pipeline"]]
    k1 = [abs(p["error"]) for p in f["pipeline"] if p["K"] == 1]
    kn = [abs(p["error"]) for p in f["pipeline"] if p["K"] > 1]
    print(f"\npipeline prediction |error|: K=1 median {statistics.median(k1):.1%} max {max(k1):.1%}; "
          f"K>1 median {statistics.median(kn):.1%} max {max(kn):.1%}; all {statistics.median(errs):.1%}")
    worst = sorted(f["pipeline"], key=lambda p: -abs(p["error"]))[:5]
    for p in worst:
        print(f"  worst: b={p['b'] // 1024} KiB N={p['N']} K={p['K']} measured {p['measured_s'] * 1e3:.3f} ms "
              f"predicted {p['predicted_s'] * 1e3:.3f} ms ({p['error']:+.0%})")
    print("\nindependent / dependent time, N >= 16, K = 1:",
          {f"{b // 1024} KiB": f"{v:.2f}" for b, v in f["independent_over_chain"].items()})
    out = Path("results/raw/kernel_chain_fit.json")
    out.write_text(json.dumps(f, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
