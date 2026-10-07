"""Clusterability of Qwen3-0.6B's cached keys (design notes 6.13).

SubGen (arXiv 2402.06082) is sublinear when the keys of each head are
(m, delta)-clusterable with m = o(n) and delta * r = o(log n), where r bounds
the query norm. This script measures, per (layer, KV head):

  r        max |q| * head_dim^-0.5 over the head's query heads and positions
           (the softmax sees exp(q . k * scale), so the scale folds into r)
  radius   cover radius after k centers of greedy farthest-first
           (Gonzalez) on the post-norm, post-RoPE keys of the first n tokens;
           also the radius covering 50 / 90 / 99% of the keys (the rest kept
           exact), in case a few outliers set the max

and reports m(n, delta r) = centers needed for radius * r <= delta r. The
greedy cover is within 2x of the optimal radius, so m here is an upper
estimate of the clusterability m at radius delta and a lower one at delta/2.

    python benchmarks/key_clusters.py                       # text + random, n up to 8192
    python benchmarks/key_clusters.py --max-n 32768 --kinds text
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent-test/v3"))

DELTA_R = (0.25, 0.5, 1.0, 2.0)
QUANTILES = (0.5, 0.9, 0.99)


def capture(model, tokens: list[int]):
    """Prefill `tokens`; return keys (L, H_kv, n, d) and per-(layer, KV head) max scaled query norm."""
    from mlx_lm.models import qwen3
    from mlx_lm.models.cache import make_prompt_cache

    q_max = {}
    orig = qwen3.Attention.__call__

    def hooked(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1))  # RoPE keeps the norm
        norms = mx.sqrt((q.astype(mx.float32) ** 2).sum(-1)) * self.scale  # (B, L, n_heads)
        per_kv = norms.reshape(B, L, self.n_kv_heads, -1).max(axis=(0, 1, 3))
        idx = id(self)
        q_max[idx] = per_kv if idx not in q_max else mx.maximum(q_max[idx], per_kv)
        return orig(self, x, mask, cache)

    qwen3.Attention.__call__ = hooked
    try:
        cache = make_prompt_cache(model)
        chunk = 2048
        for i in range(0, len(tokens), chunk):
            model(mx.array(tokens[i:i + chunk])[None], cache=cache)
            mx.eval([c.state for c in cache])
    finally:
        qwen3.Attention.__call__ = orig
    n = len(tokens)
    keys = mx.stack([c.keys[0, :, :n, :] for c in cache])  # (L, H, n, d)
    r = mx.stack([q_max[id(layer.self_attn)] for layer in model.layers])  # (L, H)
    mx.eval(keys, r)
    return keys, np.array(r)


def farthest_first(x: mx.array, k_max: int, record: list[int]) -> tuple[np.ndarray, dict]:
    """x: (G, n, d). Returns radius[G, k] = cover radius with k centers, and
    for each k in `record` the QUANTILES of the distance to the nearest center
    (radius covering that fraction of the keys; the rest would be kept exact)."""
    G, n, _ = x.shape
    x32 = x.astype(mx.float32)
    sq = (x32 * x32).sum(-1)  # (G, n)
    idx = mx.zeros((G,), dtype=mx.int32)
    gather = lambda i: mx.take_along_axis(x32, i[:, None, None], axis=1)  # (G, 1, d)  # noqa: E731

    def dist(c):
        c_sq = (c * c).sum(-1)  # (G, 1)
        return mx.maximum(sq - 2 * (x32 @ c.transpose(0, 2, 1))[..., 0] + c_sq, 0)

    best = dist(gather(idx))
    radii, quant = [], {}
    for k in range(1, k_max + 1):
        if k in record:
            srt = mx.sqrt(mx.sort(best, axis=1))
            cols = [min(n - 1, int(q * n)) for q in QUANTILES]
            quant[k] = np.array(srt[:, cols])  # (G, len(QUANTILES))
        radii.append(mx.sqrt(mx.max(best, axis=1)))
        idx = mx.argmax(best, axis=1).astype(mx.int32)
        best = mx.minimum(best, dist(gather(idx)))
        mx.eval(best, radii[-1])
    return np.stack([np.array(r) for r in radii], axis=1), quant


def centers_needed(radius: np.ndarray, r: np.ndarray, delta_r: float) -> np.ndarray:
    """radius (G, K), r (G,): smallest k with radius_k * r <= delta_r; K + 1 if none (censored)."""
    ok = radius * r[:, None] <= delta_r
    k = np.where(ok.any(1), ok.argmax(1) + 1, radius.shape[1] + 1)
    return k


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--kinds", nargs="+", default=["text", "random"])
    p.add_argument("--max-n", type=int, default=8192)
    p.add_argument("--k-frac", type=float, default=0.5, help="farthest-first runs to k = k_frac * n")
    p.add_argument("--out", default=str(ROOT / "results/raw/key_clusters.json"))
    args = p.parse_args()

    from mlx_lm import load
    from prompts import make_prompt

    model, _ = load(args.model)
    ns = [n for n in (1024, 2048, 4096, 8192, 16384, 32768) if n <= args.max_n]
    out = {"model": args.model, "delta_r": DELTA_R, "k_frac": args.k_frac, "runs": []}
    for kind in args.kinds:
        tokens = make_prompt(kind, args.max_n, 0)
        keys, r = capture(model, tokens)
        L, H, _, d = keys.shape
        r_flat = r.reshape(-1)
        key_norm = np.array(mx.sqrt((keys.astype(mx.float32) ** 2).sum(-1)).mean(axis=2)).reshape(-1)
        for n in ns:
            t0 = time.time()
            k_max = max(1, int(args.k_frac * n))
            ks = sorted({2**i for i in range(int(math.log2(k_max)) + 1)} | {k_max})
            radius, quant = farthest_first(keys[:, :, :n, :].reshape(L * H, n, d), k_max, ks)
            m = {str(dr): centers_needed(radius, r_flat, dr).tolist() for dr in DELTA_R}
            out["runs"].append({
                "kind": kind, "n": n, "k_max": k_max, "layers": L, "kv_heads": H,
                "r": r_flat.tolist(), "mean_key_norm": key_norm.tolist(),
                "radius_at_k": {str(k): radius[:, k - 1].tolist() for k in ks},
                "quantile_radius_at_k": {str(k): {str(q): quant[k][:, i].tolist() for i, q in enumerate(QUANTILES)}
                                         for k in ks},
                "m": m, "seconds": time.time() - t0,
            })
            med = {dr: int(np.median(v)) for dr, v in m.items()}
            print(f"{kind:6s} n={n:6d}  median m by delta*r {med}  ({time.time() - t0:.1f} s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
