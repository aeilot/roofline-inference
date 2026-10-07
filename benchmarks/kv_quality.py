"""Output quality of KV-cache policies on Qwen3-0.6B (design notes 6.14).

For each prompt, an exact BF16 prefill of P tokens builds the KV cache. The
next T tokens of the real text are then fed in one teacher-forced pass, once
per policy; each policy changes only what the continuation queries see of the
prefix (a mask, or a quantized copy of the prefix K/V). The continuation's own
K/V stay exact and visible (causal). Metrics per position, against the exact
BF16 pass with the same code path:

  kl     KL(p_exact || p_policy) of the next-token distribution
  top1   argmax agreement
  dnll   NLL of the true next token, policy minus exact

Noise floor: KL(p_fp32 || p_bf16) between an FP32 and the BF16 exact pass.

Policies (each maps to an estimator policy in kvpolicy.py):

  bits:b          group-wise (64) affine quantization of prefix K and V  -> full(bits = b + 0.5)
  kivi:b          as bits, but keys grouped per channel over 64 tokens
                  (KIVI), values per token                               -> full(bits = b + 0.5)
  window:W        first 4 + last W cached tokens, sliding per query       -> window(W, sink=4)
  snapkv:k        recent k/4 + top 3k/4 by attention from the last 64
                  prefill queries (SnapKV-style), fixed after prefill    -> budget(k)
  kcenter:k       recent k/4 + 3k/4 farthest-first key centers, kept
                  exact (the empirical SubGen variant), fixed            -> budget(k)
  quest:k         per query head and query: pages of 16 tokens ranked
                  by the min/max key upper bound, top k/16 pages         -> retrieval(k, 16)

    python benchmarks/kv_quality.py                       # 4k / 16k / 32k, 4 prompts each
    python benchmarks/kv_quality.py --lengths 4096 --prompts 1 --tokens 64   # quick
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

SINK = 4
OBS = 64  # SnapKV observation window
PAGE = 16
GROUP = 64  # quantization group size

# Shared with the patched attention.
STATE: dict = {"mode": "native"}


def policies(p: int) -> list[str]:
    out = ["full"] + [f"bits:{b}" for b in (8, 6, 4, 3, 2)] + [f"kivi:{b}" for b in (4, 3, 2)]
    for k in (256, 1024, 4096):
        if k < p:
            out += [f"window:{k}", f"snapkv:{k}", f"kcenter:{k}", f"quest:{k}"]
    return out


# --- attention hook ---------------------------------------------------------

def install(*models):
    """Patch Qwen3 attention; layers of every model are indexed by id()."""
    from mlx_lm.models import qwen3

    index = {id(layer.self_attn): i for m in models for i, layer in enumerate(m.layers)}
    orig = qwen3.Attention.__call__

    def hooked(self, x, mask=None, cache=None):
        i = index[id(self)]
        if STATE["mode"] == "prefill":
            if STATE.get("obs_from") is not None and cache.offset + x.shape[1] > STATE["obs_from"]:
                B, L, _ = x.shape
                q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
                q = self.rope(q, offset=cache.offset)
                STATE["obs_q"][i] = q[0, :, max(0, STATE["obs_from"] - cache.offset):, :]
            return orig(self, x, mask, cache)
        if STATE["mode"] == "cont":
            return continuation_attention(self, i, x)
        return orig(self, x, mask, cache)

    qwen3.Attention.__call__ = hooked
    return lambda: setattr(qwen3.Attention, "__call__", orig)


def continuation_attention(self, i: int, x: mx.array) -> mx.array:
    """T continuation queries over the (policy-transformed) prefix plus themselves."""
    B, T, _ = x.shape
    P = STATE["P"]
    hq, hk = self.n_heads, self.n_kv_heads
    g = hq // hk
    q = self.q_norm(self.q_proj(x).reshape(B, T, hq, -1)).transpose(0, 2, 1, 3)
    k = self.k_norm(self.k_proj(x).reshape(B, T, hk, -1)).transpose(0, 2, 1, 3)
    v = self.v_proj(x).reshape(B, T, hk, -1).transpose(0, 2, 1, 3)
    q, k = self.rope(q, offset=P), self.rope(k, offset=P)
    pk, pv = STATE["prefix_k"][i], STATE["prefix_v"][i]  # (hk, P, d)
    keys = mx.concatenate([pk, k[0].astype(pk.dtype)], axis=1)  # (hk, P+T, d)
    vals = mx.concatenate([pv, v[0].astype(pv.dtype)], axis=1)
    dt = mx.float32
    qg = q[0].reshape(hk, g, T, -1).astype(dt)
    scores = (qg @ keys.astype(dt)[:, None].transpose(0, 1, 3, 2)) * self.scale  # (hk, g, T, P+T)
    allow = STATE["mask_fn"](i, q[0], pk)  # broadcastable to (hk, g, T, P) or None
    causal = mx.arange(T)[None, :] <= mx.arange(T)[:, None]  # (T, T)
    if allow is None:
        allow_prefix = mx.ones((1, 1, T, P), dtype=mx.bool_)
    else:
        allow_prefix = mx.broadcast_to(allow, (hk, g, T, P))
    full_mask = mx.concatenate([mx.broadcast_to(allow_prefix, (hk, g, T, P)),
                                mx.broadcast_to(causal, (hk, g, T, T))], axis=-1)
    scores = mx.where(full_mask, scores, -mx.inf)
    w = mx.softmax(scores, axis=-1)
    out = (w @ vals.astype(dt)[:, None]).reshape(hq, T, -1)  # (hq, T, d)
    out = out.astype(x.dtype)[None].transpose(0, 2, 1, 3).reshape(B, T, -1)
    out = self.o_proj(out)
    mx.eval(out)
    return out


# --- policies ---------------------------------------------------------------

def quantized(a: mx.array, bits: int) -> mx.array:
    hk, p, d = a.shape
    w, s, b = mx.quantize(a.reshape(hk * p, d), group_size=GROUP, bits=bits)
    return mx.dequantize(w, s, b, group_size=GROUP, bits=bits).reshape(hk, p, d).astype(a.dtype)


def quantized_per_channel(a: mx.array, bits: int) -> mx.array:
    """Groups of GROUP tokens along each channel (KIVI's key quantization)."""
    hk, p, d = a.shape
    pad = -p % GROUP
    t = mx.pad(a, [(0, 0), (0, pad), (0, 0)]).transpose(0, 2, 1).reshape(hk * d, p + pad)
    w, s, b = mx.quantize(t, group_size=GROUP, bits=bits)
    back = mx.dequantize(w, s, b, group_size=GROUP, bits=bits).reshape(hk, d, p + pad).transpose(0, 2, 1)
    return back[:, :p, :].astype(a.dtype)


def window_mask(P: int, T: int, w: int) -> mx.array:
    """(1, 1, T, P): query t has P + t cached tokens; keep the first SINK and the last w of them."""
    j = mx.arange(P)[None, :]
    t = mx.arange(T)[:, None]
    return ((j < SINK) | (j >= P + t - w))[None, None]


def static_mask(P: int, keep: mx.array) -> mx.array:
    """keep: (hk, n) token indices -> (hk, 1, 1, P) bool."""
    m = mx.zeros((keep.shape[0], P), dtype=mx.bool_)
    m = mx.put_along_axis(m, keep, mx.array(True), axis=1)
    return m[:, None, None, :]


def snapkv_keep(i: int, k: int, P: int, pk: mx.array) -> mx.array:
    q = STATE["obs_q"][i].astype(mx.float32)  # (hq, OBS, d)
    hq, n_obs, d = q.shape
    hk = pk.shape[0]
    qg = q.reshape(hk, hq // hk, n_obs, d)
    s = (qg @ pk.astype(mx.float32)[:, None].transpose(0, 1, 3, 2)) * d**-0.5  # (hk, g, OBS, P)
    pos = P - n_obs + mx.arange(n_obs)[:, None]
    s = mx.where(mx.arange(P)[None, :] <= pos, s, -mx.inf)
    score = mx.softmax(s, axis=-1).sum(axis=(1, 2))  # (hk, P)
    recent = k // 4
    older = score[:, : P - recent]
    top = mx.argpartition(-older, kth=k - recent - 1, axis=1)[:, : k - recent]
    rec = mx.broadcast_to(mx.arange(P - recent, P)[None], (hk, recent))
    return mx.concatenate([top.astype(mx.int32), rec.astype(mx.int32)], axis=1)


def farthest_first_indices(x: mx.array, k: int) -> mx.array:
    """x: (G, n, d) -> (G, k) indices of greedy farthest-first centers, starting at 0."""
    G, n, _ = x.shape
    sq = (x.astype(mx.float32) ** 2).sum(-1)
    idx = mx.zeros((G,), dtype=mx.int32)
    chosen = [idx]

    def dist(i):
        c = mx.take_along_axis(x, i[:, None, None], axis=1)  # (G, 1, d)
        dot = (x @ c.transpose(0, 2, 1))[..., 0].astype(mx.float32)
        return sq - 2 * dot + (c.astype(mx.float32) ** 2).sum(-1)

    best = dist(idx)
    for _ in range(k - 1):
        idx = mx.argmax(best, axis=1).astype(mx.int32)
        chosen.append(idx)
        best = mx.minimum(best, dist(idx))
        mx.eval(best, idx)
    return mx.stack(chosen, axis=1)


def quest_mask(q: mx.array, pk: mx.array, k: int) -> mx.array:
    """q (hq, T, d), pk (hk, P, d) -> (hk, g, T, P) bool: top k/PAGE pages per query head."""
    hq, T, d = q.shape
    hk, P, _ = pk.shape
    n_pages = math.ceil(P / PAGE)
    pad = n_pages * PAGE - P
    kp = pk.astype(mx.float32)
    lo = mx.pad(kp, [(0, 0), (0, pad), (0, 0)], constant_values=mx.inf).reshape(hk, n_pages, PAGE, d).min(2)
    hi = mx.pad(kp, [(0, 0), (0, pad), (0, 0)], constant_values=-mx.inf).reshape(hk, n_pages, PAGE, d).max(2)
    qg = q.astype(mx.float32).reshape(hk, hq // hk, T, 1, d)
    ub = mx.maximum(qg * lo[:, None, None], qg * hi[:, None, None]).sum(-1)  # (hk, g, T, n_pages)
    n_sel = min(n_pages, k // PAGE)
    kth = n_pages - n_sel
    thr = mx.partition(ub, kth, axis=-1)[..., kth:kth + 1]
    pages = ub >= thr
    return mx.repeat(pages, PAGE, axis=-1)[..., :P]


def setup_policy(name: str, P: int, T: int, keys: list, vals: list, centers: list | None):
    """Set STATE prefix K/V and mask_fn for `name`."""
    kind, _, arg = name.partition(":")
    n = int(arg) if arg else 0
    STATE["prefix_k"], STATE["prefix_v"] = keys, vals
    STATE["mask_fn"] = lambda i, q, pk: None
    if kind == "bits":
        STATE["prefix_k"] = [quantized(a, n) for a in keys]
        STATE["prefix_v"] = [quantized(a, n) for a in vals]
    elif kind == "kivi":
        STATE["prefix_k"] = [quantized_per_channel(a, n) for a in keys]
        STATE["prefix_v"] = [quantized(a, n) for a in vals]
    elif kind == "window":
        m = window_mask(P, T, n)
        STATE["mask_fn"] = lambda i, q, pk: m
    elif kind == "snapkv":
        masks = [static_mask(P, snapkv_keep(i, n, P, keys[i])) for i in range(len(keys))]
        STATE["mask_fn"] = lambda i, q, pk: masks[i]
    elif kind == "kcenter":
        recent = n // 4
        hk = keys[0].shape[0]
        rec = mx.broadcast_to(mx.arange(P - recent, P)[None], (hk, recent)).astype(mx.int32)
        masks = [static_mask(P, mx.concatenate([centers[i][:, : n - recent], rec], axis=1))
                 for i in range(len(keys))]
        STATE["mask_fn"] = lambda i, q, pk: masks[i]
    elif kind == "quest":
        STATE["mask_fn"] = lambda i, q, pk: quest_mask(q, pk, n)
    elif kind != "full":
        raise ValueError(name)


# --- driver -----------------------------------------------------------------

def prefill(model, tokens: list[int], obs: bool):
    from mlx_lm.models.cache import make_prompt_cache

    P = len(tokens)
    cache = make_prompt_cache(model)
    STATE.update(mode="prefill", obs_from=P - OBS if obs else None, obs_q={})
    for s in range(0, P, 2048):
        model(mx.array(tokens[s:s + 2048])[None], cache=cache)
        mx.eval([c.state for c in cache])
    STATE["mode"] = "native"
    keys = [c.keys[0, :, :P, :] for c in cache]
    vals = [c.values[0, :, :P, :] for c in cache]
    mx.eval(keys, vals)
    return keys, vals, cache


def continuation_logprobs(model, cont: list[int], P: int) -> mx.array:
    STATE.update(mode="cont", P=P)
    logits = model(mx.array(cont)[None])[0].astype(mx.float32)
    STATE["mode"] = "native"
    out = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(out)
    return out


def compare(ref: mx.array, lp: mx.array, targets: mx.array) -> dict:
    kl = (mx.exp(ref) * (ref - lp)).sum(-1)
    top1 = (ref.argmax(-1) == lp.argmax(-1)).astype(mx.float32)
    nll_ref = -mx.take_along_axis(ref, targets[:, None], axis=-1)[:, 0]
    nll = -mx.take_along_axis(lp, targets[:, None], axis=-1)[:, 0]
    mx.eval(kl, top1, nll, nll_ref)
    kl = np.maximum(np.array(kl), 0)
    return {"kl": [float(f"{v:.4g}") for v in kl], "top1": float(np.array(top1).mean()),
            "dnll": float(np.array(nll - nll_ref).mean()), "nll_exact": float(np.array(nll_ref).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 32768])
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=256, help="continuation tokens T")
    ap.add_argument("--fp32-prompts", type=int, default=2, help="prompts per length with the FP32 floor")
    ap.add_argument("--out", default=str(ROOT / "results/raw/kv_quality.json"))
    a = ap.parse_args()

    from mlx_lm import load
    from prompts import corpus_sha256, corpus_tokens

    model, _ = load(a.model)
    model32 = None
    if a.fp32_prompts > 0:
        model32, _ = load(a.model)
        model32.set_dtype(mx.float32)
    uninstall = install(*[m for m in (model, model32) if m is not None])
    corpus = list(corpus_tokens())
    T = a.tokens
    out = {"model": a.model, "corpus_sha256": corpus_sha256(), "corpus_tokens": len(corpus),
           "T": T, "sink": SINK, "page": PAGE, "group": GROUP, "runs": [], "floor": [], "checks": []}
    try:
        for P in a.lengths:
            span = len(corpus) - (P + T + 1)
            if span < 0:
                raise ValueError(f"corpus too short for P={P}")
            starts = [round(span * v / max(1, a.prompts - 1)) for v in range(a.prompts)]
            names = policies(P)
            for v, s in enumerate(starts):
                t0 = time.time()
                toks = corpus[s:s + P + T + 1]
                prompt, cont, targets = toks[:P], toks[P:P + T], mx.array(toks[P + 1:P + T + 1])
                keys, vals, cache = prefill(model, prompt, obs=True)
                t_pre = time.time() - t0
                kmax = max([int(n.split(":")[1]) for n in names if n.startswith("kcenter")] or [0])
                centers = None
                if kmax:
                    recent = kmax // 4
                    stacked = mx.concatenate([kk[:, : P - recent, :] for kk in keys], axis=0)
                    hk = keys[0].shape[0]
                    idx = farthest_first_indices(stacked, kmax - recent)
                    centers = [idx[l * hk:(l + 1) * hk] for l in range(len(keys))]
                    del stacked
                t_cent = time.time() - t0 - t_pre
                ref = None
                for name in names:
                    setup_policy(name, P, T, keys, vals, centers)
                    lp = continuation_logprobs(model, cont, P)
                    if name == "full":
                        ref = lp
                    r = compare(ref, lp, targets)
                    out["runs"].append({"P": P, "prompt": v, "start": s, "policy": name, **r})
                if v == 0:
                    # Masks that keep everything must reproduce the exact pass bit for bit;
                    # mlx-lm's own cached attention must agree to within BF16 noise.
                    for noop in (f"window:{P + T}", f"quest:{P + PAGE}"):
                        setup_policy(noop, P, T, keys, vals, centers)
                        assert max(compare(ref, continuation_logprobs(model, cont, P), targets)["kl"]) == 0, noop
                    native = model(mx.array(cont)[None], cache=cache)[0].astype(mx.float32)
                    native = native - mx.logsumexp(native, axis=-1, keepdims=True)
                    r = compare(ref, native, targets)
                    out["checks"].append({"P": P, "check": "native_mlx_lm", **r})
                    print(f"    native mlx-lm vs exact path: kl p99 {np.percentile(r['kl'], 99):.2e} "
                          f"top1 {r['top1']:.3f}", flush=True)
                del cache
                print(f"P={P:6d} prompt {v} start {s:6d}: prefill {t_pre:.0f} s, centers {t_cent:.0f} s, "
                      f"total {time.time() - t0:.0f} s", flush=True)
                for row in out["runs"][-len(names):]:
                    kl = np.array(row["kl"])
                    print(f"    {row['policy']:14s} kl p50 {np.median(kl):.2e} p99 {np.percentile(kl, 99):.2e} "
                          f"top1 {row['top1']:.3f} dnll {row['dnll']:+.4f}", flush=True)
                if v < a.fp32_prompts:
                    del keys, vals, centers
                    k32, v32, c32 = prefill(model32, prompt, obs=False)
                    del c32
                    setup_policy("full", P, T, k32, v32, None)
                    lp32 = continuation_logprobs(model32, cont, P)
                    del k32, v32
                    r = compare(lp32, ref, targets)
                    out["floor"].append({"P": P, "prompt": v, "start": s, **r})
                    kl = np.array(r["kl"])
                    print(f"    {'fp32 floor':14s} kl p50 {np.median(kl):.2e} p99 {np.percentile(kl, 99):.2e} "
                          f"top1 {r['top1']:.3f}", flush=True)
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(out))
    finally:
        uninstall()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
