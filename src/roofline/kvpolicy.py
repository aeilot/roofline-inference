"""KV-cache policies for the decode step (level B/C knobs) and the level-A
storage-plan check (design notes 6.13).

A policy says, for a step with `context` cached tokens, how many cached key
and value vectors (per KV head) the step reads, how many it keeps resident,
and at what precision. `full()` is the level-A model: every cached K and V is
read once. The other policies are relaxations; their shape follows the space
and time lower bounds for streaming attention:

- Haris & Onak 2025 (arXiv 2502.15955): any algorithm that (1 +- eta)
  approximates every decode step's attention output needs Omega(n d) bits
  when d = Omega(log n), even if most scores are near zero but their
  positions are unknown. Without structure, only the precision (bits per
  element) changes the slope; the growth in n stays linear. A fixed access
  pattern (sliding window W) escapes it: Theta(d W).
  Non-adaptive algorithms (access pattern fixed before seeing the data) need
  Omega(n d) time for a step; adaptive ones (retrieval) can read less than
  they store.
- SubGen (Zandieh et al. 2024, arXiv 2402.06082): if the keys are
  (m, delta)-clusterable and |q| <= r, O(d (m t + s)) memory and time with
  t = eps^-2 e^(2 delta r) log n and s = eps^-2 d log n.

The Omega/O constants are unknown, so these policies fix the shape of the KV
term, not a numeric floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from roofline.model import ModelSpec


@dataclass(frozen=True)
class KvPolicy:
    """Per KV head, per step: `k_read(n)` cached keys and `v_read(n)` cached
    values read, `stored(n)` (key, value)-vector equivalents resident, given
    n cached tokens. `bits` overrides the KV dtype (None = model kv_dtype).
    `adaptive`: the read set depends on the query (stored may exceed read)."""

    name: str
    k_read: Callable[[int], float]
    v_read: Callable[[int], float]
    stored: Callable[[int], float]
    bits: float | None = None
    adaptive: bool = False

    def elem_bytes(self, m: ModelSpec) -> float:
        return m.kv_bytes_per_elem if self.bits is None else self.bits / 8

    def bytes_per_vector(self, m: ModelSpec) -> float:
        """One K (or V) vector of one token, summed over layers and KV heads."""
        return m.num_layers * m.kv_dim * self.elem_bytes(m)


def full(bits: float | None = None) -> KvPolicy:
    """Exact attention over every cached token (level A when bits is None)."""
    name = "full" if bits is None else f"full@{bits:g}b"
    return KvPolicy(name, lambda n: n, lambda n: n, lambda n: n, bits)


def window(w: int, sink: int = 0, bits: float | None = None) -> KvPolicy:
    """Fixed pattern: the first `sink` and the last `w` cached tokens."""
    if w < 0 or sink < 0:
        raise ValueError("window and sink must be >= 0")
    keep = lambda n: min(n, w + sink)  # noqa: E731
    return KvPolicy(f"window({w},sink={sink})", keep, keep, keep, bits)


def budget(k: int, v: int | None = None, bits: float | None = None, name: str = "budget") -> KvPolicy:
    """A fixed cache budget, e.g. k-center keys plus recent tokens (the
    empirical SubGen variant). No accuracy guarantee is implied."""
    v = k if v is None else v
    return KvPolicy(f"{name}(k={k},v={v})", lambda n: min(n, k), lambda n: min(n, v),
                    lambda n: min(n, max(k, v)), bits)


def subgen_vectors(n: int, m: float, delta_r: float, eps: float, head_dim: int) -> tuple[float, float]:
    """SubGen's (key, value) vectors kept per head, with the theorem's
    constants set to 1: keys m*t + s, values s. Capped at n (exact is cheaper)."""
    if n < 2:
        return float(n), float(n)
    log_n = math.log(n)
    t = eps**-2 * math.exp(2 * delta_r) * log_n
    s = eps**-2 * head_dim * log_n
    return min(n, m * t + s), min(n, s)


def subgen(m_of_n: Callable[[int], float], delta_r: float, eps: float, head_dim: int,
           bits: float | None = None) -> KvPolicy:
    """SubGen with its provable guarantee: m_of_n(n) clusters of radius delta,
    query norm <= r. Reads everything it stores (non-adaptive)."""
    kv = lambda n: subgen_vectors(n, m_of_n(n), delta_r, eps, head_dim)  # noqa: E731
    return KvPolicy(f"subgen(dr={delta_r:g},eps={eps:g})", lambda n: kv(n)[0], lambda n: kv(n)[1],
                    lambda n: max(kv(n)), bits)


def retrieval(k: int, page: int = 16, summary_vectors: int = 2, bits: float | None = None) -> KvPolicy:
    """Adaptive selection (Quest-style): store everything, read a summary of
    `summary_vectors` key-sized vectors per page of `page` tokens, then the
    top-k tokens' K and V."""
    k_read = lambda n: min(n, math.ceil(n / page) * summary_vectors + k)  # noqa: E731
    return KvPolicy(f"retrieval(k={k},page={page})", k_read, lambda n: min(n, k),
                    lambda n: n + math.ceil(n / page) * summary_vectors / 2, bits, adaptive=True)


def parse_policy(spec: str | None, bits: float | None = None) -> KvPolicy:
    """CLI form: full | window:W[:sink] | budget:K[:V] | retrieval:K[:page]."""
    kind, *args = (spec or "full").split(":")
    nums = [int(a) for a in args]
    makers = {"full": full, "window": window, "budget": budget, "retrieval": retrieval}
    if kind not in makers:
        raise ValueError(f"unknown KV policy {kind!r}")
    return makers[kind](*nums, bits=bits)


# --- Level A: what to keep between steps -----------------------------------

def storage_plans(m: ModelSpec, context: int, batch: int = 1) -> dict[str, dict]:
    """Extra DRAM bytes and FLOPs per decode step for the attention state of
    `context` cached tokens, under three mathematically equivalent plans:

    kv         cache post-RoPE K and V (what every engine does)
    hidden     cache each layer's input hidden state; recompute K and V
               (input norm, K/V projections, k-norm, RoPE) every step
    recompute  cache token ids only; rerun the forward pass over the context

    Level A allows recomputation, so the level-A bound is the minimum over
    plans; `kv` wins when the FLOPs added per byte saved exceed the machine
    balance (design notes 6.13).
    """
    n, L, e = batch * context, m.num_layers, m.kv_bytes_per_elem
    kv_bytes = n * m.kv_bytes_per_token
    # Elementwise work (norms, RoPE) is not counted, as in workload.py.
    kv_proj_flops = 2 * m.hidden_size * 2 * m.kv_dim
    return {
        "kv": {"bytes": kv_bytes, "flops": 0.0},
        "hidden": {"bytes": n * L * m.hidden_size * e, "flops": n * L * kv_proj_flops},
        # Everything but the last layer's Q/O/FFN; counted as the full linear work, a slight over-count.
        "recompute": {"bytes": n * 4, "flops": n * 2 * L * m.layer_linear_params},
    }


def best_storage_plan(m: ModelSpec, context: int, peak_flops: float, bandwidth: float,
                      base_flops: float = 0.0, base_bytes: float = 0.0, batch: int = 1) -> dict:
    """Step time max(FLOPs / peak, bytes / bandwidth) for each plan, given the
    rest of the step (`base_*`: weights, linear layers, attention)."""
    out = {}
    for name, p in storage_plans(m, context, batch).items():
        t_c = (base_flops + p["flops"]) / peak_flops
        t_m = (base_bytes + p["bytes"]) / bandwidth
        out[name] = {**p, "time_s": max(t_c, t_m), "compute_s": t_c, "dram_s": t_m}
    best = min(out, key=lambda k: out[k]["time_s"])
    return {"plans": out, "best": best,
            "balance_flop_per_byte": peak_flops / bandwidth,
            "hidden_flop_per_byte_saved": _intensity(out["kv"], out["hidden"])}


def _intensity(base: dict, alt: dict) -> float:
    saved = base["bytes"] - alt["bytes"]
    return math.inf if saved <= 0 else (alt["flops"] - base["flops"]) / saved
