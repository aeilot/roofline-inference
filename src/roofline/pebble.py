"""DRAM traffic lower bound with finite on-chip memory (red-blue pebble game).

The physical bound assumes unlimited on-chip storage: every weight is read once,
activations never leave the chip. This module bounds traffic when on-chip
memory holds only M bytes, under one stated assumption:

    Operator boundary at every GEMM. Each GEMM is its own operator: its
    activation input A comes from DRAM or from what is still on chip
    (at most M bytes), and its output C goes to DRAM unless it stays on chip.
    Elementwise work (norms, RoPE, SwiGLU, residuals) is free and fused into
    the GEMMs. Attention is one more operator per layer: its traffic is the
    larger of the K/V cache read and the attention bound below. Its Q and O
    traffic, and the prefill read-back of new K/V, are not counted.

This is a layer-2 bound (design notes 6.2): it depends on the fusion choice.
Without the assumption (A produced on chip by a fused producer) the GEMM
theorem below no longer applies; bounding a fully fused graph is open work
(SOAP, Kwasniewski et al. SPAA'21).

Per GEMM C[m x n] = A[m x k] @ W[k x n], in elements, fast memory M elements:

    reads  >= max( W,                               weights are re-read every step
                   W + A - M,                       each input read once, M may be resident
                   2mnk / sqrt(M) - mn - 2M )       Smith et al. 2019, Corollary 3.8
    writes >= max( persistent_out, C - M )          new K/V must persist

The third term is the red-blue pebble bound (Hong & Kung 1981; tight leading
constant from Smith, Lowery, Langou, van de Geijn, "A Tight I/O Lower Bound for
Matrix Multiplication", TOMS 2019). It only exceeds the compulsory terms when
k and n are well above sqrt(M), and it caps arithmetic intensity at about
sqrt(M)/2 FLOP/byte (bf16), see `pebble_intensity_cap`.

Per attention operator computing E entries of QK^T (all heads, causal entries
only) with head dimension d, in elements, reads + writes:

    io >= E d^2 / (M + 4 d^2) - M

This is Saha & Ye 2024 ("The I/O Complexity of Attention", Lemma 3.2) with
the constants written out; derivation in design notes 6.6. It holds for the
standard matmul algorithm with any summation order, for rectangular, causal
and GQA shapes, but needs Q and K to start in slow memory (the operator
boundary). It caps intensity at about 2(M + 4d^2)/d FLOP/byte (bf16), see
`attention_intensity_cap`.

Leaving out traffic only weakens a lower bound, so the omissions above keep
it valid. M is taken from the upper estimate of on-chip capacity for the same
reason: a larger M gives a smaller bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from roofline.workload import Attention, Gemm, Workload


@dataclass(frozen=True)
class GemmTraffic:
    gemm: Gemm
    reads: dict[str, float]  # bytes, per candidate lower bound
    writes: float  # bytes

    @property
    def read_bytes(self) -> float:
        return max(self.reads.values())

    @property
    def binding_read_term(self) -> str:
        return max(self.reads, key=self.reads.get)

    @property
    def total(self) -> float:
        return self.read_bytes + self.writes


def mmm_read_lower_bound(m: int, n: int, k: int, fast_elems: float) -> float:
    """Elements any C = A @ B must read with `fast_elems` of fast memory
    (Smith et al. 2019, Corollary 3.8). Can be negative for small problems."""
    if math.isinf(fast_elems):
        return -math.inf
    return 2 * m * n * k / math.sqrt(fast_elems) - m * n - 2 * fast_elems


def gemm_traffic(g: Gemm, on_chip_bytes: float) -> GemmTraffic:
    """Lower bound on one GEMM's DRAM traffic, times g.count."""
    e = g.elem_bytes
    M = on_chip_bytes / e  # elements
    W, A, C = g.k * g.n, g.m * g.k, g.m * g.n
    reads = {
        "weights": W * e,
        "compulsory": max(W, W + A - M) * e,
        "pebble": max(0.0, mmm_read_lower_bound(g.m, g.n, g.k, M)) * e,
    }
    writes = max(g.persistent_out_bytes, (C - M) * e)
    return GemmTraffic(
        gemm=g,
        reads={r: v * g.count for r, v in reads.items()},
        writes=writes * g.count,
    )


def attention_io_lower_bound(scores: float, head_dim: int, fast_elems: float) -> float:
    """Elements any attention computing `scores` entries of QK^T must move,
    with Q and K starting in slow memory (Saha & Ye 2024, Lemma 3.2, explicit
    constants). Can be negative for small problems."""
    if math.isinf(fast_elems):
        return -math.inf
    d2 = head_dim * head_dim
    return scores * d2 / (fast_elems + 4 * d2) - fast_elems


def attention_traffic(a: Attention, kv_read_bytes: float, on_chip_bytes: float) -> dict[str, float]:
    """Candidate lower bounds (bytes, all `a.count` layers) on attention
    operator traffic; the K/V cache read is compulsory on its own."""
    regular = a.count if a.final_scores is None else a.count - 1
    io = max(0.0, attention_io_lower_bound(a.scores, a.head_dim, on_chip_bytes / a.elem_bytes)) * regular
    if a.final_scores is not None:
        io += max(0.0, attention_io_lower_bound(a.final_scores, a.head_dim, on_chip_bytes / a.elem_bytes))
    return {"kv_read": kv_read_bytes, "pebble": io * a.elem_bytes}


def operator_traffic(w: Workload, on_chip_bytes: float) -> dict:
    """DRAM bytes for `w` under the operator-boundary assumption.

    Starts from the physical traffic and replaces the GEMM weight reads and
    the new-K/V writes by the per-GEMM bounds, which include them, and the
    K/V cache read by the attention bound, which takes the larger of the two.
    """
    if not w.gemms:
        raise ValueError("workload has no GEMM shapes")
    per_gemm = [gemm_traffic(g, on_chip_bytes) for g in w.gemms]
    gemm_weights = sum(t.reads["weights"] for t in per_gemm)
    other = dict(w.bytes)
    other["norm_weights"] = other.pop("weights") - gemm_weights
    other.pop("kv_write")  # covered by qkv's persistent_out
    attention = None
    if w.attention is not None:
        terms = attention_traffic(w.attention, other.pop("kv_read"), on_chip_bytes)
        other["attention"] = max(terms.values())
        attention = {
            "scores": w.attention.scores, "head_dim": w.attention.head_dim, "count": w.attention.count,
            "bytes": other["attention"],
            "binding_term": max(terms, key=terms.get),
            "terms": terms,
        }
    bytes_ = {**other, **{f"gemm.{t.gemm.name}": t.total for t in per_gemm}}
    return {
        "bytes": bytes_,
        "total": sum(bytes_.values()),
        "attention": attention,
        "gemms": {
            t.gemm.name: {
                "m": t.gemm.m, "n": t.gemm.n, "k": t.gemm.k, "count": t.gemm.count,
                "read_bytes": t.read_bytes, "write_bytes": t.writes,
                "binding_read_term": t.binding_read_term,
                "read_terms": t.reads,
                "pebble_critical_on_chip_bytes": pebble_critical_on_chip(t.gemm),
            }
            for t in per_gemm
        },
    }


def pebble_intensity_cap(on_chip_bytes: float, elem_bytes: int) -> float:
    """Upper limit on a large GEMM's FLOP per DRAM byte implied by the pebble
    bound: 2mnk FLOP over about 2mnk/sqrt(M) elements read."""
    return math.sqrt(on_chip_bytes / elem_bytes) / elem_bytes


def attention_intensity_cap(on_chip_bytes: float, head_dim: int, elem_bytes: int) -> float:
    """Upper limit on attention's FLOP per DRAM byte for many scores:
    4*E*d FLOP over E*d^2 / (M + 4d^2) elements moved."""
    M = on_chip_bytes / elem_bytes
    return 4 * (M + 4 * head_dim**2) / (head_dim * elem_bytes)


def pebble_critical_on_chip(g: Gemm) -> float:
    """Largest on-chip size (bytes) at which the pebble term exceeds the
    compulsory read term for this GEMM; 0 if it never does. Bisection on M."""
    e = g.elem_bytes

    def excess(M: float) -> float:
        return mmm_read_lower_bound(g.m, g.n, g.k, M) - max(g.k * g.n, g.k * g.n + g.m * g.k - M)

    lo, hi = 1.0, float(g.k * g.n + g.m * g.k + g.m * g.n)
    if excess(lo) <= 0:
        return 0.0
    if excess(hi) > 0:
        return hi * e
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    return lo * e
