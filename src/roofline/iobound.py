"""Multi-level, multi-processor I/O lower bounds on the computation graph
(design notes 6.8).

Each boundary of the memory hierarchy gets its own lower bound Q_b on the
bytes that must cross it, hence an arithmetic-intensity cap FLOPs / Q_b and a
time bound Q_b / B_b. Two bounds per boundary:

fused     Any implementation allowed at level A (design notes 9: every product
          term computed at least once, summation order, softmax form, fusion
          and recomputation free). The larger of compulsory traffic (every
          used input element crosses once, every output once, minus what may
          already sit in fast memory, P * S) and the fused pebble bound below
          (design notes 6.12).

operator  Every contraction and every attention is its own operator whose
          activation input comes from beyond the boundary and whose output
          goes there (at most P * S may stay). Elementwise and row-wise work
          is fused into neighbours and free; residual-stream traffic is not
          counted. Per operator the larger of
            compulsory  weights + (A - P S)+ + max(new K/V, C - P S)
            pebble      GEMM: 2mnk / sqrt(S) - 3 P S       (Smith et al. 2019,
                        Lemma 3.4 per core; see design notes 6.8)
                        attention: E d^2 / (S + 4 d^2) - P S  (Saha & Ye 2024,
                        constants in design notes 6.6)
          summed over operators, minus P S once for inputs already resident.
          The fused bound also holds for these schedules, so the larger of
          the two is reported.

Multi-level (Savage 1995): the boundary just outside level l sees all storage
of levels <= l as fast memory. Multi-processor: a private level has P
instances; per-core bounds are summed over cores. Capacities are upper
estimates, so the bounds stay valid; omitted traffic only weakens them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from roofline.device import Device
from roofline.graph import Graph, Op


@dataclass(frozen=True)
class Boundary:
    name: str  # level just inside the boundary
    fast_bytes: float  # S: fast storage per processor on the inner side
    processors: int  # P: private copies of that storage
    bandwidth: float  # aggregate bytes/s across the boundary
    dram: bool
    inputs: dict


def boundaries(device: Device, engine: str = "gpu") -> tuple[Boundary, ...]:
    if not device.hierarchy:
        raise ValueError(f"device {device.name!r} has no memory hierarchy")
    out = []
    for i, lv in enumerate(device.hierarchy):
        inner = device.hierarchy[: i + 1]
        S = sum(l.capacity.effective * l.instances for l in inner) / lv.instances
        bw = device.bandwidth(engine) if lv.dram else lv.bandwidth.effective
        inputs = {f"{l.name}.capacity": {**l.capacity.describe(), "instances": l.instances} for l in inner}
        inputs[f"{lv.name}.bandwidth"] = lv.bandwidth.describe()
        out.append(Boundary(lv.name, S, lv.instances, bw, lv.dram, inputs))
    return tuple(out)


# --- operators of the graph --------------------------------------------------


@dataclass(frozen=True)
class GemmOp:
    """Contractions sharing one activation input, merged: C[m x n] = A[m x k] @ W[k x n]."""

    name: str
    group: str
    m: int
    n: int
    k: int
    elem_bytes: int
    persist: int  # output elements that must reach memory (new K/V)

    @property
    def flops(self) -> float:
        return 2 * self.m * self.n * self.k


@dataclass(frozen=True)
class AttentionOp:
    name: str
    scores: int
    head_dim: int
    q: int  # elements of Q (= O)
    kv_new: int  # elements of new K and V, produced on chip by the QKV operator
    kv_cached: int  # elements of cached K and V, graph inputs
    elem_bytes: int

    @property
    def flops(self) -> float:
        return 4 * self.scores * self.head_dim


def _persisted(graph: Graph, tensors: list[str]) -> int:
    """Elements of new K/V reached from `tensors` through elementwise and
    row-wise ops only (the epilogue of the operator that produced them)."""
    seen, stack, total = set(), list(tensors), 0
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if graph.tensors[t].role == "kv_new":
            total += graph.tensors[t].numel
        for op in graph.consumers(t):
            if op.kind in ("elementwise", "rowwise"):
                stack.extend(op.outputs)
    return total


def operators(graph: Graph) -> tuple[list[GemmOp], list[AttentionOp], float]:
    """Merged GEMMs, attentions, and the bytes of all other compulsory traffic
    (inputs used by no contraction or attention, sampled tokens)."""
    merged: dict[str, list[Op]] = {}
    for op in graph.ops:
        if op.kind == "contraction":
            merged.setdefault(op.inputs[0], []).append(op)
    gemms = []
    for ops in merged.values():
        k = ops[0].shape["k"]
        if any(o.shape["k"] != k or o.shape["m"] != ops[0].shape["m"] for o in ops):
            raise ValueError(f"cannot merge {[o.name for o in ops]}")
        names = [o.name.rsplit(".", 1)[-1].removesuffix("_proj") for o in ops]
        prefix = ops[0].name.rsplit(".", 1)[0] + "." if "." in ops[0].name else ""
        gemms.append(GemmOp(
            name=prefix + "_".join(names), group=ops[0].group, m=ops[0].shape["m"],
            n=sum(o.shape["n"] for o in ops), k=k, elem_bytes=graph.tensors[ops[0].inputs[1]].elem_bytes,
            persist=_persisted(graph, [t for o in ops for t in o.outputs]),
        ))
    attentions = []
    for op in graph.ops:
        if op.kind != "attention":
            continue
        q, k, v = (graph.tensors[t] for t in op.inputs[:3])
        cached = sum(graph.tensors[t].numel for t in op.inputs[3:])
        attentions.append(AttentionOp(op.name, op.shape["scores"], op.shape["head_dim"],
                                      q.numel, k.numel + v.numel, cached, q.elem_bytes))
    heavy = {t for op in graph.ops if op.kind in ("contraction", "attention") for t in op.inputs}
    other = sum(graph.compulsory_read_bytes(t.name) for t in graph.inputs if t.name not in heavy)
    other += sum(t.bytes for t in graph.outputs if t.role == "sampled")
    return gemms, attentions, other


# --- per-operator terms (elements) --------------------------------------------


def gemm_terms(g: GemmOp, S: float, P: int) -> dict[str, float]:
    """Lower bounds (elements) on one GEMM's traffic across a boundary with
    S elements of fast memory per processor and P processors; weights counted
    cold (the caller subtracts resident inputs once)."""
    W, A, C = g.k * g.n, g.m * g.k, g.m * g.n
    compulsory = W + max(0.0, A - P * S) + max(g.persist, C - P * S)
    pebble = -math.inf if math.isinf(S) else 2 * g.m * g.n * g.k / math.sqrt(S) - 3 * P * S
    return {"compulsory": compulsory, "pebble": max(0.0, pebble)}


def attention_terms(a: AttentionOp, S: float, P: int) -> dict[str, float]:
    compulsory = a.kv_cached + max(0.0, a.q + a.kv_new - P * S) + max(0.0, a.q - P * S)
    d2 = a.head_dim**2
    pebble = -math.inf if math.isinf(S) else a.scores * d2 / (S + 4 * d2) - P * S
    return {"compulsory": compulsory, "pebble": max(0.0, pebble)}


# --- fused pebble bound (design notes 6.12) ------------------------------------
#
# Valid for any level-A implementation: fusion, any summation order, softmax
# form, and recomputation. Segment argument (Hong-Kung / Smith): split the
# schedule of one processor into segments of R I/O operations. The values a
# segment can use from outside, D (red at its start or read in it), number at
# most X = S + R, and they dominate everything it computes.
#
# Counted products: the matmuls whose activation input sits behind a residual
# update, all layers (gate_up, down) and layers >= 1 (qkv); o_proj, attention,
# lm_head and layer-0 qkv are left out (fewer counted products only weaken the
# bound). Per segment, a useful product of contraction c feeds an output that
# is either completed in the segment from products computed in it ("full":
# needs a whole A row and a whole W column) or crosses the segment boundary as
# a partial sum (in D at the start, or red / written at the end). So
#
#     products of c  <=  sqrt(alpha_c beta_c gamma_c) + alpha_c beta_c / K_c
#
# (Loomis-Whitney for the partial ones), beta_c = c's weights in D, gamma_c =
# c's partial sums in D plus those leaving (<= S + R in total), alpha_c = A
# values of c available in the segment. An A value is available if a vertex of
# its private chain is in D, or it is recomputed from scratch, which needs the
# column's upstream weights (omega_c of them) in D and every row prerequisite:
#
#   qkv_l:     column k needs Wd_{l-1}[:,k] and Wo_{l-1}[:,k]; row i needs
#              act_{l-1}[i,:] (per column j: a private vertex, or Wg/Wu[:,j])
#              and attn_{l-1}[i,:] (per V column: >= |R| V values, or Wv column)
#   gate_up_l: column k needs Wo_l[:,k]; row i needs attn_l[i,:] (as above)
#   down_l:    column j needs Wg_l[:,j], Wu_l[:,j]; row i needs h2_l[i,:] (per
#              column k: a private vertex, or Wo_l[:,k])
#
# so alpha_c <= X + rec_c(X), with rec_c the most (row x column) values such a
# budget can recompute. Weights and partial sums of different contractions are
# distinct vertices; by Cauchy-Schwarz the products of all counted contractions
# in one segment are at most
#
#     F(X) = X sqrt(X + r*) + (X + r*) X / K_min,     r* = max_c rec_c(X)
#
# and with P processors (per-core segments summed) the traffic is at least
# R (N / F(S + R) - P) for every R. Dropped prerequisites only enlarge rec_c.


@dataclass(frozen=True)
class FusedContraction:
    """Parameters of one counted contraction type for the fused bound."""

    name: str
    rows: int  # m: rows of the activation input (tokens)
    K: int  # inner dimension
    n: int  # output columns
    count: int  # instances per step (layers)
    omega: int  # upstream weights per A column for a from-scratch recompute
    row_cost: tuple[tuple[int, int], ...]  # (columns, cap): row prerequisites cost sum cols * min(r, cap)

    @property
    def products(self) -> float:
        return self.rows * self.K * self.n * self.count


def fused_contractions(graph: Graph) -> list[FusedContraction]:
    """Counted contraction types of a Qwen3-style graph (qkv, gate_up, down)."""
    gemms, _, _ = operators(graph)
    by = {}
    for g in gemms:
        if "." not in g.name:
            continue  # lm_head
        layer, kind = g.name.split(".", 1)
        by.setdefault(kind, []).append((int(layer[1:]), g))
    qkv, o, gu, down = (by.get(k, []) for k in ("q_k_v", "o", "gate_up", "down"))
    if not (qkv and o and gu and down):
        return []
    hidden, q_dim, inter = qkv[0][1].k, o[0][1].k, down[0][1].k
    kv_dim = (qkv[0][1].n - q_dim) // 2
    m = qkv[0][1].m
    attn_rows = ((kv_dim, hidden),)  # per V column: >= r V values or the Wv column
    return [
        FusedContraction("qkv", m, hidden, qkv[0][1].n, sum(1 for l, _ in qkv if l >= 1),
                         inter + q_dim, ((inter, 2 * hidden),) + attn_rows),
        FusedContraction("gate_up", m, hidden, gu[0][1].n, len(gu), q_dim, attn_rows),
        FusedContraction("down", m, inter, down[0][1].n, len(down), 2 * hidden, ((hidden, q_dim),)),
    ]


def recompute_capacity(c: FusedContraction, Y: float) -> float:
    """rec_c(Y): most A values of c recomputable from scratch with Y values in
    D, max over rows r of r * min(K, (Y - cost(r)) / omega). Exact maximum of
    the continuous relaxation: cost(r) is piecewise linear, the objective
    concave on each piece, so endpoints, stationary points and kinks suffice."""
    def cost(r: float) -> float:
        return sum(cols * min(r, cap) for cols, cap in c.row_cost)

    def f(r: float) -> float:
        return r * min(c.K, max(0.0, (Y - cost(r)) / c.omega))

    bps = sorted({1.0, float(c.rows)} | {float(cap) for _, cap in c.row_cost if 1 < cap < c.rows})
    best = f(1.0)
    for lo, hi in zip(bps, bps[1:]):
        c1 = sum(cols for cols, cap in c.row_cost if cap >= hi)
        c0 = sum(cols * cap for cols, cap in c.row_cost if cap <= lo)
        cands = [lo, hi]
        if c1 > 0:
            cands += [(Y - c0) / (2 * c1), (Y - c0 - c.omega * c.K) / c1]
        best = max([best] + [f(r) for r in cands if lo <= r <= hi])
    return best


def fused_products_per_segment(cs: list[FusedContraction], X: float) -> float:
    """F(X): most counted products one segment with |D| <= X can compute."""
    r = max(recompute_capacity(c, X) for c in cs)
    kmin = min(c.K for c in cs)
    return X * math.sqrt(X + r) + (X + r) * X / kmin


def fused_pebble_elements(cs: list[FusedContraction], S: float, P: int = 1) -> float:
    """Lower bound (elements moved, reads + writes) for any implementation with
    S elements of fast memory per processor: max over R of R (N / F(S + R) - P)."""
    if not cs or math.isinf(S):
        return 0.0
    N = sum(c.products for c in cs)
    best = 0.0
    for R in S * 2.0 ** (np.arange(-6, 8.01, 0.125)):
        best = max(best, R * (N / fused_products_per_segment(cs, S + R) - P))
    return best


# --- bounds -------------------------------------------------------------------


@dataclass(frozen=True)
class LevelBound:
    boundary: Boundary
    fused_bytes: float
    operator_bytes: float  # already >= fused_bytes
    groups: dict[str, dict]  # operator bound by roofline group, before the resident-input credit
    fused_terms: dict[str, float] | None = None  # compulsory and pebble parts of the fused bound

    def bytes(self, kind: str) -> float:
        return {"fused": self.fused_bytes, "operator": self.operator_bytes}[kind]

    def time_s(self, kind: str) -> float:
        return self.bytes(kind) / self.boundary.bandwidth


@dataclass(frozen=True)
class IOBound:
    graph: Graph
    flops: float
    peak_flops: float
    levels: tuple[LevelBound, ...]
    inputs: dict

    def resource_time_s(self, kind: str) -> dict[str, float]:
        out = {"compute": self.flops / self.peak_flops}
        out |= {f"io.{lv.boundary.name}": lv.time_s(kind) for lv in self.levels}
        return out

    def time_s(self, kind: str) -> float:
        return max(self.resource_time_s(kind).values())

    def bottleneck(self, kind: str) -> str:
        r = self.resource_time_s(kind)
        return max(r, key=r.get)

    def intensity(self, kind: str) -> dict[str, float]:
        """Arithmetic-intensity cap per boundary, FLOP per byte crossing it."""
        return {lv.boundary.name: self.flops / lv.bytes(kind) for lv in self.levels}

    def to_dict(self) -> dict:
        g = self.graph
        out = {"phase": g.phase, "batch": g.batch, "new": g.new, "cached": g.cached,
               "flops": self.flops, "peak_flops": self.peak_flops, "inputs": self.inputs}
        for kind in ("fused", "operator"):
            out[kind] = {
                "time_s": self.time_s(kind),
                "bottleneck": self.bottleneck(kind),
                "resource_time_s": self.resource_time_s(kind),
                "intensity": self.intensity(kind),
                "bytes": {lv.boundary.name: lv.bytes(kind) for lv in self.levels},
            }
        out["operator"]["groups"] = {lv.boundary.name: lv.groups for lv in self.levels}
        out["fused"]["terms"] = {lv.boundary.name: lv.fused_terms for lv in self.levels}
        return out


def _level(graph: Graph, b: Boundary, gemms: list[GemmOp], attns: list[AttentionOp], other: float,
           fused_cs: list[FusedContraction]) -> LevelBound:
    P = b.processors
    compulsory = max(0.0, graph.total_compulsory_bytes - P * b.fast_bytes)
    e = gemms[0].elem_bytes if gemms else 2
    pebble = fused_pebble_elements(fused_cs, b.fast_bytes / e, P) * e
    fused = max(compulsory, pebble)
    groups: dict[str, dict] = {}

    def add(group: str, flops: float, terms: dict[str, float], e: int) -> None:
        r = groups.setdefault(group, {"flops": 0.0, "bytes": 0.0, "binding": {}})
        r["flops"] += flops
        r["bytes"] += max(terms.values()) * e
        term = max(terms, key=terms.get)
        r["binding"][term] = r["binding"].get(term, 0) + 1

    for g in gemms:
        add(g.group, g.flops, gemm_terms(g, b.fast_bytes / g.elem_bytes, P), g.elem_bytes)
    for a in attns:
        add("attention", a.flops, attention_terms(a, b.fast_bytes / a.elem_bytes, P), a.elem_bytes)
    total = sum(r["bytes"] for r in groups.values()) + other
    operator = max(fused, total - P * b.fast_bytes)
    return LevelBound(b, fused, operator, groups, {"compulsory": compulsory, "pebble": pebble})


def traffic_at(graph: Graph, fast_bytes: float, processors: int = 1) -> tuple[float, float]:
    """(fused, operator) lower bounds on bytes crossing one boundary with
    `fast_bytes` of fast memory per processor: Q_min(S) for any S, independent
    of a device spec."""
    gemms, attns, other = operators(graph)
    b = Boundary("S", fast_bytes, processors, math.nan, False, {})
    lv = _level(graph, b, gemms, attns, other, fused_contractions(graph))
    return lv.fused_bytes, lv.operator_bytes


def io_bound(graph: Graph, device: Device, engine: str = "gpu", dtype: str = "bf16") -> IOBound:
    peak = device.engine(engine).peak_flops(dtype)
    gemms, attns, other = operators(graph)
    fused_cs = fused_contractions(graph)
    levels = tuple(_level(graph, b, gemms, attns, other, fused_cs) for b in boundaries(device, engine))
    inputs = {f"{engine}.compute.{dtype}": peak.describe()}
    for lv in levels:
        inputs |= lv.boundary.inputs
    return IOBound(graph, graph.total_flops, peak.effective, levels, inputs)
