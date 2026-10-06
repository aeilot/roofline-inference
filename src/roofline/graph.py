"""Operator-level computation graph of one inference step (design notes 6.7).

Nodes are operators, edges are tensors. The graph is not element level
(a decode step alone has ~6e8 multiply vertices); each operator carries the
shape the I/O bounds need instead:

    contraction  C[m x n] = A[m x k] @ W[k x n]: m*n*k product terms
    attention    QK^T and PV over `scores` (query, key) pairs, head_dim d

Under the A-level choice (design notes 9) every product term of every
contraction must be computed at least once and everything else is free:
summation order, softmax formulation, fusion, recomputation. I/O bounds may
therefore use only these product sets plus compulsory traffic.

Elementwise and row-wise work (norms, RoPE, SwiGLU, residuals, sampling)
carries no FLOPs, as in workload.py. RoPE tables are generated on chip.

Compulsory traffic: every element of a graph input that is used is read
once, every graph output is written once. An input consumed only by gathers
(an untied embedding table) counts the gathered rows, assuming distinct
tokens; any other consumer makes the whole tensor compulsory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from roofline.model import ModelSpec
from roofline.workload import TOKEN_BYTES

INPUT_ROLES = ("weight", "kv_cache", "token")
OUTPUT_ROLES = ("kv_new", "sampled")


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    elem_bytes: int
    role: str  # INPUT_ROLES, OUTPUT_ROLES or "activation"

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def bytes(self) -> int:
        return self.numel * self.elem_bytes


@dataclass(frozen=True)
class Op:
    name: str
    kind: str  # "contraction" | "attention" | "rowwise" | "elementwise" | "gather" | "sample"
    group: str  # roofline label: "qkv" | "o_proj" | "mlp" | "lm_head" | "attention" | "other"
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    flops: float = 0.0
    shape: dict[str, int] = field(default_factory=dict)
    layer: int | None = None


@dataclass(frozen=True)
class Graph:
    phase: str  # "decode" | "prefill"
    batch: int
    new: int  # new tokens per sequence this step
    cached: int  # cached tokens per sequence before this step
    tensors: dict[str, Tensor]
    ops: tuple[Op, ...]  # topological order

    @property
    def tokens(self) -> int:
        return self.batch * self.new

    @property
    def total_flops(self) -> float:
        return sum(op.flops for op in self.ops)

    @property
    def inputs(self) -> list[Tensor]:
        return [t for t in self.tensors.values() if t.role in INPUT_ROLES]

    @property
    def outputs(self) -> list[Tensor]:
        return [t for t in self.tensors.values() if t.role in OUTPUT_ROLES]

    def consumers(self, tensor: str) -> list[Op]:
        return [op for op in self.ops if tensor in op.inputs]

    def compulsory_read_bytes(self, tensor: str) -> int:
        t = self.tensors[tensor]
        users = self.consumers(tensor)
        if users and all(op.kind == "gather" for op in users):
            row = t.bytes // t.shape[0]
            return min(t.bytes, sum(op.shape["rows"] for op in users) * row)
        return t.bytes

    def compulsory_bytes(self) -> dict[str, int]:
        """Compulsory DRAM traffic by tensor role."""
        out: dict[str, int] = {}
        for t in self.inputs:
            out[t.role] = out.get(t.role, 0) + self.compulsory_read_bytes(t.name)
        for t in self.outputs:
            out[t.role] = out.get(t.role, 0) + t.bytes
        return out

    @property
    def total_compulsory_bytes(self) -> int:
        return sum(self.compulsory_bytes().values())

    def flops_by_group(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for op in self.ops:
            out[op.group] = out.get(op.group, 0.0) + op.flops
        return out

    def validate(self) -> None:
        """Each tensor has at most one producer, inputs have none, every
        operand exists before use, and every output is produced."""
        produced: set[str] = set()
        for op in self.ops:
            for name in op.inputs:
                if name not in self.tensors:
                    raise ValueError(f"{op.name}: unknown tensor {name!r}")
                if self.tensors[name].role not in INPUT_ROLES and name not in produced:
                    raise ValueError(f"{op.name}: {name!r} used before it is produced")
            for name in op.outputs:
                if name in produced or self.tensors[name].role in INPUT_ROLES:
                    raise ValueError(f"{op.name}: {name!r} produced twice or is an input")
                produced.add(name)
        missing = [t.name for t in self.outputs if t.name not in produced]
        if missing:
            raise ValueError(f"outputs never produced: {missing}")


class _Builder:
    def __init__(self, m: ModelSpec) -> None:
        self.m = m
        self.tensors: dict[str, Tensor] = {}
        self.ops: list[Op] = []

    def tensor(self, name: str, shape: tuple[int, ...], role: str = "activation", elem_bytes: int | None = None) -> str:
        if name in self.tensors:
            raise ValueError(f"duplicate tensor {name!r}")
        e = self.m.weight_bytes_per_param if elem_bytes is None else elem_bytes
        self.tensors[name] = Tensor(name, shape, e, role)
        return name

    def op(self, name: str, kind: str, group: str, inputs: tuple[str, ...], outputs: tuple[str, ...],
           flops: float = 0.0, layer: int | None = None, **shape: int) -> None:
        self.ops.append(Op(name, kind, group, inputs, outputs, flops, shape, layer))

    def linear(self, name: str, group: str, x: str, w: str, out: str, layer: int | None = None) -> None:
        m_, k = self.tensors[x].shape
        k_w, n = self.tensors[w].shape
        if k != k_w:
            raise ValueError(f"{name}: inner dims {k} != {k_w}")
        self.op(name, "contraction", group, (x, w), (out,), 2 * m_ * n * k, layer, m=m_, n=n, k=k)


def _build(m: ModelSpec, phase: str, batch: int, new: int, cached: int) -> Graph:
    """Forward pass over `new` tokens per sequence for `batch` sequences, each
    with `cached` tokens of K/V already in DRAM. Query position i (0-based
    among the new tokens) attends to cached + i + 1 keys. Logits are only
    produced for the last position of each sequence."""
    b = _Builder(m)
    rows, d, kv_e = batch * new, m.hidden_size, m.kv_bytes_per_elem
    head_rows = batch

    tokens = b.tensor("tokens", (rows,), "token", TOKEN_BYTES)
    embed = b.tensor("embed", (m.vocab_size, d), "weight")
    x = b.tensor("x0", (rows, d))
    b.op("embed_gather", "gather", "other", (tokens, embed), (x,), rows=rows)

    scores = m.num_heads * batch * (new * cached + new * (new + 1) // 2)
    for l in range(m.num_layers):
        p = f"L{l}."
        W = {
            "in_norm": (d,), "q": (d, m.q_dim), "k": (d, m.kv_dim), "v": (d, m.kv_dim),
            "o": (m.q_dim, d), "post_norm": (d,), "gate": (d, m.intermediate_size),
            "up": (d, m.intermediate_size), "down": (m.intermediate_size, d),
        }
        if m.qk_norm:
            W |= {"q_norm": (m.head_dim,), "k_norm": (m.head_dim,)}
        w = {k: b.tensor(f"{p}W_{k}", s, "weight") for k, s in W.items()}

        h = b.tensor(p + "h", (rows, d))
        b.op(p + "in_norm", "rowwise", "other", (x, w["in_norm"]), (h,), layer=l)
        q, k, v = (b.tensor(p + n, (rows, dim)) for n, dim in
                   (("q_raw", m.q_dim), ("k_raw", m.kv_dim), ("v", m.kv_dim)))
        for name, out in (("q", q), ("k", k), ("v", v)):
            b.linear(f"{p}{name}_proj", "qkv", h, w[name], out, l)
        if m.qk_norm:
            qn, kn = b.tensor(p + "q_normed", (rows, m.q_dim)), b.tensor(p + "k_normed", (rows, m.kv_dim))
            b.op(p + "q_norm", "rowwise", "other", (q, w["q_norm"]), (qn,), layer=l)
            b.op(p + "k_norm", "rowwise", "other", (k, w["k_norm"]), (kn,), layer=l)
            q, k = qn, kn
        # New K (after RoPE) and V are also step outputs: they must reach the cache.
        qr = b.tensor(p + "q", (rows, m.q_dim))
        kr = b.tensor(p + "k", (rows, m.kv_dim), "kv_new", kv_e)
        b.op(p + "rope", "elementwise", "other", (q, k), (qr, kr), layer=l)
        vn = b.tensor(p + "v_new", (rows, m.kv_dim), "kv_new", kv_e)
        b.op(p + "v_store", "elementwise", "other", (v,), (vn,), layer=l)

        attn_in = (qr, kr, vn)
        if cached:
            kc = b.tensor(p + "K_cache", (batch, cached, m.kv_dim), "kv_cache", kv_e)
            vc = b.tensor(p + "V_cache", (batch, cached, m.kv_dim), "kv_cache", kv_e)
            attn_in += (kc, vc)
        o = b.tensor(p + "attn_out", (rows, m.q_dim))
        b.op(p + "attention", "attention", "attention", attn_in, (o,), 4 * scores * m.head_dim, l,
             scores=scores, head_dim=m.head_dim, heads=m.num_heads, kv_heads=m.num_kv_heads,
             batch=batch, new=new, cached=cached)

        a = b.tensor(p + "o", (rows, d))
        b.linear(p + "o_proj", "o_proj", o, w["o"], a, l)
        x1 = b.tensor(p + "x_mid", (rows, d))
        b.op(p + "resid_attn", "elementwise", "other", (x, a), (x1,), layer=l)
        h2 = b.tensor(p + "h2", (rows, d))
        b.op(p + "post_norm", "rowwise", "other", (x1, w["post_norm"]), (h2,), layer=l)
        g = b.tensor(p + "gate", (rows, m.intermediate_size))
        u = b.tensor(p + "up", (rows, m.intermediate_size))
        b.linear(p + "gate_proj", "mlp", h2, w["gate"], g, l)
        b.linear(p + "up_proj", "mlp", h2, w["up"], u, l)
        act = b.tensor(p + "act", (rows, m.intermediate_size))
        b.op(p + "swiglu", "elementwise", "other", (g, u), (act,), layer=l)
        dn = b.tensor(p + "down", (rows, d))
        b.linear(p + "down_proj", "mlp", act, w["down"], dn, l)
        x_next = b.tensor(f"x{l + 1}", (rows, d))
        b.op(p + "resid_mlp", "elementwise", "other", (x1, dn), (x_next,), layer=l)
        x = x_next

    last = b.tensor("x_last", (head_rows, d))
    b.op("last_rows", "gather", "other", (x,), (last,), rows=head_rows)
    final_norm = b.tensor("W_final_norm", (d,), "weight")
    hf = b.tensor("h_final", (head_rows, d))
    b.op("final_norm", "rowwise", "other", (last, final_norm), (hf,))
    if m.tie_word_embeddings:
        head = embed
    else:
        head = b.tensor("lm_head", (m.vocab_size, d), "weight")
    # logits = h_final @ head^T: the table is stored [vocab x d], so not via linear().
    logits = b.tensor("logits", (head_rows, m.vocab_size))
    b.op("lm_head", "contraction", "lm_head", (hf, head), (logits,),
         2 * head_rows * m.vocab_size * d, m=head_rows, n=m.vocab_size, k=d)
    nxt = b.tensor("next_tokens", (batch,), "sampled", TOKEN_BYTES)
    b.op("sample", "sample", "other", (logits,), (nxt,))

    graph = Graph(phase, batch, new, cached, b.tensors, tuple(b.ops))
    graph.validate()
    return graph


def decode_graph(m: ModelSpec, batch: int, context: int) -> Graph:
    """One decode step, matching workload.decode."""
    if batch < 1 or context < 0:
        raise ValueError("batch must be >= 1 and context >= 0")
    return _build(m, "decode", batch, new=1, cached=context)


def prefill_graph(m: ModelSpec, batch: int, prompt_len: int) -> Graph:
    """Causal prefill of `prompt_len` tokens, matching workload.prefill."""
    if batch < 1 or prompt_len < 1:
        raise ValueError("batch and prompt_len must be >= 1")
    return _build(m, "prefill", batch, new=prompt_len, cached=0)
