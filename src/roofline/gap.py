"""Per-plan cost T_A(plan) and the gap between an engine and the bound
(design notes 5.3, 5.4, 6.9).

The engine's time is split into telescoping pieces measured by ablation
(benchmarks/ablation_mlx.py), so the pieces add up to the measured total:

decode    gemv       matmuls only (weights streamed)
          attention  + scaled_dot_product_attention over the KV cache
          small      + norms, RoPE, residual adds, SwiGLU, KV-cache writes, argmax
          engine     + sampling sync, Python and the generate loop
prefill   gemm, attention, small (elementwise), engine, same idea per chunk

Each piece has an ideal share of the physical bound (bytes / B_dram or
FLOPs / P_peak) and a model prediction from kernel-level calibration
(configs/hardware/*.kernels.json): fixed cost per dependent kernel t0,
streaming bandwidth B_eff, measured kernel times P(shape), CPU encode cost e.
The model's sum is T_A(plan); measured - model per piece is what the model
does not explain (negative when kernels overlap more than in isolation).
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path

from roofline.bound import physical_bound
from roofline.device import Device
from roofline.model import ModelSpec
from roofline.workload import decode, prefill

VIEWS = {"Transpose", "Flatten", "Unflatten", "Reshape", "Broadcast", "Slice", "Squeeze",
         "ExpandDims", "AsStrided", "StopGradient", "Depends"}


def _interp(points: list[tuple[float, float]], x: float) -> float:
    """Piecewise linear in log-log, clamped extrapolation by the end slopes."""
    pts = sorted(points)
    xs = [math.log(p[0]) for p in pts]
    ys = [math.log(p[1]) for p in pts]
    lx = math.log(x)
    i = min(max(bisect.bisect_left(xs, lx), 1), len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return math.exp(y0 + (y1 - y0) * (lx - x0) / (x1 - x0))


@dataclass(frozen=True)
class Calibration:
    data: dict

    @property
    def t0(self) -> float:
        return self.data["chain"]["t0_s"]

    @property
    def B_eff(self) -> float:
        return self.data["chain"]["B_eff"]

    @property
    def e(self) -> float:
        return self.data["chain"]["e_s"]

    @property
    def c_commit(self) -> float:
        return self.data["chain"]["c_commit_s"]

    @property
    def c_sync(self) -> float:
        return self.data["chain"]["c_sync_s"]

    def gemm(self, name: str, M: int) -> float:
        return _interp([tuple(p) for p in self.data["gemm"][name]], M)

    def sdpa(self, Lq: int, Lk: int) -> float:
        rows = self.data["sdpa"]
        if Lq == 1:
            return _interp([(lk, t) for q, lk, t in rows if q == 1], Lk)
        if Lq == Lk:  # first chunk: square causal
            return _interp([(lk, t) for q, lk, t in rows if q == lk], Lk)
        # Later chunks: Lq queries at the end of Lk keys; scale the measured
        # (Lq_ref, Lk) row by the number of query rows.
        ref = max(q for q, lk, t in rows if q > 1)
        return _interp([(lk, t) for q, lk, t in rows if q == ref and lk >= ref], Lk) * Lq / ref

    def small(self, op: str, rows: float) -> float:
        return _interp([tuple(p) for p in self.data["small"][op]], max(rows, 1))


def load_calibration(path: str | Path) -> Calibration:
    return Calibration(json.loads(Path(path).read_text()))


def kernel_kinds(primitives: dict[str, int]) -> dict[str, int]:
    """GPU kernels of one graph by kind (view-only primitives dispatch nothing)."""
    out = {"gemm": 0, "attention": 0, "small": 0}
    for k, n in primitives.items():
        if k in VIEWS:
            continue
        out["gemm" if k == "Matmul" else "attention" if k == "ScaledDotProductAttention" else "small"] += n
    return out


def _piece(measured: float, ideal: float, model: float, **terms: float) -> dict:
    return {"measured_s": measured, "ideal_s": ideal, "model_s": model, "loss_s": measured - ideal,
            "unexplained_s": measured - model, "terms": terms}


def decode_gap(m: ModelSpec, device: Device, cal: Calibration, row: dict, structure: dict,
               engine: str = "gpu") -> dict:
    """Decomposition of one decode step at context row['ctx']."""
    ctx = row["ctx"]
    w = decode(m, 1, ctx)
    B = device.bandwidth(engine)
    phys = physical_bound(w, device, engine, m.weight_dtype).time_s
    matmul_w = (m.num_layers * m.layer_linear_params + m.vocab_size * m.hidden_size) * m.weight_bytes_per_param
    kv = w.bytes["kv_read"]
    rest = w.total_bytes - matmul_w - kv
    kinds = kernel_kinds(structure["decode_primitives"])

    gemv_model = kinds["gemm"] * cal.t0 + matmul_w / cal.B_eff
    attn_model = kinds["attention"] * cal.sdpa(1, ctx)
    small_model = kinds["small"] * cal.t0 + rest / cal.B_eff
    gpu = gemv_model + attn_model + small_model
    n = sum(kinds.values())
    cpu = structure["decode_graph_build_s"] + n * cal.e + cal.c_commit
    engine_model = max(0.0, cpu - gpu)

    gemv = row["gemv_s"]
    attn = row["gemv_attn_s"] - row["gemv_s"]
    small = row["forward_s"] - row["gemv_attn_s"]
    eng = row["generate_s"] - row["forward_s"]
    pieces = {
        "gemv": _piece(gemv, matmul_w / B, gemv_model,
                       fixed=kinds["gemm"] * cal.t0, bandwidth=matmul_w * (1 / cal.B_eff - 1 / B)),
        "attention": _piece(attn, kv / B, attn_model, kernels=kinds["attention"]),
        "small": _piece(small, rest / B, small_model, fixed=kinds["small"] * cal.t0),
        "engine": _piece(eng, 0.0, engine_model, cpu_per_token=cpu, gpu_model=gpu),
    }
    return {"phase": "decode", "ctx": ctx, "phys_s": phys, "measured_s": row["generate_s"],
            "model_s": gpu + engine_model, "kernels": kinds, "pieces": pieces}


def prefill_chunks(L: int, chunk: int = 2048) -> list[tuple[int, int]]:
    """(start, size) as mlx-lm prefills: L - 1 tokens in <= chunk pieces, then the
    last token alone (it alone gets logits)."""
    return [(s, min(chunk, L - 1 - s)) for s in range(0, L - 1, chunk)] + [(L - 1, 1)]


def prefill_gap(m: ModelSpec, device: Device, cal: Calibration, row: dict, structure: dict,
                chunk: int = 2048, engine: str = "gpu") -> dict:
    """Decomposition of time to first token for a prompt of row['L'] tokens."""
    L = row["L"]
    w = prefill(m, 1, L)
    pb = physical_bound(w, device, engine, m.weight_dtype)
    P = device.engine(engine).peak_flops(m.weight_dtype).effective
    B = device.bandwidth(engine)
    chunks = prefill_chunks(L, chunk)
    projs = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    gemm_model = sum(m.num_layers * sum(cal.gemm(p, c) for p in projs) for _, c in chunks) + cal.gemm("lm_head", 1)
    attn_model = sum(m.num_layers * cal.sdpa(c, s + c) for s, c in chunks if c > 1)
    attn_model += m.num_layers * cal.sdpa(1, L)  # the last token attends to all L keys
    per_layer = lambda c: (2 * cal.small("rms_norm", c) + cal.small("rms_norm", c * m.q_dim / m.hidden_size)  # noqa: E731
                           + cal.small("rms_norm", c * m.kv_dim / m.hidden_size)
                           + cal.small("rope", c * m.q_dim / m.hidden_size) + cal.small("rope", c * m.kv_dim / m.hidden_size)
                           + 2 * cal.small("add", c) + cal.small("swiglu", c) + 2 * cal.small("slice_update", c))
    small_model = sum(m.num_layers * per_layer(c) for _, c in chunks)
    engine_model = len(chunks) * cal.c_sync

    lin = w.flops["linear"] + w.flops["lm_head"]
    gemm_ideal = max(lin / P, w.total_bytes / B)
    attn_ideal = w.flops["attention"] / P
    gemm = row["gemm_s"]
    attn = row["gemm_attn_s"] - row["gemm_s"]
    small = row["forward_s"] - row["gemm_attn_s"]
    eng = row["generate_s"] - row["forward_s"]
    pieces = {
        "gemm": _piece(gemm, gemm_ideal, gemm_model, efficiency=lin / gemm_model * 1 / P if gemm_model else 0.0),
        "attention": _piece(attn, attn_ideal, attn_model),
        "small": _piece(small, 0.0, small_model),
        "engine": _piece(eng, 0.0, engine_model, chunks=len(chunks)),
    }
    return {"phase": "prefill", "L": L, "phys_s": pb.time_s, "measured_s": row["generate_s"],
            "model_s": gemm_model + attn_model + small_model + engine_model,
            "ideal_overlap_s": gemm_ideal + attn_ideal - pb.time_s, "pieces": pieces}


def decode_plan_time(cal: Calibration, n_gemm: int, n_attention: int, n_small: int, weight_bytes: float,
                     ctx: int, other_bytes: float = 0.0, cpu_per_token: float = 0.0) -> float:
    """T_A for a decode plan described by its kernel counts: GPU work of the
    dependent kernels, overlapped with the CPU per token (pipelined loop)."""
    gpu = (n_gemm + n_small) * cal.t0 + (weight_bytes + other_bytes) / cal.B_eff + n_attention * cal.sdpa(1, ctx)
    return max(gpu, cpu_per_token)
