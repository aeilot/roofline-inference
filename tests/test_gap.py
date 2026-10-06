import math
from pathlib import Path

import pytest

from roofline import load_device, load_model
from roofline.gap import Calibration, _interp, decode_gap, decode_plan_time, kernel_kinds, prefill_gap

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "configs/models/qwen3_0.6b.yaml"
M4 = ROOT / "configs/hardware/m4_mba_24g.yaml"

# Hand-made calibration: round numbers, independent of measurements.
CAL = Calibration({
    "chain": {"e_s": 2e-6, "c_commit_s": 2e-5, "t0_s": 2e-6, "B_eff": 1e11, "c_sync_s": 1e-4},
    "gemm": {p: [[1, 1e-5], [2048, 1e-3]] for p in
             ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")},
    "sdpa": [[1, 512, 1e-5], [1, 32768, 6.4e-4], [16, 16, 1e-5], [2048, 2048, 4e-3], [2048, 32768, 1.2e-1]],
    "small": {op: [[1, 3e-6], [2048, 1e-4]] for op in ("rms_norm", "rope", "add", "swiglu", "slice_update")},
})
STRUCTURE = {"decode_primitives": {"Matmul": 197, "ScaledDotProductAttention": 28, "RMSNorm": 113,
                                   "Transpose": 300, "Add": 56}, "decode_graph_build_s": 5e-4}


@pytest.fixture
def model():
    return load_model(QWEN)


@pytest.fixture
def device():
    return load_device(M4)


def test_interp_is_exact_on_points_and_log_linear_between():
    pts = [(1, 1.0), (100, 100.0)]
    assert _interp(pts, 1) == pytest.approx(1.0)
    assert _interp(pts, 10) == pytest.approx(10.0)
    assert _interp(pts, 1000) == pytest.approx(1000.0)  # extrapolates along the end slope


def test_kernel_kinds_skip_views():
    assert kernel_kinds(STRUCTURE["decode_primitives"]) == {"gemm": 197, "attention": 28, "small": 169}


def test_decode_pieces_telescope_to_measured(model, device):
    row = {"ctx": 4096, "gemv_s": 0.013, "gemv_attn_s": 0.0145, "forward_s": 0.0155, "generate_s": 0.016}
    g = decode_gap(model, device, CAL, row, STRUCTURE)
    p = g["pieces"]
    assert sum(x["measured_s"] for x in p.values()) == pytest.approx(row["generate_s"])
    # Ideal shares add up to the physical bound (decode is DRAM bound).
    assert sum(x["ideal_s"] for x in p.values()) == pytest.approx(g["phys_s"], rel=1e-9)
    assert p["gemv"]["terms"]["fixed"] == pytest.approx(197 * 2e-6)
    assert g["model_s"] == pytest.approx(sum(x["model_s"] for x in p.values()))


def test_prefill_pieces_telescope_to_measured(model, device):
    row = {"L": 4096, "gemm_s": 1.2, "gemm_attn_s": 1.6, "forward_s": 1.7, "generate_s": 1.75}
    g = prefill_gap(model, device, CAL, row, {})
    assert sum(x["measured_s"] for x in g["pieces"].values()) == pytest.approx(1.75)
    assert g["pieces"]["engine"]["terms"]["chunks"] == 3  # 2048 + 2047 + the last token
    assert g["ideal_overlap_s"] >= -1e-12


def test_plan_time_drops_with_fewer_kernels():
    full = decode_plan_time(CAL, 197, 28, 311, 1.19e9, 4096)
    fused = decode_plan_time(CAL, 113, 28, 0, 1.19e9, 4096)
    assert fused < full
    assert full - fused == pytest.approx((197 + 311 - 113) * 2e-6)
    # CPU-bound plan: the GPU work is hidden behind the CPU.
    assert decode_plan_time(CAL, 1, 0, 0, 1e6, 4096, cpu_per_token=1.0) == 1.0
