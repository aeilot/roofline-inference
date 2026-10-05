from pathlib import Path

import pytest

from roofline import decode, load_device, load_model, physical_bound, prefill
from roofline.device import Device, Engine, Quantity

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "configs/models/qwen3_0.6b.yaml"
M4 = ROOT / "configs/hardware/m4_mba_24g.yaml"


def q(v):
    return Quantity(value=v, source="estimate", confidence="low")


@pytest.fixture
def model():
    return load_model(QWEN)


@pytest.fixture
def device():
    """Round numbers so expectations can be checked by hand,
    independent of later measurements in the real spec."""
    gpu = Engine(name="gpu", compute={"bf16": q(4e12)}, bandwidth=q(100e9), usable_memory=q(16e9))
    return Device(name="toy", total_bandwidth=q(120e9), capacity=q(24e9), engines={"gpu": gpu})


def test_qwen3_param_count(model):
    # 0.6B total, 0.44B non-embedding (Qwen3 model card).
    assert model.total_params == 596_049_920
    assert model.total_params - model.embedding_params == 440_467_456
    assert model.kv_bytes_per_token == 2 * 28 * 8 * 128 * 2  # 112 KiB


def test_decode_batch1_is_bandwidth_bound(model, device):
    b = physical_bound(decode(model, batch=1, context=0), device)
    assert b.bottleneck == "dram_bandwidth"
    # Tied embeddings: every parameter is read once (embedding table == LM head).
    expected_bytes = model.weight_bytes + model.hidden_size * 2 + model.kv_bytes_per_token
    assert b.workload.total_bytes == expected_bytes
    # Engine bandwidth (100 GB/s) is below the total (120 GB/s), so it applies.
    assert b.time_s == pytest.approx(expected_bytes / 100e9)
    assert b.tokens_per_s == pytest.approx(100e9 / expected_bytes)
    assert b.slack["dram_bandwidth"] == 1.0
    assert b.slack["compute"] > 1.0


def test_decode_large_batch_becomes_compute_bound(model, device):
    assert physical_bound(decode(model, batch=256, context=0), device).bottleneck == "compute"


def test_decode_kv_traffic_grows_with_context(model, device):
    short = physical_bound(decode(model, batch=1, context=0), device)
    long = physical_bound(decode(model, batch=1, context=32768), device)
    assert long.workload.bytes["kv_read"] == 32768 * model.kv_bytes_per_token
    assert long.tokens_per_s < short.tokens_per_s


def test_prefill_long_prompt_is_compute_bound(model, device):
    b = physical_bound(prefill(model, batch=1, prompt_len=1024), device)
    assert b.bottleneck == "compute"
    p = 1024
    attn = 4 * 28 * 16 * 128 * (p * (p + 1) // 2)
    assert b.workload.flops["attention"] == attn
    assert b.workload.flops["lm_head"] == 2 * model.vocab_size * model.hidden_size
    assert b.to_dict()["ttft_s"] == b.time_s


def test_capacity_check(model, device):
    assert physical_bound(decode(model, batch=1, context=4096), device).fits
    # 64 x 4096 tokens x 112 KiB = 30 GB of KV cache.
    assert not physical_bound(decode(model, batch=64, context=4096), device).fits


def test_measured_overrides_value():
    qty = Quantity(value=120e9, source="datasheet", confidence="medium", measured=95e9)
    assert qty.effective == 95e9
    assert qty.describe()["origin"] == "measured"


def test_real_specs_load(model):
    m4 = load_device(M4)
    b = physical_bound(decode(model, batch=1, context=0), m4)
    assert b.to_dict()["inputs"]["gpu.bandwidth"]["origin"] in {"estimate", "measured"}
    assert 0 < b.tokens_per_s < 1000


# --- finite on-chip memory (pebble.py) ---------------------------------------

from roofline import operator_bound  # noqa: E402
from roofline.pebble import gemm_traffic, mmm_read_lower_bound, pebble_critical_on_chip  # noqa: E402
from roofline.workload import Gemm  # noqa: E402


def test_mmm_read_lower_bound_formula():
    # Smith et al. Corollary 3.8: 2mnk/sqrt(M) - mn - 2M.
    assert mmm_read_lower_bound(1000, 1000, 1000, 10_000) == 2e9 / 100 - 1e6 - 2e4


def test_operator_bound_reduces_to_physical_with_unlimited_on_chip(model, device):
    for w in (decode(model, 8, 512), prefill(model, 2, 2048)):
        p = physical_bound(w, device)
        o = operator_bound(w, device, on_chip_bytes=float("inf"))
        assert o.dram_bytes == pytest.approx(p.dram_bytes)
        assert o.time_s == pytest.approx(p.time_s)


def test_operator_traffic_grows_as_on_chip_shrinks(model, device):
    w = prefill(model, 4, 2048)
    sizes = [2**k for k in range(14, 26)]  # 16 KiB .. 32 MiB
    traffic = [operator_bound(w, device, on_chip_bytes=s).dram_bytes for s in sizes]
    assert all(a >= b for a, b in zip(traffic, traffic[1:]))
    assert traffic[-1] >= physical_bound(w, device).dram_bytes


def test_pebble_term_binds_below_critical_size():
    g = Gemm("x", m=4096, n=4096, k=4096, count=1, elem_bytes=2)
    crit = pebble_critical_on_chip(g)
    assert 0 < crit < (g.k * g.n + g.m * g.k) * 2
    assert gemm_traffic(g, crit / 2).binding_read_term == "pebble"
    assert gemm_traffic(g, crit * 2).binding_read_term != "pebble"


def test_operator_bound_needs_on_chip(model, device):
    with pytest.raises(ValueError):
        operator_bound(decode(model, 1, 0), device)  # toy device has no on_chip


def test_real_spec_on_chip_loads(model):
    b = operator_bound(prefill(model, 1, 1024), load_device(M4))
    assert b.to_dict()["inputs"]["memory.on_chip"]["origin"] in {"estimate", "measured"}
    assert b.to_dict()["bound"] == "operator"
