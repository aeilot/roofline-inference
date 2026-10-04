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
    assert b.to_dict()["inputs"]["gpu.bandwidth"]["origin"] == "estimate"
    assert 0 < b.tokens_per_s < 1000
