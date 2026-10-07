from pathlib import Path

import pytest

from roofline import decode, load_device, load_model, physical_bound
from roofline.kvpolicy import (best_storage_plan, budget, full, parse_policy, retrieval, storage_plans,
                               subgen, subgen_vectors, window)

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "configs/models/qwen3_0.6b.yaml"
M4 = ROOT / "configs/hardware/m4_mba_24g.yaml"


@pytest.fixture
def model():
    return load_model(QWEN)


def test_full_policy_is_level_a(model):
    for ctx in (0, 1, 4096):
        a, b = decode(model, 2, ctx), decode(model, 2, ctx, full())
        assert a.bytes == b.bytes and a.flops == b.flops and a.memory == b.memory
        assert a.special == b.special


def test_level_a_matches_old_formulas(model):
    w = decode(model, 1, 1000)
    assert w.bytes["kv_read"] == 1000 * model.kv_bytes_per_token
    assert w.bytes["kv_write"] == model.kv_bytes_per_token
    assert w.flops["attention"] == 4 * 28 * 16 * 128 * 1001
    assert w.memory["kv_cache"] == 1001 * model.kv_bytes_per_token


def test_bits_scale_kv_bytes_only(model):
    a, b = decode(model, 1, 8192), decode(model, 1, 8192, full(bits=4))
    assert b.bytes["kv_read"] == a.bytes["kv_read"] / 4
    assert b.bytes["weights"] == a.bytes["weights"]
    assert b.flops == a.flops


def test_window_saturates(model):
    p = window(4096, sink=4)
    short, long_ = decode(model, 1, 1000, p), decode(model, 1, 32768, p)
    assert short.bytes["kv_read"] == decode(model, 1, 1000).bytes["kv_read"]
    assert long_.bytes["kv_read"] == decode(model, 1, 4100).bytes["kv_read"]
    assert long_.memory["kv_cache"] == decode(model, 1, 4100).memory["kv_cache"]


def test_window_at_32k_is_near_4k_bound(model):
    device = load_device(M4)
    t = lambda w: physical_bound(w, device).time_s  # noqa: E731
    assert t(decode(model, 1, 32768, window(4096))) == pytest.approx(t(decode(model, 1, 4096)), rel=1e-3)


def test_retrieval_reads_less_than_it_stores(model):
    p = retrieval(k=512, page=16)
    w, a = decode(model, 1, 32768, p), decode(model, 1, 32768)
    assert p.adaptive
    assert w.memory["kv_cache"] > a.memory["kv_cache"]
    # keys: 2 summary vectors per 16-token page + 512; values: 512
    assert w.bytes["kv_read"] == (32768 // 16 * 2 + 512 + 512) * model.kv_bytes_per_token // 2


def test_budget(model):
    w = decode(model, 1, 9000, budget(4500))
    assert w.bytes["kv_read"] == decode(model, 1, 4500).bytes["kv_read"]


def test_subgen_theory_constants_are_vacuous_below_32k(model):
    # eps = 0.1: s = 100 * 128 * ln n > n for every n <= 32k, so the provable
    # version keeps every value and is never cheaper than exact attention.
    for n in (1024, 8192, 32768):
        k, v = subgen_vectors(n, m=16, delta_r=0.5, eps=0.1, head_dim=128)
        assert k == v == n
    # It becomes sublinear only at astronomically long contexts (eps = 0.5).
    k, v = subgen_vectors(10**7, m=64, delta_r=0.5, eps=0.5, head_dim=128)
    assert v < 10**7 * 1e-3
    w = decode(model, 1, 32768, subgen(lambda n: 16, 0.5, 0.1, 128))
    assert w.bytes == decode(model, 1, 32768).bytes


def test_parse_policy():
    assert parse_policy(None).name == "full"
    assert parse_policy("window:4096:4").name == "window(4096,sink=4)"
    assert parse_policy("budget:100:50", bits=4).bits == 4
    with pytest.raises(ValueError):
        parse_policy("nope")


def test_storage_plans_qwen3(model):
    p = storage_plans(model, 32768)
    assert p["hidden"]["bytes"] == p["kv"]["bytes"] / 2  # hidden 1024 vs K+V 2048
    # 2 * 1024 * 2048 FLOPs per token per layer
    assert p["hidden"]["flops"] == 32768 * 28 * 2 * 1024 * 2048


def test_kv_cache_plan_wins_on_m4(model):
    device = load_device(M4)
    peak = device.engine("gpu").peak_flops("bf16").effective
    bw = device.bandwidth("gpu")
    for ctx in (512, 4096, 32768):
        w = decode(model, 1, ctx)
        base_bytes = w.total_bytes - w.bytes["kv_read"]
        r = best_storage_plan(model, ctx, peak, bw, w.total_flops, base_bytes)
        assert r["best"] == "kv"
    # FLOPs added per byte saved are ~60x the machine balance.
    assert r["hidden_flop_per_byte_saved"] == pytest.approx(2048)
    assert r["hidden_flop_per_byte_saved"] / r["balance_flop_per_byte"] > 50
