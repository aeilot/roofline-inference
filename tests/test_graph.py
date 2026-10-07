from dataclasses import replace
from pathlib import Path

import pytest

from roofline import decode, load_model, prefill
from roofline.graph import TOKEN_BYTES, Graph, Op, Tensor, decode_graph, prefill_graph

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "configs/models/qwen3_0.6b.yaml"

CASES = [("decode", 1, 0), ("decode", 1, 4096), ("decode", 4, 512), ("prefill", 1, 16), ("prefill", 2, 1024)]


@pytest.fixture
def model():
    return load_model(QWEN)


def build(model, phase, batch, n):
    if phase == "decode":
        return decode_graph(model, batch, n), decode(model, batch, n)
    return prefill_graph(model, batch, n), prefill(model, batch, n)


@pytest.mark.parametrize("phase,batch,n", CASES)
def test_flops_match_workload(model, phase, batch, n):
    g, w = build(model, phase, batch, n)
    by = g.flops_by_group()
    assert by["qkv"] + by["o_proj"] + by["mlp"] == w.flops["linear"]
    assert by["attention"] == w.flops["attention"]
    assert by["lm_head"] == w.flops["lm_head"]
    assert g.total_flops == w.total_flops
    assert sum(op.shape["scores"] for op in g.ops if op.kind == "attention") == w.attention.total_scores


@pytest.mark.parametrize("phase,batch,n", CASES)
def test_compulsory_bytes_vs_workload(model, phase, batch, n):
    g, w = build(model, phase, batch, n)
    c = g.compulsory_bytes()
    assert c["weight"] == model.weight_bytes  # tied: every parameter once
    assert c.get("kv_cache", 0) == w.bytes["kv_read"]
    assert c["kv_new"] == w.bytes["kv_write"]
    assert c["token"] == g.tokens * TOKEN_BYTES
    assert c["sampled"] == batch * TOKEN_BYTES
    assert w.bytes["embedding_gather"] == 0  # tied: rows are inside the LM-head read
    assert w.bytes["token_io"] == (g.tokens + batch) * TOKEN_BYTES
    assert g.total_compulsory_bytes == w.total_bytes


def test_untied_embedding_counts_gathered_rows(model):
    m = replace(model, tie_word_embeddings=False)
    g, w = decode_graph(m, 2, 128), decode(m, 2, 128)
    assert g.compulsory_bytes()["weight"] == w.bytes["weights"] + w.bytes["embedding_gather"]


def test_graph_structure(model):
    g = decode_graph(model, 1, 64)
    per_layer = [op for op in g.ops if op.layer == 0]
    assert [op.name for op in per_layer if op.kind == "contraction"] == [
        "L0.q_proj", "L0.k_proj", "L0.v_proj", "L0.o_proj", "L0.gate_proj", "L0.up_proj", "L0.down_proj"]
    assert sum(op.kind == "attention" for op in g.ops) == model.num_layers
    assert {t.role for t in g.inputs} == {"weight", "kv_cache", "token"}
    # Prefill has no cache to read.
    assert "kv_cache" not in {t.role for t in prefill_graph(model, 1, 32).inputs}


def test_validate_rejects_use_before_produce():
    t = {"a": Tensor("a", (1,), 2, "activation"), "b": Tensor("b", (1,), 2, "activation")}
    g = Graph("decode", 1, 1, 0, t, (Op("f", "elementwise", "other", ("a",), ("b",)),))
    with pytest.raises(ValueError):
        g.validate()
