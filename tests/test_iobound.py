import math
from dataclasses import replace
from pathlib import Path

import pytest

from roofline import decode, load_device, load_model, physical_bound, prefill
from roofline.device import Level, Quantity
from roofline.graph import decode_graph, prefill_graph
from roofline.iobound import AttentionOp, GemmOp, attention_terms, boundaries, gemm_terms, io_bound, operators
from roofline.toy import attention_dag, flash_attention_io, matmul_dag, optimal_io, tiled_gemm_io

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "configs/models/qwen3_0.6b.yaml"
M4 = ROOT / "configs/hardware/m4_mba_24g.yaml"


@pytest.fixture
def model():
    return load_model(QWEN)


@pytest.fixture
def device():
    return load_device(M4)


def q(v):
    return Quantity(value=v, source="estimate", confidence="low")


# --- boundaries ---------------------------------------------------------------


def test_boundaries_are_cumulative(device):
    core, chip = boundaries(device)
    per_core = device.hierarchy[0].capacity.effective
    assert (core.processors, core.fast_bytes) == (10, per_core)
    # Savage: the DRAM boundary sees all on-chip storage, private and shared.
    assert chip.processors == 1 and chip.dram
    assert chip.fast_bytes == device.on_chip.effective + 10 * per_core
    assert chip.bandwidth == device.bandwidth("gpu")


# --- whole-graph structure ----------------------------------------------------


def test_operators_cover_graph_flops(model):
    g = prefill_graph(model, 1, 256)
    gemms, attns, other = operators(g)
    assert sum(x.flops for x in gemms) + sum(a.flops for a in attns) == g.total_flops
    assert len(gemms) == 4 * model.num_layers + 2  # qkv, o, gate_up, down per layer + lm_head
    qkv = gemms[0]
    assert (qkv.n, qkv.persist) == (model.q_dim + 2 * model.kv_dim, 256 * 2 * model.kv_dim)
    # Norm weights and token ids in, sampled token out.
    norms = (model.num_layers * model.layer_norm_params + model.hidden_size) * 2
    assert other == norms + 256 * 4 + 4


def test_fused_with_tiny_fast_memory(model, device):
    for w, g in ((decode(model, 1, 4096), decode_graph(model, 1, 4096)),
                 (prefill(model, 1, 512), prefill_graph(model, 1, 512))):
        tiny = replace(device, hierarchy=tuple(replace(l, capacity=q(1.0)) for l in device.hierarchy))
        b = io_bound(g, tiny, include_experimental_fused=True)
        for lv in b.levels:
            S = lv.boundary.processors * lv.boundary.fast_bytes
            assert lv.fused_terms["compulsory"] == pytest.approx(w.total_bytes - S)
            # Almost no fast memory: in prefill (many rows share each weight) the pebble
            # term forces far more traffic than compulsory; decode reads each weight once anyway.
            if w.phase == "prefill":
                assert lv.fused_bytes > 10 * w.total_bytes
        assert b.time_s("fused") >= physical_bound(w, device).time_s * (1 - 1e-6)


@pytest.mark.parametrize("phase,n", [("decode", 512), ("decode", 32768), ("prefill", 64), ("prefill", 32768)])
def test_operator_at_least_fused_and_never_beats_compute(model, device, phase, n):
    g = decode_graph(model, 1, n) if phase == "decode" else prefill_graph(model, 1, n)
    b = io_bound(g, device)
    for lv in b.levels:
        assert lv.operator_bytes >= lv.fused_bytes > 0
    assert b.time_s("operator") >= b.time_s("fused") >= b.flops / b.peak_flops


def test_bigger_fast_memory_weakens_bounds(model, device):
    g = prefill_graph(model, 1, 4096)
    prev = None
    for kib in (64, 256, 1024, 4096):
        d = replace(device, hierarchy=(replace(device.hierarchy[0], capacity=q(kib * 1024.0)),
                                       *device.hierarchy[1:]))
        core = io_bound(g, d).levels[0]
        if prev is not None:
            assert core.operator_bytes <= prev
        prev = core.operator_bytes


# --- pebble terms vs explicit schedules (upper bounds) -------------------------


@pytest.mark.parametrize("m,n,k", [(4096, 4096, 1024), (32768, 1024, 3072), (512, 151936, 1024), (8, 1024, 1024)])
@pytest.mark.parametrize("S", [2**13, 2**16, 2**18, 6.3e6])
@pytest.mark.parametrize("P", [1, 10])
def test_gemm_lower_bound_below_tiled_schedule(m, n, k, S, P):
    g = GemmOp("x", "x", m, n, k, 2, 0)
    lower = max(gemm_terms(g, S, P).values())
    assert lower <= tiled_gemm_io(m, n, k, S, P)


def test_gemm_pebble_term_is_nearly_tight():
    # Smith et al.: 2mnk/sqrt(S) is attained up to lower-order terms.
    m = n = k = 8192
    S = 2**16
    lower = gemm_terms(GemmOp("x", "x", m, n, k, 2, 0), S, 1)["pebble"]
    assert 1 <= tiled_gemm_io(m, n, k, S) / lower < 1.05


@pytest.mark.parametrize("n,L", [(4096, 4096), (32768, 32768), (2, 32768), (512, 512)])
@pytest.mark.parametrize("S", [2**14, 2**16, 2**18, 6.3e6])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_lower_bound_below_flash_schedule(n, L, S, causal):
    d = 128
    scores = n * (n + 1) // 2 if causal and n == L else n * L
    a = AttentionOp("x", scores, d, n * d, 2 * L * d, 0, 2)
    lower = max(attention_terms(a, S, 1).values())
    io, _ = flash_attention_io(n, L, d, S, causal=causal and n == L)
    assert lower <= io


def test_attention_pebble_term_within_constant_of_flash():
    # Large n, L: FlashAttention moves ~4 n L d^2 / S, the bound ~n L d^2 / (S + 4d^2):
    # ratio ~4 (1 + 4d^2/S) = 5 here.
    n = L = 65536
    d, S = 128, 2**18
    lower = attention_terms(AttentionOp("x", n * L, d, n * d, 2 * L * d, 0, 2), S, 1)["pebble"]
    io, _ = flash_attention_io(n, L, d, S)
    assert 1 < io / lower < 8


# --- exact red-blue pebbling on toy graphs -------------------------------------


def test_exact_solver_on_known_graphs():
    # One FMA chain c = a0 b0 + a1 b1: 4 inputs, 1 output, 4 red pebbles suffice.
    assert optimal_io(matmul_dag(1, 1, 2), S=4) == 5
    # The second FMA needs its three parents and itself red: impossible with 3.
    with pytest.raises(RuntimeError):
        optimal_io(matmul_dag(1, 1, 2), S=3)
    # 2x2 outer product with 3 pebbles: a0 b0, a0 b1, a1 b1, then b0 again:
    # 4 inputs + 4 outputs + 1 reload.
    assert optimal_io(matmul_dag(2, 2, 1), S=3) == 9


@pytest.mark.parametrize("dag,S", [
    (matmul_dag(2, 2, 1), 3), (matmul_dag(2, 2, 1), 4), (matmul_dag(1, 2, 2), 4),
    (matmul_dag(2, 1, 2), 4), (attention_dag(2, 2, 1), 4), (attention_dag(2, 2, 1), 5),
])
def test_bounds_never_exceed_exact_optimum(dag, S):
    opt = optimal_io(dag, S)
    # Cold start: every input loaded, every output stored.
    assert len(dag.inputs) + len(dag.outputs) <= opt
    names = dag.names
    if names[0].startswith("a"):
        m = 1 + max(int(x[1]) for x in names if x[0] == "a")
        k = 1 + max(int(x[2]) for x in names if x[0] == "a")
        n = 1 + max(int(x[2]) for x in names if x[0] == "b")
        assert max(gemm_terms(GemmOp("x", "x", m, n, k, 1, 0), S, 1).values()) <= opt
    else:
        n = 1 + max(int(x[1]) for x in names if x[0] == "q")
        L = 1 + max(int(x[1]) for x in names if x[0] == "k")
        a = AttentionOp("x", n * L, 1, n, 2 * L, 0, 1)
        assert attention_terms(a, S, 1)["pebble"] <= opt


def test_recomputation_can_beat_storing():
    # Two outputs share an expensive-to-reload input pair; with S = 3 the
    # solver may recompute instead of storing. Just check it terminates and
    # respects the compulsory count.
    dag = matmul_dag(2, 1, 1)
    assert optimal_io(dag, S=3) >= len(dag.inputs) + len(dag.outputs)


def test_traffic_at_matches_device_boundary(model, device):
    from roofline.iobound import traffic_at

    g = prefill_graph(model, 1, 4096)
    chip = io_bound(g, device).levels[-1]
    fused, operator = traffic_at(g, chip.boundary.fast_bytes)
    assert (fused, operator) == pytest.approx((chip.fused_bytes, chip.operator_bytes))
    # Q_min(S) never grows with S; the fused bound retains compulsory output writes even when inputs fit.
    qs = [traffic_at(g, s)[1] for s in (2**12, 2**16, 2**20, 2**24)]
    assert all(a >= b for a, b in zip(qs, qs[1:]))
    assert traffic_at(g, 1e12)[0] == sum(t.bytes for t in g.outputs)


# --- fused pebble bound (design notes 6.12) ------------------------------------

from roofline.iobound import (  # noqa: E402
    FusedContraction, fused_contractions, fused_pebble_elements, recompute_capacity,
)


@pytest.mark.parametrize("Y", [1e3, 3e4, 2e5, 1e6, 3e6])
def test_recompute_capacity_is_the_exact_maximum(Y):
    # Small instance: brute force over integer rows must not beat the analytic max.
    c = FusedContraction("x", rows=700, K=64, n=64, count=1, omega=300, row_cost=((40, 128), (20, 64)))
    def f(r):
        cost = sum(cols * min(r, cap) for cols, cap in c.row_cost)
        return r * min(c.K, max(0.0, (Y - cost) / c.omega))
    brute = max(f(r) for r in range(1, c.rows + 1))
    assert recompute_capacity(c, Y) >= brute - 1e-9
    assert recompute_capacity(c, Y) <= brute * 1.01 + 1  # continuous relaxation, barely above


def test_fused_contractions_of_qwen3(model):
    cs = {c.name: c for c in fused_contractions(prefill_graph(model, 1, 1024))}
    assert cs["qkv"].count == model.num_layers - 2  # layer 0 reads the embedding table directly
    assert cs["qkv"].omega == model.intermediate_size + model.q_dim
    assert cs["down"].K == model.intermediate_size and cs["gate_up"].n == 2 * model.intermediate_size


@pytest.mark.parametrize("L", [4096, 32768])
@pytest.mark.parametrize("S", [2**12, 2**15, 2**18, 2**21])
@pytest.mark.parametrize("P", [1, 10])
def test_fused_bound_below_a_real_schedule(model, L, S, P):
    # Operator-by-operator square-tiled GEMMs are one valid implementation:
    # their traffic on the counted contractions bounds the optimum from above.
    cs = fused_contractions(prefill_graph(model, 1, L))
    schedule = sum(c.count * tiled_gemm_io(c.rows, c.n, c.K, S, P) for c in cs)
    assert fused_pebble_elements(cs, S, P) <= schedule


def test_fused_bound_shrinks_with_capacity_negligible_for_decode(model):
    cs = fused_contractions(prefill_graph(model, 1, 16384))
    qs = [fused_pebble_elements(cs, s) for s in (2**10, 2**14, 2**18, 2**22)]
    assert all(a >= b for a, b in zip(qs, qs[1:])) and qs[0] > 0
    # Decode (one row): the bound is positive but negligible next to the 1.2 GB of weights.
    assert fused_pebble_elements(fused_contractions(decode_graph(model, 1, 4096)), 2**18) * 2 < 1e-4 * 1.2e9


def test_fused_bound_beats_compulsory_only_for_long_prompts_at_core_level(model, device):
    for L, expect in ((4096, False), (32768, True)):
        core = io_bound(prefill_graph(model, 1, L), device, include_experimental_fused=True).levels[0]
        assert bool(core.fused_terms["pebble"] > core.fused_terms["compulsory"]) is expect
        assert core.operator_bytes >= core.fused_bytes
