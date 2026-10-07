"""Regression checks for the scientific-review corrections."""
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import pytest

from roofline import load_device, load_model, physical_bound, prefill
from roofline.graph import prefill_graph
from roofline.iobound import io_bound

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "configs/models/qwen3_0.6b.yaml"
DEVICE = ROOT / "configs/hardware/m4_mba_24g.yaml"
spec = importlib.util.spec_from_file_location("corrected_analysis", ROOT / "agent-test/v3/analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


@pytest.mark.parametrize("layers,batch,length", [(1, 2, 1), (1, 2, 17), (3, 1, 512), (28, 2, 1024)])
def test_useful_graph_removes_only_unused_final_rows(layers, batch, length):
    m = replace(load_model(MODEL), num_layers=layers)
    old = prefill(m, batch, length, prune_final=False)
    new = prefill(m, batch, length)
    dead_linear = 2 * m.hidden_size * m.q_dim + 3 * m.hidden_size * m.intermediate_size
    assert old.flops["linear"] - new.flops["linear"] == 2 * dead_linear * batch * (length - 1)
    assert old.flops["attention"] - new.flops["attention"] == (
        4 * m.num_heads * m.head_dim * batch * (length * (length + 1) // 2 - length))
    graph = prefill_graph(m, batch, length)
    assert graph.total_flops == new.total_flops
    assert sum(2 * g.m * g.n * g.k * g.count for g in new.gemms) == new.flops["linear"] + new.flops["lm_head"]
    assert graph.compulsory_bytes()["kv_new"] == old.bytes["kv_write"] == new.bytes["kv_write"]
    assert sum(g.persistent_out_bytes * g.count for g in new.gemms) == new.bytes["kv_write"]
    assert prefill_graph(m, batch, length, prune_final=False).total_flops == old.total_flops
    final_q = next(o for o in graph.ops if o.name == f"L{layers - 1}.q_proj")
    final_k = next(o for o in graph.ops if o.name == f"L{layers - 1}.k_proj")
    assert final_q.shape["m"] == batch
    assert final_k.shape["m"] == batch * length


def test_resident_credit_never_removes_compulsory_writes():
    m, d = load_model(MODEL), load_device(DEVICE)
    w = prefill(m, 1, 16)
    cold = physical_bound(w, d, initial_resident_bytes=0)
    warm = physical_bound(w, d, initial_resident_bytes=1e20)
    assert cold.dram_bytes == w.total_bytes
    assert warm.dram_bytes == w.bytes["kv_write"] + w.batch * 4
    assert warm.to_dict()["hardware_ceiling_certified"] is False
    assert warm.to_dict()["computation_scope"] == "useful_product_graph"
    with pytest.raises(ValueError):
        physical_bound(w, d, initial_resident_bytes=-1)


def test_experimental_fused_proof_is_not_applied_by_default():
    g = prefill_graph(load_model(MODEL), 1, 32768)
    d = load_device(DEVICE)
    default = io_bound(g, d).levels[0]
    experimental = io_bound(g, d, include_experimental_fused=True).levels[0]
    assert default.fused_bytes == default.fused_terms["compulsory"]
    assert default.fused_terms["experimental_applied"] is False
    assert default.fused_terms["experimental_pebble"] > 0
    assert experimental.fused_bytes >= default.fused_bytes


@pytest.mark.parametrize("n", [6, 8, 12])
def test_sign_test_has_exact_control_at_null_boundary(n):
    # Sum null probability over every rejecting count, not a Monte Carlo claim.
    rejection = sum(math.comb(n, k) / 2**n for k in range(n + 1)
                    if analysis.sign_p([1.0] * k + [-1.0] * (n - k), 0) <= 0.05)
    assert rejection <= 0.05
    assert analysis.sign_p([1.0] * n, 0) == 2**-n
    assert analysis.sign_p([0.0] * n, 0) == 1.0
    assert analysis.holm([2**-6] * 4) == [0.0625] * 4


@pytest.mark.parametrize("kind", ["operator", "hierarchy"])
def test_unsupported_kv_graph_combination_fails_explicitly(kind):
    proc = subprocess.run([sys.executable, "-m", "roofline", "--model", str(MODEL),
                           "--bound", kind, "decode", "--context", "32768", "--kv-bits", "4"],
                          capture_output=True, text=True)
    assert proc.returncode == 2
    assert "KV policies are supported only" in proc.stderr


def test_large_residency_preserves_hierarchy_output_traffic():
    from roofline.iobound import traffic_at
    g = prefill_graph(load_model(MODEL), 1, 4096)
    assert traffic_at(g, 1e20)[0] == sum(t.bytes for t in g.outputs)


@pytest.mark.parametrize("gate_state", ["missing", "failed_e0"])
def test_analysis_does_not_accept_missing_or_failed_correctness(tmp_path, gate_state):
    import json
    runs = tmp_path / "runs"
    runs.mkdir()
    for session in (1, 2):
        for block in range(3):
            for engine, elapsed in (("E0", 2.0), ("E2", 1.0)):
                record = {"unit": {"phase": "main", "regime": "idle", "scenario": "ttft64",
                                   "engine": engine, "session": session, "block": block}, "ttft_s": elapsed}
                (runs / f"{session}-{block}-{engine}.json").write_text(json.dumps(record))
    (tmp_path / "divergence_records.json").write_text("[]")
    if gate_state == "failed_e0":
        (tmp_path / "gate-s1.json").write_text(json.dumps({"unit_tests": {"returncode": 0},
              "engines": {"E0": {"pass": False}, "E2": {"pass": True}}, "all_pass": False}))
    proc = subprocess.run([sys.executable, str(ROOT / "agent-test/v3/analyze.py"),
                           "--results", str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = json.loads((tmp_path / "summary-corrected.json").read_text())
    assert out["primary"]["idle"]["TTFT/ttft64"]["verdict"].startswith("作废")
    assert not (tmp_path / "report.md").exists()


def test_pairing_rejects_different_prompts_and_duplicates():
    def rec(engine, digest):
        return dict(unit=dict(phase="main", regime="idle", scenario="ttft64",
                              session=1, block=0, engine=engine, gen=8),
                    ttft_s=1.0, prompt_sha256=digest, model_snapshot="same")
    with pytest.raises(ValueError, match="Mismatched"):
        analysis.paired([rec("E0", "a"), rec("E2", "b")],
                        "main", "idle", "E0", "E2", "ttft_s", "ttft64")
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.paired([rec("E0", "a"), rec("E0", "a")],
                        "main", "idle", "E0", "E2", "ttft_s", "ttft64")


def test_confirm_schedule_is_complete_and_gates_every_engine():
    spec = importlib.util.spec_from_file_location("v3_runner", ROOT / "agent-test/v3/run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    from collections import Counter
    all_units = runner.schedule(1, False, "confirm") + runner.schedule(2, False, "confirm")
    counts = Counter((u["scenario"], u["engine"]) for u in all_units)
    assert len(all_units) == 648
    assert set(counts.values()) == {24}
    assert all(u["gate"] and u["warm_full"] for u in all_units)
    for session in (1, 2):
        kinds = Counter(u["kind"] for u in all_units if u["session"] == session)
        assert kinds["random"] == kinds["text"]
