"""Conservative generated-token screening for the official vLLM comparator.

Exact matches or FP32 near-ties on four disjoint fixtures are necessary,
not proof of quality noninferiority. Main-run divergences are additionally judged against FP32.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--probe-ref", required=True)
    p.add_argument("--reference-gate", required=True)
    p.add_argument("--reuse", help="Rejudge saved fixture records without new timing")
    a = p.parse_args()
    fixtures = []
    with tempfile.TemporaryDirectory(prefix="roofline-vllm-gate-") as td:
        for kind in (() if a.reuse else ("random", "text")):
            for length in (64, 512):
                records = []
                for engine in ("E0", "V0"):
                    dest = Path(td) / f"{kind}-{length}-{engine}.json"
                    subprocess.run([sys.executable, str(HERE / "worker.py"), "--engine", engine,
                        "--kind", kind, "--length", str(length), "--variant", "2000000", "--gen", "32",
                        "--regime", "idle", "--probe-ref", a.probe_ref, "--out", str(dest)],
                        check=True, timeout=900)
                    records.append(json.loads(dest.read_text()))
                ok = (all(not r.get("excluded") for r in records)
                      and records[0]["prompt_sha256"] == records[1]["prompt_sha256"]
                      and records[0]["model_snapshot"] == records[1]["model_snapshot"]
                      and records[0].get("tokens") == records[1].get("tokens"))
                fixtures.append(dict(kind=kind, length=length, pass_exact=ok, records=records))
    if a.reuse:
        fixtures = json.loads(Path(a.reuse).read_text())["fixtures"]
    reference = json.loads(Path(a.reference_gate).read_text())
    threshold = reference["e0_margin_err_p99"]
    model32 = None
    for f in fixtures:
        a0, v0 = f["records"]
        valid = (all(not r.get("excluded") for r in (a0, v0))
                 and a0["prompt_sha256"] == v0["prompt_sha256"]
                 and a0["model_snapshot"] == v0["model_snapshot"]
                 and len(a0.get("tokens", [])) == len(v0.get("tokens", [])) == 32)
        j = next((i for i, (x, y) in enumerate(zip(a0.get("tokens", []), v0.get("tokens", []))) if x != y), None)
        f["pass"] = valid and j is None
        if valid and j is not None:
            import mlx.core as mx
            import adapters
            import prompts
            if model32 is None:
                from mlx_lm import load
                model32, _ = load("Qwen/Qwen3-0.6B")
                model32.set_dtype(mx.float32)
            logits = adapters.E0(model32).teacher_forced(
                prompts.make_prompt(f["kind"], f["length"], 2000000), a0["tokens"][:j+1])[-1]
            gap = abs(logits[a0["tokens"][j]] - logits[v0["tokens"][j]]).item()
            f["first_divergence"] = dict(position=j, fp32_gap=gap, threshold=threshold)
            f["pass"] = gap < threshold
    passed = reference["all_pass"] and all(x["pass"] for x in fixtures)
    Path(a.out).write_text(json.dumps(dict(fixtures=fixtures, criterion="exact_or_first_divergence_FP32_near_tie",
        **{"pass": passed}), indent=2))


if __name__ == "__main__":
    main()
