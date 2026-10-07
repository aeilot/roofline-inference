"""Judge generated-sequence divergences against FP32 (PLAN.md section 7).

For every main-phase group (session, regime, block, scenario), compare each
engine's tokens with E0's. At the first differing position j, run the FP32
model on prompt + E0 tokens[:j] and record the FP32 logit gap between the two
chosen tokens. A divergence is acceptable only if the gap is below E0's p99
top-2 margin noise from the correctness gate. Results are cached.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompts  # noqa: E402


def load_runs(results):
    runs = []
    for f in sorted((results / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        if "unit" in r and not r.get("excluded"):
            runs.append(r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    results = Path(a.results)
    cache_file = results / "divergences.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    gates = [json.loads(f.read_text()) for f in sorted(results.glob("gate-s*.json"))]
    threshold = min(g["e0_margin_err_p99"] for g in gates) if gates else None

    groups = {}
    for r in load_runs(results):
        u = r["unit"]
        if u["phase"] != "main":
            continue
        groups.setdefault((u["session"], u["regime"], u["block"], u["scenario"]), {})[u["engine"]] = r

    todo = []
    for key, g in groups.items():
        if "E0" not in g:
            continue
        base = g["E0"]["tokens"]
        for eng, r in g.items():
            if eng == "E0":
                continue
            j = next((i for i, (x, y) in enumerate(zip(base, r["tokens"])) if x != y), None)
            if j is None:
                continue
            u = r["unit"]
            cid = f"{u['kind']}-{u['length']}-{u['variant']}-{j}-{base[j]}-{r['tokens'][j]}"
            todo.append((cid, key, eng, u, base, j, r["tokens"][j]))

    model32 = None
    for cid, key, eng, u, base, j, other in todo:
        if cid in cache:
            continue
        if model32 is None:
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
            model32, _ = load("Qwen/Qwen3-0.6B")
            model32.set_dtype(mx.float32)
        seq = mx.array(prompts.make_prompt(u["kind"], u["length"], u["variant"]) + base[:j])[None]
        c = make_prompt_cache(model32)
        for s in range(0, seq.shape[1] - 1, 2048):
            model32(seq[:, s:min(s + 2048, seq.shape[1] - 1)], cache=c)
            mx.eval([x.state for x in c])
        logits = model32(seq[:, -1:], cache=c)[0, -1]
        gap = abs(logits[base[j]] - logits[other]).item()
        cache[cid] = dict(position=j, e0_token=base[j], other_token=other, fp32_gap=gap)
        cache_file.write_text(json.dumps(cache, indent=1))

    records = []
    for cid, key, eng, u, base, j, other in todo:
        c = cache[cid]
        records.append(dict(session=key[0], regime=key[1], block=key[2], scenario=key[3], engine=eng, **c,
                            threshold=threshold, near_tie=threshold is not None and c["fp32_gap"] < threshold))
    (results / "divergence_records.json").write_text(json.dumps(records, indent=1))
    bad = [r for r in records if not r["near_tie"]]
    print(f"{len(records)} divergences, {len(bad)} not FP32 near-ties")


if __name__ == "__main__":
    main()
