"""Blind the pilot's final engines for evaluation.

Writes runs/candidates.json ({blinded name: engine dir}, read by the v3 harness via NAV_CANDIDATES)
and runs/blind_key.json ({blinded name: run id}, read only by nav_analyze.py after timing).

    python agent-test/nav/make_eval.py P1 B1 B2 P2
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
runs = sys.argv[1:]
missing = [r for r in runs if not (HERE / "runs" / r / "engine" / "engine.py").exists()]
if missing:
    sys.exit(f"no engine snapshot for {missing}")
order = runs[:]
random.Random(20261008).shuffle(order)
names = {f"C-{i + 1}": r for i, r in enumerate(order)}
(HERE / "runs" / "candidates.json").write_text(json.dumps(
    {n: str(HERE / "runs" / r / "engine") for n, r in names.items()}, indent=1))
(HERE / "runs" / "blind_key.json").write_text(json.dumps(names, indent=1))
print("candidates:", ", ".join(names))
