"""Create an isolated workspace for one navigation-experiment agent run.

    python agent-test/nav/make_workspace.py --arm P --run P1 --hours 2
    python agent-test/nav/make_workspace.py --arm B --run B1 --hours 2

Arm P (profiling only): starting engine, tools, HARDWARE.md, BASELINE.md.
Arm B (bound): the same plus BOUND.md and roofline_tool/ (reference lines, gap decomposition, what-if).
The workspace lives outside this repository (default ~/nav-runs/<run>) and is a fresh git repo,
so the agent sees neither this project's docs, code, memory nor skills.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KIT = HERE / "kit"
PY = ROOT / ".venv-vllm-metal/bin/python"

MATERIALS = {
    "P": "- `docs/HARDWARE.md` — calibrated hardware numbers for this machine.\n"
         "- `docs/BASELINE.md` — stock mlx-lm timings on the scenarios.",
    "B": "- `docs/HARDWARE.md` — calibrated hardware numbers for this machine.\n"
         "- `docs/BASELINE.md` — stock mlx-lm timings on the scenarios.\n"
         "- `docs/BOUND.md` — resource reference lines per scenario (how fast any level-A implementation could be,\n"
         "  given the calibrated rates), mlx-lm's gap to them, and where mlx-lm loses time.\n"
         "- `roofline_tool/` — the calculator behind BOUND.md, for what-if questions (see BOUND.md).",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["P", "B"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--hours", type=float, default=2)
    ap.add_argument("--root", default=str(Path.home() / "nav-runs"))
    ap.add_argument("--start", help="ISO time the run will start (default: now)")
    a = ap.parse_args()
    ws = Path(a.root) / a.run
    if ws.exists():
        sys.exit(f"{ws} exists; refusing to overwrite")
    shutil.copytree(KIT / "common", ws, ignore=shutil.ignore_patterns("__pycache__"))
    if a.arm == "B":
        shutil.copytree(KIT / "bound/docs", ws / "docs", dirs_exist_ok=True)
        tool = ws / "roofline_tool"
        shutil.copytree(ROOT / "src/roofline", tool / "roofline", ignore=shutil.ignore_patterns("__pycache__"))
        (tool / "configs/models").mkdir(parents=True)
        (tool / "configs/hardware").mkdir(parents=True)
        shutil.copy(ROOT / "configs/models/qwen3_0.6b.yaml", tool / "configs/models/")
        for f in (ROOT / "configs/hardware").glob("m4_mba_24g*"):
            shutil.copy(f, tool / "configs/hardware/")
    start = dt.datetime.fromisoformat(a.start) if a.start else dt.datetime.now()
    deadline = start + dt.timedelta(hours=a.hours)
    mlx_v = subprocess.check_output([str(PY), "-c", "import mlx.core as m; print(m.__version__)"], text=True).strip()
    lm_v = subprocess.check_output([str(PY), "-c", "import mlx_lm; print(mlx_lm.__version__)"], text=True).strip()
    task = (KIT / "TASK_TEMPLATE.md").read_text()
    task = (task.replace("{MATERIALS}", MATERIALS[a.arm]).replace("{HOURS}", f"{a.hours:g}")
            .replace("{DEADLINE}", deadline.strftime("%Y-%m-%d %H:%M"))
            .replace("{PYTHON}", str(PY)).replace("{MLX_VERSION}", mlx_v).replace("{MLXLM_VERSION}", lm_v))
    (ws / "TASK.md").write_text(task)
    (ws / ".gitignore").write_text("__pycache__/\nbench_last.json\n")
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    # the agent runs with its own empty HOME: the identity lives in the repository's config
    subprocess.run(["git", "config", "user.name", "agent"], cwd=ws, check=True)
    subprocess.run(["git", "config", "user.email", "agent@local"], cwd=ws, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "commit", "-qm", "starting point"],
                   cwd=ws, check=True)
    (HERE / "runs").mkdir(exist_ok=True)   # arm label kept in the repo, not next to the workspace
    (HERE / "runs" / f"{a.run}.meta.json").write_text(json.dumps(dict(
        arm=a.arm, run=a.run, hours=a.hours, start=start.isoformat(), deadline=deadline.isoformat(),
        workspace=str(ws)), indent=1))
    print(ws)


if __name__ == "__main__":
    main()
