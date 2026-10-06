"""Collect kernel-level calibration into one tracked file (plan items 11-13).

Reads the newest results/raw/{kernel_chain_fit,compute_shapes-*}.json and
writes configs/hardware/m4_mba_24g.kernels.json: the chain-fit constants and
the P(shape) tables that src/roofline/gap.py uses. results/raw is not
tracked; this file is, with the source file names recorded.

    python scripts/fit_kernel_chain.py && python scripts/calibrate.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path


def newest(pattern: str) -> str:
    return sorted(glob.glob(pattern))[-1]


def main() -> None:
    fit = json.loads(Path("results/raw/kernel_chain_fit.json").read_text())
    shapes_path = newest("results/raw/compute_shapes-*.json")
    shapes = json.loads(Path(shapes_path).read_text())
    r = shapes["results"]
    out = {
        "note": "Kernel-level calibration for the per-plan cost model (design notes 5.3, 6.9). Burst values.",
        "sources": {"kernel_chain": fit["source"], "compute_shapes": shapes_path,
                    "timestamp": shapes["timestamp"], "mlx_version": shapes["mlx_version"]},
        "chain": {k: fit[k] for k in ("e_s", "c_commit_s", "t0_s", "B_eff", "c_sync_s")},
        "independent_over_chain": fit["independent_over_chain"],
        # gemm[name] = [[M, seconds], ...] for mx.matmul at Qwen3-0.6B shapes
        "gemm": {name: [[row["M"], row["time_s"]] for row in rows] for name, rows in r["gemm"].items()},
        "gemm_shapes": {name: [rows[0]["K"], rows[0]["N"]] for name, rows in r["gemm"].items()},
        # sdpa = [[Lq, Lk, seconds], ...]; Lq > 1 is causal (queries at the end of the keys)
        "sdpa": [[row["Lq"], row["Lk"], row["time_s"]] for row in r["sdpa"]],
        # small[op] = [[M, seconds per op in a dependent chain], ...]
        "small": {op: [[row["M"], row[op]["chain_s"]] for row in r["small"]]
                  for op in ("rms_norm", "rope", "add", "swiglu", "slice_update")},
        "exp_per_s": r["exp"]["exp_per_s"],
    }
    path = Path("configs/hardware/m4_mba_24g.kernels.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
