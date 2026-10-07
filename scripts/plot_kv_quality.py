"""Decode-time bound vs output quality of KV policies (design notes 6.14).

x: the estimator's physical decode bound at context P with the policy's
estimator counterpart (kvpolicy.py). y: p99 over positions and prompts of
KL(p_exact || p_policy) (top row) and the mean NLL increase on the true
continuation (bottom row), from benchmarks/kv_quality.py. Gray line: the
BF16-vs-FP32 noise floor (p99 KL). It is a numerical reference, not a quality-equivalence threshold. Time
coordinates are model references, not measured sparse/quantized latency.

  results/figures/kv_quality.png
  results/figures/kv_quality.json

    python scripts/plot_kv_quality.py [--raw results/raw/kv_quality.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from roofline import decode, load_device, load_model, physical_bound  # noqa: E402
from roofline.kvpolicy import budget, full, retrieval, window  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
# Fixed categorical order (validated: dataviz validate_palette.js, light); markers carry identity too.
FAMILIES = [("bits", "per-token quant", "#2a78d6", "o"), ("kivi", "KIVI quant", "#eb6834", "s"),
            ("window", "sink + window", "#1baf7a", "^"), ("snapkv", "SnapKV budget", "#eda100", "D"),
            ("kcenter", "k-center budget", "#e87ba4", "v"), ("quest", "Quest retrieval", "#008300", "P")]


def estimator_policy(name: str, sink: int, page: int):
    kind, _, arg = name.partition(":")
    n = int(arg) if arg else 0
    if kind == "full":
        return None
    if kind in ("bits", "kivi"):
        return full(bits=n + 0.5)  # 16-bit scale and bias per 64 elements
    if kind == "window":
        return window(n, sink=sink)
    if kind in ("snapkv", "kcenter"):
        return budget(n)
    if kind == "quest":
        return retrieval(n, page=page)
    raise ValueError(name)


def summarize(raw: dict) -> dict:
    model = load_model(ROOT / "configs/models/qwen3_0.6b.yaml")
    device = load_device(ROOT / "configs/hardware/m4_mba_24g.yaml")
    rows = {}
    for r in raw["runs"]:
        rows.setdefault((r["P"], r["policy"]), []).append(r)
    out = {"model": raw["model"], "T": raw["T"], "points": [], "floor": {}, "interpretation": "calibrated_reference_vs_exploratory_quality",
           "decode_time_measured": False, "quality_equivalence_established": False,
           "limitations": ["Overlapping prompts; token positions are not independent samples.",
                           "No confidence intervals or task-level quality tolerance.",
                           "Teacher forcing; continuation KV remains exact.",
                           "Quest query-head selection differs from the KV-head traffic model."]}
    for (P, name), rs in sorted(rows.items()):
        kl = np.concatenate([r["kl"] for r in rs])
        kv = estimator_policy(name, raw["sink"], raw["page"])
        t = physical_bound(decode(model, 1, P, kv), device).time_s
        out["points"].append({
            "P": P, "policy": name, "prompts": len(rs), "bound_ms": t * 1e3,
            "kl_p50": float(np.median(kl)), "kl_p99": float(np.percentile(kl, 99)),
            "top1": float(np.mean([r["top1"] for r in rs])),
            "dnll": float(np.mean([r["dnll"] for r in rs])),
        })
    for P in sorted({f["P"] for f in raw["floor"]}):
        fs = [f for f in raw["floor"] if f["P"] == P]
        kl = np.concatenate([f["kl"] for f in fs])
        out["floor"][str(P)] = {"kl_p50": float(np.median(kl)), "kl_p99": float(np.percentile(kl, 99)),
                                "top1": float(np.mean([f["top1"] for f in fs])), "prompts": len(fs)}
    return out


def plot(s: dict, path: Path) -> None:
    lengths = sorted({p["P"] for p in s["points"]})
    fig, axes = plt.subplots(2, len(lengths), figsize=(4.2 * len(lengths), 7.8), squeeze=False,
                             facecolor=SURFACE)
    for col, P in enumerate(lengths):
        pts = [p for p in s["points"] if p["P"] == P]
        exact = next(p for p in pts if p["policy"] == "full")
        for row, (key, label) in enumerate([("kl_p99", "p99 KL vs exact (nats)"),
                                            ("dnll", "mean NLL increase (nats)")]):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            for fam, name, color, marker in FAMILIES:
                fp = sorted((p for p in pts if p["policy"].split(":")[0] == fam), key=lambda p: p["bound_ms"])
                if not fp:
                    continue
                x = [p["bound_ms"] for p in fp]
                y = [max(p[key], 1e-5) if key == "kl_p99" else p[key] for p in fp]
                ax.plot(x, y, color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=2, label=name, zorder=3)
                if fam in ("kivi", "quest"):  # selective direct labels
                    for p, xi, yi in zip(fp, x, y):
                        ax.annotate(p["policy"].split(":")[1], (xi, yi), xytext=(5, 4),
                                    textcoords="offset points", fontsize=8, color=INK2)
            ax.axvline(exact["bound_ms"], color=MUTED, lw=1, ls="--", zorder=1)
            ax.annotate("exact", (exact["bound_ms"], 1), xycoords=("data", "axes fraction"), xytext=(-4, -12),
                        textcoords="offset points", ha="right", fontsize=8, color=MUTED)
            floor = s["floor"].get(str(P))
            if key == "kl_p99" and floor:
                ax.axhline(floor["kl_p99"], color=MUTED, lw=1, zorder=1)
                ax.annotate("BF16 vs FP32 noise", (0, floor["kl_p99"]), xycoords=("axes fraction", "data"),
                            xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
            if key == "kl_p99":
                ax.set_yscale("log")
            else:  # can be slightly negative (noise)
                ax.set_yscale("symlog", linthresh=1e-2)
                ax.axhline(0, color=MUTED, lw=1, zorder=1)
            ax.grid(True, color=GRID, lw=0.8, zorder=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(colors=INK2, labelsize=9)
            if row == 0:
                ax.set_title(f"context {P // 1024}k", color=INK, fontsize=11)
            if row == 1:
                ax.set_xlabel("modeled decode reference, M4 (ms; unmeasured)", color=INK2, fontsize=9)
            if col == 0:
                ax.set_ylabel(label, color=INK2, fontsize=9)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False,
               fontsize=9, labelcolor=INK2)
    fig.suptitle(f"Qwen3-0.6B KV policies: modeled time vs exploratory quality (T = {s['T']} teacher-forced tokens)",
                 color=INK, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results/raw/kv_quality.json"))
    ap.add_argument("--out", default=str(ROOT / "results/figures/kv_quality"))
    a = ap.parse_args()
    s = summarize(json.loads(Path(a.raw).read_text()))
    Path(a.out + ".json").write_text(json.dumps(s, indent=1))
    plot(s, Path(a.out + ".png"))
    print(f"{'P':>6} {'policy':14s} {'bound':>8} {'kl p50':>9} {'kl p99':>9} {'top1':>6} {'dnll':>8}")
    for p in s["points"]:
        print(f"{p['P']:6d} {p['policy']:14s} {p['bound_ms']:7.1f}ms {p['kl_p50']:9.2e} {p['kl_p99']:9.2e} "
              f"{p['top1']:6.3f} {p['dnll']:+8.4f}")
    for P, f in s["floor"].items():
        print(f"{P:>6} {'fp32 floor':14s} {'':8s} {f['kl_p50']:9.2e} {f['kl_p99']:9.2e} {f['top1']:6.3f}")


if __name__ == "__main__":
    main()
