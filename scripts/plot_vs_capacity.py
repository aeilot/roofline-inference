"""Performance bound vs fast-memory capacity S (design notes 6.10).

Arithmetic intensity is not an input here: for each step's computation graph,
Q_min(S) (src/roofline/iobound.py) gives the least traffic across a boundary
with S bytes of fast memory, so

    AI_max(S)   = FLOPs / Q_min(S)
    perf(S)    <= FLOPs / max(FLOPs / P_peak, Q_min(S) / B)

Two-level model: fast memory of S bytes (one processor) in front of DRAM at
the measured bandwidth. Solid lines: fused bound (any level-A implementation;
only compulsory traffic is proven). Dashed: operator-boundary bound (adds the
pebble terms). Vertical lines: capacities of the M4's on-chip levels.

  results/figures/perf_vs_capacity.png
  results/figures/perf_vs_capacity.json

    python scripts/plot_vs_capacity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from roofline import load_device, load_model  # noqa: E402
from roofline.graph import decode_graph, prefill_graph  # noqa: E402
from roofline.iobound import traffic_at  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SERIES = [("decode ctx 4k", "#2a78d6"), ("prefill L 512", "#eb6834"),
          ("prefill L 4k", "#1baf7a"), ("prefill L 32k", "#eda100")]
AI_CLIP = 3e6
# M4 on-chip levels (bytes): device spec and design notes 6.8 / 8.
LEVELS = [(32 * 2**10, "threadgroup\n32 KiB"), (512 * 2**10, "per-core SRAM\n512 KiB (est.)"),
          (3 * 2**20, "GPU L2\n~3 MiB"), (12 * 2**20, "L2 + SLC\n12 MiB")]


def style(ax, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ticks = [2**k for k in range(10, 33, 2)]
    ax.set_xticks(ticks, [f"{t // 2**30} GiB" if t >= 2**30 else f"{t // 2**20} MiB" if t >= 2**20
                          else f"{t // 2**10} KiB" for t in ticks], rotation=45)
    ax.set_xlabel("fast-memory capacity S", color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left")


def main() -> None:
    model = load_model("configs/models/qwen3_0.6b.yaml")
    device = load_device("configs/hardware/m4_mba_24g.yaml")
    peak = device.engine("gpu").peak_flops(model.weight_dtype).effective
    B = device.bandwidth("gpu")
    ridge = peak / B
    graphs = [decode_graph(model, 1, 4096), prefill_graph(model, 1, 512),
              prefill_graph(model, 1, 4096), prefill_graph(model, 1, 32768)]
    S = 2.0 ** np.arange(10, 32.01, 0.25)

    data = []
    for (label, _), g in zip(SERIES, graphs):
        q = np.array([traffic_at(g, s) for s in S])  # columns: fused, operator
        data.append({"label": label, "flops": g.total_flops, "fused": q[:, 0], "operator": q[:, 1]})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE)
    style(a1, "arithmetic-intensity cap  FLOPs / Q_min(S)  (FLOP/byte)", "AI is derived: FLOPs / Q_min(S)")
    style(a2, "performance bound (FLOP/s)", f"Performance bound, DRAM {B / 1e9:.1f} GB/s")

    for (label, color), d in zip(SERIES, data):
        for kind, dash in (("fused", "-"), ("operator", (0, (4, 2)))):
            q = d[kind]
            # Q_min = 0 once S holds the whole working set (no DRAM traffic in
            # steady state): AI is unbounded there, drawn at the clip line.
            ai = np.where(q > 0, d["flops"] / np.maximum(q, 1.0), AI_CLIP)
            perf = d["flops"] / np.maximum(d["flops"] / peak, q / B)
            a1.plot(S, np.minimum(ai, AI_CLIP), color=color, linestyle=dash, linewidth=1.8)
            a2.plot(S, perf, color=color, linestyle=dash, linewidth=1.8)
        # direct label at the left end of the fused curve
        ai0 = d["flops"] / max(d["fused"][0], 1.0)
        dy = 0.62 if label == "prefill L 512" else 1.12  # keep neighbouring labels apart
        a1.annotate(label, (S[0] * 1.15, ai0 * dy), color=INK2, fontsize=8, va="top" if dy < 1 else "baseline")

    a1.axhline(ridge, color=INK, linewidth=1.1)
    a1.annotate(f"machine balance {ridge:.0f} FLOP/B\nabove: compute-bound", (2**27, ridge * 1.25),
                color=INK2, fontsize=8)
    a1.set_ylim(0.5, AI_CLIP * 1.6)
    a1.axhspan(AI_CLIP, AI_CLIP * 1.6, color=GRID, zorder=0)
    a1.annotate("unbounded: S holds all weights + KV", (2**22, AI_CLIP * 1.12), color=INK2, fontsize=7.5)
    a2.axhline(peak, color=INK, linewidth=1.1)
    a2.annotate(f"BF16 peak {peak / 1e12:.2f} TFLOP/s; prefill (fused) sits here at every S",
                (2**16, peak * 1.06), color=INK2, fontsize=8)
    for ax in (a1, a2):
        for s, name in LEVELS:
            ax.axvline(s, color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)), zorder=0)
        ax.set_xlim(S[0], S[-1])
    for k, (s, name) in enumerate(LEVELS):  # stagger so neighbours never collide
        a1.annotate(name, (s * 1.08, AI_CLIP / (1 if k % 2 == 0 else 12)), color=MUTED, fontsize=7, va="top")

    a2.legend(handles=[
        *[Line2D([], [], color=c, linewidth=1.8, label=l) for l, c in SERIES],
        Line2D([], [], color=MUTED, linewidth=1.8, label="fused (any implementation, proven)"),
        Line2D([], [], color=MUTED, linewidth=1.8, linestyle=(0, (4, 2)), label="operator boundary"),
    ], loc="lower right", fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle("Qwen3-0.6B BF16 on MacBook Air M4: performance bound as a function of fast-memory capacity "
                 "(two-level, one processor)", color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    out = Path("results/figures")
    fig.savefig(out / "perf_vs_capacity.png", dpi=160, facecolor=SURFACE)

    record = {"peak_flops": peak, "bandwidth": B, "S_bytes": S.tolist(),
              "series": [{"label": d["label"], "flops": d["flops"], "fused_bytes": d["fused"].tolist(),
                          "operator_bytes": d["operator"].tolist()} for d in data]}
    (out / "perf_vs_capacity.json").write_text(json.dumps(record))

    # Where each curve crosses the machine balance (becomes compute-bound).
    print(f"{'step':<16}{'bound':<10}{'compute-bound once S >=':>26}")
    for d in data:
        for kind in ("fused", "operator"):
            ai = d["flops"] / np.maximum(d[kind], 1.0)
            ok = np.nonzero(ai >= ridge)[0]
            # first S from which the curve stays above the ridge
            idx = next((i for i in range(len(S)) if (ai[i:] >= ridge).all()), None)
            where = "never" if idx is None else ("all S plotted" if idx == 0 else f"{S[idx] / 2**10:,.0f} KiB")
            print(f"{d['label']:<16}{kind:<10}{where:>26}")
    print(f"\nwrote {out}/perf_vs_capacity.png, perf_vs_capacity.json")


if __name__ == "__main__":
    main()
