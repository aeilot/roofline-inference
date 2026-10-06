"""Roofline figures from the pebble-game I/O bounds (design notes 6.7, 6.8).

  results/figures/roofline_dram.png          whole-step points at the DRAM boundary,
                                             fused vs operator bound, plus per-operator
                                             points (operator bound)
  results/figures/roofline_hierarchy.png     the same steps at every boundary of the
                                             hierarchy (operator bound)
  results/figures/roofline.json              every number plotted, with provenance

Every point sits on its roof by construction (it is an upper bound); what the
figure shows is where the intensity cap lies relative to the ridge.

    python scripts/plot_roofline.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from roofline import load_device, load_model  # noqa: E402
from roofline.graph import decode_graph, prefill_graph  # noqa: E402
from roofline.iobound import io_bound  # noqa: E402

DECODE_CTX = (512, 4096, 32768)
PREFILL_LEN = (16, 64, 512, 4096, 32768)
OP_DETAIL = ("prefill", 4096)  # per-operator points for this step
GROUPS = ("qkv", "o_proj", "mlp", "attention", "lm_head")

# Reference palette (dataviz skill), light mode; slots 1-2 validated.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
PHASE_COLOR = {"decode": "#2a78d6", "prefill": "#eb6834"}
BOUNDARY_MARKER = {"chip": "o", "core": "s"}
BOUNDARY_LABEL = {"chip": "DRAM ↔ chip", "core": "L2 ↔ cores"}


def short(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 else str(n)


def steps(model):
    for n in DECODE_CTX:
        yield "decode", n, decode_graph(model, 1, n)
    for n in PREFILL_LEN:
        yield "prefill", n, prefill_graph(model, 1, n)


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("arithmetic intensity cap (FLOP / byte across the boundary)", color=INK2, fontsize=10)
    ax.set_ylabel("performance bound (FLOP/s)", color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left")


ROOF_STYLE = {"nominal": (MUTED, 1.0, (0, (4, 3))), "chip": (INK, 1.6, "-"), "core": (INK2, 1.4, (0, (6, 2, 1, 2)))}


def roof(ax, peak: float, bw: float, x_lo: float, x_hi: float, label: str, kind: str) -> Line2D:
    """Bandwidth slope up to the ridge, then the compute ceiling. Labels go in
    the legend so they never sit on a line."""
    color, width, dash = ROOF_STYLE[kind]
    ridge = peak / bw
    ax.plot([x_lo, ridge, x_hi], [x_lo * bw, peak, peak], color=color, linewidth=width, linestyle=dash, zorder=1)
    return Line2D([], [], color=color, linewidth=width, linestyle=dash, label=label)


def perf(ai: float, peak: float, bw: float) -> float:
    return min(peak, ai * bw)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="configs/hardware/m4_mba_24g.yaml")
    p.add_argument("--model", default="configs/models/qwen3_0.6b.yaml")
    p.add_argument("--out", default="results/figures")
    p.add_argument("--label", default="MacBook Air M4", help="device name in titles")
    args = p.parse_args()

    model, device = load_model(args.model), load_device(args.device)
    bounds = [(ph, n, io_bound(g, device, dtype=model.weight_dtype)) for ph, n, g in steps(model)]
    peak = bounds[0][2].peak_flops
    nominal = device.total_bandwidth.value
    by_name = {lv.boundary.name: lv.boundary for lv in bounds[0][2].levels}
    dram = by_name["chip"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- figure 1: DRAM boundary -------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    x_lo, x_hi = 0.3, 3e5
    style(ax, "Whole step, DRAM boundary")
    roofs = [roof(ax, peak, nominal, x_lo, x_hi, f"nominal {nominal / 1e9:.0f} GB/s", "nominal"),
             roof(ax, peak, dram.bandwidth, x_lo, x_hi,
                  f"DRAM {dram.bandwidth / 1e9:.1f} GB/s measured; BF16 {peak / 1e12:.2f} TFLOP/s", "chip")]
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(2e10, 8e12)

    for ph in ("decode", "prefill"):
        rows = [(n, b) for p_, n, b in bounds if p_ == ph]
        for n, b in rows:
            x0, x1 = b.intensity("fused")["chip"], b.intensity("operator")["chip"]
            y0, y1 = perf(x0, peak, dram.bandwidth), perf(x1, peak, dram.bandwidth)
            ax.scatter([x0], [y0], s=26, color=PHASE_COLOR[ph], zorder=4)
            ax.scatter([x1], [y1], s=120, facecolors="none", edgecolors=PHASE_COLOR[ph], linewidths=1.5, zorder=3)
            if x1 < x0 * 0.98:
                ax.annotate("", (x1 * 1.25, y1), (x0 / 1.25, y0),
                            arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9), zorder=2)
            on_roof = y0 >= peak
            ax.annotate(short(n), (x0, y0 * (0.62 if on_roof else 1.4)), ha="center", fontsize=8, color=INK2)

    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=PHASE_COLOR["decode"], label="decode (context length)"),
        Line2D([], [], marker="o", linestyle="", color=PHASE_COLOR["prefill"], label="prefill (prompt length)"),
        Line2D([], [], marker="o", linestyle="", markersize=5, color=MUTED, label="fused: whole graph, proven"),
        Line2D([], [], marker="o", linestyle="", markersize=10, markerfacecolor="none", markeredgecolor=MUTED,
               label="operator boundary (arrow: shift)"),
        *roofs,
    ], loc="lower right", fontsize=8, frameon=False, labelcolor=INK2)

    # Per-operator dot plot: intensity cap of each operator group vs the ridge.
    ax2.set_facecolor(SURFACE)
    ax2.set_xscale("log")
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color(MUTED)
    ax2.tick_params(colors=INK2, labelsize=9, left=False)
    ax2.grid(True, axis="x", which="major", color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    ridge = peak / dram.bandwidth
    ax2.axvline(ridge, color=INK, linewidth=1.2, zorder=1)
    ax2.annotate(f"ridge {ridge:.0f} FLOP/B\nleft: DRAM-bound, right: compute-bound", (ridge * 1.15, -0.5),
                 fontsize=8, color=INK2, va="top")
    detail = {(p_, n): b for p_, n, b in bounds}
    picks = (("decode", 4096), OP_DETAIL, ("prefill", 32768))
    offsets = (-0.18, 0.0, 0.18)
    for (ph, n), dy in zip(picks, offsets):
        groups = {lv.boundary.name: lv.groups for lv in detail[(ph, n)].levels}["chip"]
        xs = [groups[g]["flops"] / groups[g]["bytes"] for g in GROUPS]
        hollow = n == 32768
        ax2.scatter(xs, [i + dy for i in range(len(GROUPS))], s=40, zorder=3,
                    facecolors="none" if hollow else PHASE_COLOR[ph], edgecolors=PHASE_COLOR[ph], linewidths=1.4)
    ax2.set_yticks(range(len(GROUPS)), GROUPS)
    ax2.set_ylim(-0.6, len(GROUPS) - 0.4)
    ax2.invert_yaxis()
    ax2.set_xlim(0.3, 3e5)
    ax2.set_xlabel("arithmetic intensity cap (FLOP / DRAM byte)", color=INK2, fontsize=10)
    ax2.set_title("Per operator group, operator bound, DRAM boundary", color=INK, fontsize=11, loc="left")
    ax2.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=PHASE_COLOR["decode"], label="decode ctx 4k"),
        Line2D([], [], marker="o", linestyle="", color=PHASE_COLOR["prefill"], label=f"prefill L {short(OP_DETAIL[1])}"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=PHASE_COLOR["prefill"],
               label="prefill L 32k"),
    ], loc="lower right", fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle(f"Qwen3-0.6B BF16 on {args.label}: pebble-game I/O lower bounds → roofline",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "roofline_dram.png", dpi=160, facecolor=SURFACE)

    # --- figure 2: every boundary --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.6), facecolor=SURFACE)
    style(ax, "Whole step at every boundary, operator bound (fused is ≤ this)")
    x_lo, x_hi = 0.3, 3e5
    roofs = [roof(ax, peak, b.bandwidth, x_lo, x_hi,
                  f"{BOUNDARY_LABEL[name]}: {b.bandwidth / 1e9:.0f} GB/s, S = {b.fast_bytes / 2**20:.1f} MiB"
                  f"{' per core × ' + str(b.processors) if b.processors > 1 else ''}", name)
             for name, b in by_name.items()]
    for ph, n, b in bounds:
        for name, bd in by_name.items():
            x = b.intensity("operator")[name]
            y = perf(x, peak, bd.bandwidth)
            ax.scatter([x], [y], s=52, marker=BOUNDARY_MARKER[name], facecolors=PHASE_COLOR[ph],
                       edgecolors=SURFACE, linewidths=1.2, zorder=3)
            if name == "core" and (ph, n) in {("decode", 32768), ("prefill", 512), ("prefill", 32768)}:
                ax.annotate(f"{ph} {short(n)}", (x, y * (0.62 if y >= peak else 1.35)), ha="center",
                            fontsize=8, color=INK2)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(2e10, 2e13)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=PHASE_COLOR["decode"], label="decode"),
        Line2D([], [], marker="o", linestyle="", color=PHASE_COLOR["prefill"], label="prefill"),
        *[Line2D([], [], marker=BOUNDARY_MARKER[k], linestyle="", color=MUTED, label=BOUNDARY_LABEL[k])
          for k in by_name],
        *roofs,
    ], loc="lower right", fontsize=8, frameon=False, labelcolor=INK2)
    ax.set_title(f"Whole step at every boundary, operator bound; BF16 peak {peak / 1e12:.2f} TFLOP/s",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "roofline_hierarchy.png", dpi=160, facecolor=SURFACE)

    record = {
        "device": device.name, "model": model.name,
        "boundaries": {k: {"fast_bytes_per_processor": v.fast_bytes, "processors": v.processors,
                           "bandwidth": v.bandwidth} for k, v in by_name.items()},
        "steps": [b.to_dict() for _, _, b in bounds],
    }
    (out / "roofline.json").write_text(json.dumps(record, indent=1))
    print(f"wrote {out}/roofline_dram.png, roofline_hierarchy.png, roofline.json")
    print(f"\n{'step':<16}{'bound':<10}{'time ms':>10}  {'bottleneck':<10}" +
          "".join(f"{'AI ' + k:>12}" for k in by_name))
    for ph, n, b in bounds:
        for kind in ("fused", "operator"):
            ai = b.intensity(kind)
            print(f"{ph + ' ' + short(n):<16}{kind:<10}{b.time_s(kind) * 1e3:>10.2f}  {b.bottleneck(kind):<10}" +
                  "".join(f"{ai[k]:>12.1f}" for k in by_name))


if __name__ == "__main__":
    main()
