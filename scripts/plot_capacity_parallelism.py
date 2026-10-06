"""Performance bound over (fast-memory capacity S, available parallelism T)
(design notes 6.11, plan 16d).

    perf(S, T) <= FLOPs / max( FLOPs / P(T),  Q_min(S) / B(T) )

Q_min(S): least traffic for the step's graph with S bytes of fast memory
(src/roofline/iobound.py, operator-boundary bound). B(T) and P(T): measured
read bandwidth and FMA throughput with T concurrent threads
(benchmarks/concurrency.py; 4 loads in flight per thread). Also fits
Little's law B(n) = min(B_max, n * 16 B / L) to the unsaturated points.

  results/figures/perf_capacity_parallelism.png
  results/figures/perf_capacity_parallelism.json

    python scripts/plot_capacity_parallelism.py
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from roofline import load_model  # noqa: E402
from roofline.graph import decode_graph, prefill_graph  # noqa: E402
from roofline.iobound import traffic_at  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
# Sequential blue ramp (dataviz reference palette, steps 100 -> 700).
RAMP = ListedColormap(["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
LEVELS = [(32 * 2**10, "threadgroup"), (512 * 2**10, "core SRAM (est.)"), (12 * 2**20, "L2+SLC")]


def little_fit(threads: list[int], bw: list[float], ilp: int) -> dict:
    """B_max = median of the plateau (T >= 8192); latency L from the points
    below 70% of B_max: L = bytes in flight / bandwidth."""
    bmax = statistics.median([b for t, b in zip(threads, bw) if t >= 8192])
    lats = [t * ilp * 16 / b for t, b in zip(threads, bw) if b < 0.7 * bmax]
    L = statistics.median(lats)
    return {"B_max": bmax, "latency_s": L, "bytes_in_flight_to_saturate": bmax * L,
            "latency_spread_s": [min(lats), max(lats)]}


def main() -> None:
    rec = json.loads(Path(sorted(glob.glob("results/raw/concurrency-*.json"))[-1]).read_text())
    r = rec["results"]
    threads = np.array(r["threads"], dtype=float)
    fit1 = little_fit(r["threads"], r["bandwidth_ilp1"], 1)
    fit4 = little_fit(r["threads"], r["bandwidth_ilp4"], 4)
    # Monotone envelopes of the measured tables (more threads never hurt the roof).
    B = np.minimum.accumulate(np.array(r["bandwidth_ilp4"])[::-1])[::-1]
    B = np.maximum.accumulate(np.minimum(B, fit4["B_max"]))
    P = np.maximum.accumulate(np.array(r["compute_bf16"]))
    peak = P.max()

    model = load_model("configs/models/qwen3_0.6b.yaml")
    cases = [("decode ctx 4k", decode_graph(model, 1, 4096)), ("prefill L 4k", prefill_graph(model, 1, 4096))]
    # Up to 256 MiB, far beyond the on-chip levels: from ~1 GiB the weights start to fit and
    # decode needs no DRAM traffic, which perf_vs_capacity.png already shows.
    S = 2.0 ** np.arange(10, 28.01, 0.25)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1, 0.95], "wspace": 0.42})
    out_cases = []
    for ax, (label, g) in zip(axes[:2], cases):
        F = g.total_flops
        Q = np.array([traffic_at(g, s)[1] for s in S])
        perf = F / np.maximum(F / P[:, None], Q[None, :] / B[:, None])  # [T, S]
        best = perf.max()
        frac = perf / best  # each step against its own best bound in the plotted range
        ax.set_facecolor(SURFACE)
        mesh = ax.pcolormesh(S, threads, frac, cmap=RAMP, vmin=0, vmax=1, shading="nearest")
        levels = [0.25, 0.5, 0.9]
        ax.contour(S, threads, frac, levels=levels, colors=[INK], linewidths=0.8)
        ix = int(np.searchsorted(S, 2**24))  # label the contours in one column, away from the edges
        for lv in levels:
            it = int(np.argmax(frac[:, ix] >= lv))
            if frac[it, ix] >= lv:
                ax.annotate(f"{lv:.0%}", (S[ix], threads[it]), color=INK, fontsize=7.5, ha="center",
                            va="bottom", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        for s, name in LEVELS:
            ax.axvline(s, color="white", linewidth=0.8, linestyle=(0, (2, 2)))
            ax.annotate(name, (s * 1.1, threads[0] * 1.3), color="white", fontsize=7, rotation=90, va="bottom")
        xt = [2**k for k in range(10, 29, 3)]
        ax.set_xticks(xt, [f"{t // 2**20} MiB" if t >= 2**20 else f"{t // 2**10} KiB" for t in xt])
        yt = [2**k for k in range(5, 21, 3)]
        ax.set_yticks(yt, [f"{t:,}" for t in yt])
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.set_xlabel("fast-memory capacity S", color=INK2, fontsize=10)
        ax.set_ylabel("concurrent threads T (4 loads in flight each)", color=INK2, fontsize=10)
        t_best = F / best
        unit = f"{t_best * 1e3:.1f} ms" if t_best < 1 else f"{t_best:.2f} s"
        ax.set_title(f"{label}: bound / best bound ({unit}, {best / peak:.0%} of peak)",
                     color=INK, fontsize=10.5, loc="left")
        out_cases.append({"label": label, "flops": F, "best_time_s": t_best, "Q_bytes": Q.tolist(),
                          "fraction_of_best": frac.tolist()})
    cb = fig.colorbar(mesh, ax=axes[1], fraction=0.05, pad=0.03)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.set_label("bound / best bound", color=INK2, fontsize=9)

    # Right panel: the two roofs vs parallelism.
    ax = axes[2]
    ax.set_facecolor(SURFACE)
    ax.plot(threads, np.array(r["bandwidth_ilp1"]) / fit1["B_max"], color="#2a78d6", linewidth=1.8,
            marker="o", markersize=3, label="read bandwidth, 1 load in flight")
    b4 = np.array(r["bandwidth_ilp4"]) / fit4["B_max"]
    odd = b4 > 1.15  # above DRAM: on-chip cache hits, not a DRAM measurement
    ax.plot(threads[~odd], b4[~odd], color="#2a78d6", linewidth=1.8, linestyle=(0, (4, 2)), marker="o",
            markersize=3, label="read bandwidth, 4 loads in flight")
    ax.scatter(threads[odd], np.minimum(b4[odd], 1.2), s=22, facecolors="none", edgecolors="#2a78d6",
               label="outlier above DRAM (cache), clipped")
    ax.plot(threads, np.array(r["compute_bf16"]) / peak, color="#eb6834", linewidth=1.8, marker="s",
            markersize=3, label="BF16 FMA throughput")
    lit = np.minimum(1, threads * 16 / fit1["latency_s"] / fit1["B_max"])
    ax.plot(threads, lit, color=MUTED, linewidth=1.0, linestyle=(0, (1, 2)),
            label=f"Little's law, L = {fit1['latency_s'] * 1e9:.0f} ns")
    ax.set_xscale("log", base=2)
    ax.set_xticks(yt, [f"{t:,}" for t in yt], rotation=30)
    ax.set_ylim(0, 1.25)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_xlabel("concurrent threads T", color=INK2, fontsize=10)
    ax.set_ylabel("fraction of the roof", color=INK2, fontsize=10)
    ax.set_title("Roofs vs parallelism (measured)", color=INK, fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=INK2)

    fig.suptitle("Qwen3-0.6B BF16 on MacBook Air M4: performance bound over capacity and parallelism",
                 color=INK, fontsize=12, x=0.01, ha="left")
    out = Path("results/figures")
    fig.savefig(out / "perf_capacity_parallelism.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    (out / "perf_capacity_parallelism.json").write_text(json.dumps({
        "source": rec["timestamp"], "threads": r["threads"], "S_bytes": S.tolist(),
        "little_ilp1": fit1, "little_ilp4": fit4, "peak_flops": peak, "cases": out_cases}))

    print(f"Little's law, ILP 1: B_max {fit1['B_max'] / 1e9:.1f} GB/s, L {fit1['latency_s'] * 1e9:.0f} ns "
          f"(spread {fit1['latency_spread_s'][0] * 1e9:.0f}-{fit1['latency_spread_s'][1] * 1e9:.0f}), "
          f"saturates at {fit1['bytes_in_flight_to_saturate'] / 1024:.0f} KiB in flight")
    print(f"Little's law, ILP 4: B_max {fit4['B_max'] / 1e9:.1f} GB/s, L {fit4['latency_s'] * 1e9:.0f} ns")
    for frac_target in (0.5, 0.9):
        tb = threads[np.argmax(B >= frac_target * B.max())]
        tp = threads[np.argmax(P >= frac_target * peak)]
        print(f"{frac_target:.0%} of roof: bandwidth needs T >= {tb:,.0f}, compute needs T >= {tp:,.0f}")
    print(f"wrote {out}/perf_capacity_parallelism.png")


if __name__ == "__main__":
    main()
