"""Gap decomposition report: engine vs physical bound (plan item 16, design notes 6.9).

Inputs (newest of each in results/raw unless given):
  ablation_mlx-*.json         telescoping measurements (benchmarks/ablation_mlx.py)
  baseline_mlx-*.json         kernel structure of a decode step (benchmarks/baseline_mlx.py)
  baseline_llamacpp_*.json    llama-bench totals (no decomposition: no ablation)
  configs/hardware/m4_mba_24g.kernels.json   calibration (scripts/calibrate.py)

Outputs:
  results/figures/gap_mlx.png     100%-stacked bars: bound + loss per piece
  results/figures/gap.json        every number, measured and modeled

    python scripts/gap_report.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from roofline import load_device, load_model  # noqa: E402
from roofline.bound import physical_bound  # noqa: E402
from roofline.gap import decode_gap, load_calibration, prefill_gap  # noqa: E402
from roofline.workload import decode, prefill  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BOUND = "#c9c8c2"
PIECE_COLOR = {"gemv": "#2a78d6", "gemm": "#2a78d6", "attention": "#eb6834", "small": "#1baf7a", "engine": "#eda100"}
PIECE_LABEL = {"gemv": "matmul (GEMV / GEMM)", "gemm": "matmul (GEMV / GEMM)", "attention": "attention",
               "small": "small kernels", "engine": "engine (sync, Python, loop)"}


def newest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def short(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 else str(n)


def llama_rows(model, device) -> list[dict]:
    rows = []
    for path in (newest("results/raw/baseline_llamacpp_pp-*.json"), newest("results/raw/baseline_llamacpp_tg-*.json")):
        if not path:
            continue
        for r in json.loads(Path(path).read_text()):
            if r["n_gen"] == 0:  # prefill: avg_ts = prompt tokens / s
                t = r["n_prompt"] / r["avg_ts"]
                phys = physical_bound(prefill(model, 1, r["n_prompt"]), device).time_s
                rows.append({"engine": "llama.cpp", "phase": "prefill", "n": r["n_prompt"], "measured_s": t,
                             "phys_s": phys, "efficiency": phys / t, "flash_attn": r.get("flash_attn")})
            else:  # decode at depth d: avg_ts = generated tokens / s
                t = 1 / r["avg_ts"]
                ctx = r.get("n_depth", 0) + r["n_gen"] // 2
                phys = physical_bound(decode(model, 1, ctx), device).time_s
                rows.append({"engine": "llama.cpp", "phase": "decode", "n": r.get("n_depth", 0), "measured_s": t,
                             "phys_s": phys, "efficiency": phys / t, "flash_attn": r.get("flash_attn")})
    return rows


def bars(ax, gaps: list[dict], labels: list[str], title: str) -> None:
    ax.set_facecolor(SURFACE)
    y = range(len(gaps))
    for i, g in enumerate(gaps):
        left = 0.0
        segs = [("bound", g["phys_s"], BOUND)]
        segs += [(k, max(0.0, p["loss_s"]), PIECE_COLOR[k]) for k, p in g["pieces"].items()]
        # A negative piece (measurement noise, or overlap the ablation cannot see)
        # is not drawn; bars are shares of bound + positive losses, and the
        # negative piece is named beside the measured time.
        denom = sum(v for _, v, _ in segs)
        negative = {k: p["loss_s"] for k, p in g["pieces"].items() if p["loss_s"] < 0}
        for name, val, color in segs:
            w = val / denom
            if w <= 0:
                continue
            ax.barh(i, w, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=2)
            if w > 0.045:
                ax.text(left + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=7.5,
                        color=INK if name in ("bound", "small", "engine") else "white")
            left += w
        total = g["measured_s"]
        unit = "ms" if total < 2 else "s"
        val = total * 1e3 if unit == "ms" else total
        note = "".join(f"\n{k} {v * 1e3:+.1f} ms" for k, v in negative.items())
        ax.text(1.01, i, f"{val:.1f} {unit}{note}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(list(y), labels, fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=11, loc="left")


def main() -> None:
    model = load_model("configs/models/qwen3_0.6b.yaml")
    device = load_device("configs/hardware/m4_mba_24g.yaml")
    cal = load_calibration("configs/hardware/m4_mba_24g.kernels.json")
    abl = json.loads(Path(newest("results/raw/ablation_mlx-*.json")).read_text())
    base = json.loads(Path(newest("results/raw/baseline_mlx-*.json")).read_text())
    structure = base["structure"]

    dec = [decode_gap(model, device, cal, r, structure) for r in abl["decode"]]
    pre = [prefill_gap(model, device, cal, r, structure) for r in abl["prefill"]]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2), facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [len(dec), len(pre)]})
    bars(a1, dec, [f"decode ctx {short(g['ctx'])}" for g in dec], "mlx-lm decode: share of measured time per token")
    bars(a2, pre, [f"prefill L {short(g['L'])}" for g in pre], "mlx-lm prefill: share of measured time to first token")
    handles = [matplotlib.patches.Patch(color=BOUND, label="physical bound")]
    handles += [matplotlib.patches.Patch(color=PIECE_COLOR[k], label=f"loss: {PIECE_LABEL[k]}")
                for k in ("gemv", "attention", "small", "engine")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, frameon=False, labelcolor=INK2)
    fig.suptitle("Where mlx-lm's time goes vs the level-A bound (Qwen3-0.6B BF16, MacBook Air M4)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = Path("results/figures")
    fig.savefig(out / "gap_mlx.png", dpi=160, facecolor=SURFACE)

    llama = llama_rows(model, device)
    (out / "gap.json").write_text(json.dumps({"mlx_lm": {"decode": dec, "prefill": pre}, "llama_cpp": llama,
                                              "sources": {"ablation": abl["timestamp"], "baseline": base["timestamp"],
                                                          "calibration": cal.data["sources"]}}, indent=1))

    def row(g, name):
        p = g["pieces"]
        # measured loss vs modeled loss (model - ideal) per piece
        cells = "".join(f"{p[k]['loss_s'] * 1e3:>9.2f}{(p[k]['model_s'] - p[k]['ideal_s']) * 1e3:>8.2f}" for k in p)
        print(f"{name:<14}{g['phys_s'] * 1e3:>9.2f}{g['measured_s'] * 1e3:>10.2f}{g['phys_s'] / g['measured_s']:>6.0%}"
              f"{g['model_s'] * 1e3:>9.2f}{cells}")

    hdr = "".join(f"{k + ' loss':>9}{'model':>8}" for k in ("mat", "attn", "small", "eng"))
    print(f"{'mlx-lm (ms)':<14}{'bound':>9}{'measured':>10}{'eff':>6}{'T_A':>9}{hdr}")
    for g in dec:
        row(g, f"decode {short(g['ctx'])}")
    for g in pre:
        row(g, f"prefill {short(g['L'])}")
    if llama:
        print(f"\n{'llama.cpp':<14}{'bound ms':>10}{'measured ms':>13}{'eff':>6}")
        for r in llama:
            print(f"{r['phase'] + ' ' + short(r['n']):<14}{r['phys_s'] * 1e3:>10.2f}{r['measured_s'] * 1e3:>13.2f}"
                  f"{r['efficiency']:>6.0%}")
    print(f"\nwrote {out}/gap_mlx.png, gap.json")


if __name__ == "__main__":
    main()
