"""Navigation pilot analysis (PILOT.md). Descriptive: two runs per arm support no inference.

Per blinded candidate and metric: paired ratio t_E0 / t_candidate (pairs = same session and block),
its median and geometric mean, and a one-sided sign test of median > 1.02 (descriptive p).
Primary metric per candidate: geometric mean over the 10 metrics of the paired median ratio.
A candidate failing the FP32 gate is void; a metric with an FP32-worse divergence beyond noise is
void for that candidate and left out of its primary metric (reported).
Unblinding uses runs/blind_key.json (name -> run id) and runs/<run>.meta.json (arm).

    python agent-test/nav/nav_analyze.py --results agent-test/v3/results-nav-<date>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = [("TTFT", "ttft_s", s) for s in ("ttft16", "ttft64", "ttft256", "ttft1024", "ctx4096", "ctx32768")] + \
          [("decode", "decode_s_median", s) for s in ("ctx512", "ctx4096", "ctx32768")] + \
          [("request", "request_completed_s", "req256x256")]


def sign_p(values, margin=math.log(1.02)):
    n = len(values)
    k = sum(v > margin for v in values)
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n if n else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    res = Path(a.results)
    runs = []
    for f in sorted((res / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        if "unit" in r and not r.get("excluded"):
            runs.append(r)
    gates = [json.loads(f.read_text()) for f in sorted(res.glob("gate-s*.json"))]
    divs = json.loads((res / "divergence_records.json").read_text()) if (res / "divergence_records.json").exists() else []
    cands = sorted({r["unit"]["engine"] for r in runs} - {"E0"})
    gate_fail = {e for g in gates for e, v in g["engines"].items() if not v["pass"]}

    t = defaultdict(dict)
    for r in runs:
        u = r["unit"]
        t[(u["scenario"], u["session"], u["block"])][u["engine"]] = r
    lines = ["# 导航预实验：候选引擎对 mlx-lm（E0）", "",
             "描述性结果：每组只有两次运行，不能下统计结论。比值 = t(E0) / t(候选)，> 1 表示候选更快。", ""]
    summary = {}
    for c in cands:
        lines += [f"## {c}" + ("（未通过 FP32 门槛：作废）" if c in gate_fail else ""), "",
                  "| 族 | 场景 | 对数 | 配对中位数比 | 几何平均比 | 符号检验 p（>1.02，描述） | 状态 |", "|---|---|---|---|---|---|---|"]
        medians, voids = [], []
        for fam, metric, sid in METRICS:
            lr = [math.log(d["E0"][metric] / d[c][metric]) for (s, _, _), d in t.items()
                  if s == sid and "E0" in d and c in d and d["E0"].get(metric) and d[c].get(metric)]
            if not lr:
                continue
            bad = any(x["engine"] == c and x["scenario"] == sid and x.get("invalidates") for x in divs)
            med = math.exp(statistics.median(lr))
            gm = math.exp(statistics.fmean(lr))
            state = "作废（分叉劣于 E0）" if bad else "有效"
            (voids if bad else medians).append((sid, metric, med))
            lines.append(f"| {fam} | {sid} | {len(lr)} | {med:.3f} | {gm:.3f} | {sign_p(lr):.4f} | {state} |")
        primary = math.exp(statistics.fmean(math.log(m) for _, _, m in medians)) if medians else None
        summary[c] = dict(primary=primary, n_metrics=len(medians), void_metrics=[v[:2] for v in voids],
                          gate_failed=c in gate_fail)
        lines += ["", f"主指标（{len(medians)} 项有效指标的配对中位数比的几何平均）："
                      + (f"**{primary:.3f}**" if primary and c not in gate_fail else "无"), ""]

    key_f = HERE / "runs" / "blind_key.json"
    if key_f.exists():
        key = json.loads(key_f.read_text())
        arms = defaultdict(list)
        lines += ["## 揭盲", "", "| 候选 | 运行 | 组 | 主指标 |", "|---|---|---|---|"]
        for c in cands:
            run = key.get(c)
            arm = json.loads((HERE / "runs" / f"{run}.meta.json").read_text())["arm"] if run else "?"
            p = summary[c]["primary"]
            valid = p is not None and not summary[c]["gate_failed"]
            if valid:
                arms[arm].append(p)
            lines.append(f"| {c} | {run} | {arm} | {p:.3f} |" if valid else f"| {c} | {run} | {arm} | 作废 |")
        lines += ["", "| 组 | 运行数 | 主指标（各次） | 几何平均 |", "|---|---|---|---|"]
        for arm, ps in sorted(arms.items()):
            gm = math.exp(statistics.fmean(math.log(p) for p in ps))
            lines.append(f"| {arm} | {len(ps)} | {', '.join(f'{p:.3f}' for p in ps)} | {gm:.3f} |")
        summary["_arms"] = {k: v for k, v in arms.items()}
    usage_f = HERE / "runs" / "usage.json"
    if usage_f.exists():
        lines += ["## Agent 用量（来自 Codex 会话记录）", "",
                  "| 运行 | 时长 h | 模型请求 | 工具调用 | 工具时间占比 | 输入 token（缓存） | 输出 token | 思考 token |",
                  "|---|---|---|---|---|---|---|---|"]
        for u in json.loads(usage_f.read_text()):
            lines.append(f"| {u['run']} | {u['wall_s'] / 3600:.2f} | {u['requests']} | {u['tool_calls']} | "
                         f"{u['tool_time_s'] / max(u['wall_s'], 1):.0%} | {u['input_tokens']:,} ({u['cached_input_tokens']:,}) | "
                         f"{u['output_tokens']:,} | {u['reasoning_output_tokens']:,} |")
        lines.append("")
    (res / "nav_report.md").write_text("\n".join(lines))
    (res / "nav_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
