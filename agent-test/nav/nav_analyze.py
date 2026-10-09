"""Navigation pilot analysis (PILOT.md, Revision 3). Descriptive: two runs per arm support no inference.

Per blinded candidate and metric: paired ratio t_E0 / t_candidate (pairs = same session and block),
its median and geometric mean, and a one-sided sign test of median > 1.02 (descriptive p).

Primary score per candidate (the scoring rule stated in TASK.md):
  scenario score = geometric mean of the paired median ratios of the scenario's metrics
                   (TTFT for ttft*, TTFT and decode for ctx*, whole request for req256x256);
  primary        = geometric mean of the 8 scenario scores.
A scenario with an FP32-worse divergence beyond noise (signed rule) is void and scores 1.0 (no
speed-up: a user would fall back to mlx-lm); this keeps every candidate on the same 8 scenarios.
Sensitivity: the geometric mean over valid scenarios only. A candidate failing the FP32 gate is void.
Plausibility: a metric whose median time is below 0.9x the resource reference line (BOUND.md) is
flagged for manual review (measurement or rule problem); static audit findings come from nav_audit.py.
Unblinding uses runs/blind_key.json (name -> run id) and runs/<run>.meta.json (arm).

    python agent-test/nav/nav_analyze.py --results agent-test/v3/results-nav-<date>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENARIOS = {"ttft16": ["ttft_s"], "ttft64": ["ttft_s"], "ttft256": ["ttft_s"], "ttft1024": ["ttft_s"],
             "ctx512": ["ttft_s", "decode_s_median"], "ctx4096": ["ttft_s", "decode_s_median"],
             "ctx32768": ["ttft_s", "decode_s_median"], "req256x256": ["request_completed_s"]}
LABEL = {"ttft_s": "TTFT", "decode_s_median": "decode", "request_completed_s": "request"}
# Resource reference lines in seconds (kit/bound/docs/BOUND.md; ctx512 prefill and 256+256 request
# computed the same way: prefill 0.130 s; 61.4 ms + 255 x 12.0 ms).
REFERENCE = {("ttft16", "ttft_s"): 0.0115, ("ttft64", "ttft_s"): 0.0151, ("ttft256", "ttft_s"): 0.0614,
             ("ttft1024", "ttft_s"): 0.269, ("ctx512", "ttft_s"): 0.130, ("ctx4096", "ttft_s"): 1.45,
             ("ctx32768", "ttft_s"): 39.9, ("ctx512", "decode_s_median"): 0.0120,
             ("ctx4096", "decode_s_median"): 0.0160, ("ctx32768", "decode_s_median"): 0.0480,
             ("req256x256", "request_completed_s"): 3.12}


def sign_p(values, margin=math.log(1.02)):
    n = len(values)
    k = sum(v > margin for v in values)
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n if n else 1.0


def gmean(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else None


def fmt(x):
    return f"{x:.3f}" if x is not None else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    res = Path(a.results)
    runs, excluded = [], Counter()
    for f in sorted((res / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        eng = (r.get("unit") or {}).get("engine") or r.get("engine", "?")
        if r.get("excluded"):
            excluded[(eng, r["excluded"])] += 1
        elif "unit" in r:
            runs.append(r)
    gates = {f.stem: json.loads(f.read_text()) for f in sorted(res.glob("gate-s*.json"))}
    divs = json.loads((res / "divergence_records.json").read_text()) if (res / "divergence_records.json").exists() else []
    cands = sorted({r["unit"]["engine"] for r in runs} - {"E0"})
    gate_fail = {e for g in gates.values() for e, v in g["engines"].items() if not v["pass"]}

    t = defaultdict(dict)
    for r in runs:
        u = r["unit"]
        t[(u["scenario"], u["session"], u["block"])][u["engine"]] = r
    lines = ["# 导航预实验：候选引擎对 mlx-lm（E0）", "",
             "描述性结果：每组只有两次运行，不能下统计结论。比值 = t(E0) / t(候选)，> 1 表示候选更快。",
             "主指标：8 个场景等权；场景得分 = 该场景各指标配对中位数比的几何平均；作废场景记 1.0。"
             "敏感性：只用有效场景。", ""]
    summary, flags = {}, []
    for c in cands:
        lines += [f"## {c}" + ("（未通过 FP32 门槛：作废）" if c in gate_fail else ""), "",
                  "| 场景 | 指标 | 对数 | 候选中位时间 | 配对中位数比 | 几何平均比 | 符号检验 p（>1.02，描述） | 状态 |",
                  "|---|---|---|---|---|---|---|---|"]
        scores, valid_scores, void, missing = {}, {}, [], []
        for sid, metrics in SCENARIOS.items():
            bad = any(x["engine"] == c and x["scenario"] == sid and x.get("invalidates") for x in divs)
            meds = []
            for metric in metrics:
                ds = [d for (s, _, _), d in t.items() if s == sid and "E0" in d and c in d
                      and d["E0"].get(metric) and d[c].get(metric)]
                if not ds:
                    continue
                lr = [math.log(d["E0"][metric] / d[c][metric]) for d in ds]
                med = math.exp(statistics.median(lr))
                tc = statistics.median(d[c][metric] for d in ds)
                meds.append(med)
                if tc < 0.9 * REFERENCE[(sid, metric)]:
                    flags.append(f"{c} {sid} {LABEL[metric]}: 中位时间 {tc * 1e3:.1f} ms 低于参考线 "
                                 f"{REFERENCE[(sid, metric)] * 1e3:.1f} ms 的 0.9 倍")
                lines.append(f"| {sid} | {LABEL[metric]} | {len(lr)} | {tc * 1e3:.1f} ms | {med:.3f} | "
                             f"{math.exp(statistics.fmean(lr)):.3f} | {sign_p(lr):.4f} | "
                             + ("作废（分叉劣于 E0）" if bad else "有效") + " |")
            if len(meds) < len(metrics):
                missing.append(sid)
                continue
            if bad:
                void.append(sid)
                scores[sid] = 1.0
            else:
                scores[sid] = valid_scores[sid] = gmean(meds)
        complete = not missing
        primary = gmean(list(scores.values())) if complete and c not in gate_fail else None
        sens = gmean(list(valid_scores.values())) if c not in gate_fail else None
        summary[c] = dict(primary=primary, sensitivity_valid_only=sens, scenario_scores=scores,
                          void_scenarios=void, missing_scenarios=missing, gate_failed=c in gate_fail)
        lines += ["", "| 场景 | 得分 |", "|---|---|"] + [f"| {s} | {v:.3f}{'（作废，记 1.0）' if s in void else ''} |"
                                                         for s, v in scores.items()]
        lines += ["", f"主指标：**{fmt(primary)}**" + ("" if complete else f"（缺少场景 {missing}：不计算）")
                  + f"；敏感性（仅 {len(valid_scores)} 个有效场景）：{fmt(sens)}", ""]

    key_f = HERE / "runs" / "blind_key.json"
    if key_f.exists():
        key = json.loads(key_f.read_text())
        arms = defaultdict(list)
        lines += ["## 揭盲", "", "| 候选 | 运行 | 组 | 主指标 | 敏感性 | 作废场景 |", "|---|---|---|---|---|---|"]
        for c in cands:
            run = key.get(c)
            arm = json.loads((HERE / "runs" / f"{run}.meta.json").read_text())["arm"] if run else "?"
            s = summary[c]
            if s["primary"] is not None:
                arms[arm].append((s["primary"], s["sensitivity_valid_only"]))
            lines.append(f"| {c} | {run} | {arm} | {fmt(s['primary']) if not s['gate_failed'] else '作废'} | "
                         f"{fmt(s['sensitivity_valid_only'])} | {', '.join(s['void_scenarios']) or '—'} |")
        lines += ["", "| 组 | 运行数 | 主指标（各次） | 几何平均 | 敏感性几何平均 |", "|---|---|---|---|---|"]
        for arm, ps in sorted(arms.items()):
            lines.append(f"| {arm} | {len(ps)} | {', '.join(f'{p:.3f}' for p, _ in ps)} | "
                         f"{fmt(gmean([p for p, _ in ps]))} | {fmt(gmean([s for _, s in ps if s]))} |")
        lines.append("")
        summary["_arms"] = {k: v for k, v in arms.items()}

    lines += ["## 可信度复核", ""]
    lines += [f"- {f}" for f in flags] or ["- 没有指标低于参考线的 0.9 倍。"]
    audit_f = HERE / "runs" / "audit.json"
    if audit_f.exists():
        audit = json.loads(audit_f.read_text())
        for c, hits in sorted(audit.items()):
            lines.append(f"- 静态审计 {c}：" + ("; ".join(f"{h['file']}:{h['line']} `{h['match']}`" for h in hits[:12])
                                             + (f" …共 {len(hits)} 处" if len(hits) > 12 else "") if hits else "无可疑调用"))
    lines.append("")
    summary["_plausibility_flags"] = flags

    lines += ["## 数据质量", ""]
    # kit check.py hard-codes the evaluation threshold so agents test against the same number
    kit_noise = 0.2241
    for name, g in gates.items():
        thr = g.get("e0_margin_err_p99")
        if thr is not None and abs(thr - kit_noise) > 0.01 * kit_noise:
            lines.append(f"- **警告**：{name} 的噪声阈值 {thr:.4f} 与 kit check.py 的 NOISE {kit_noise} 不一致，"
                         "agent 自查与评估的标准不同")
    for name, g in gates.items():
        lines.append(f"- {name}: " + ", ".join(f"{e} {'PASS' if v['pass'] else 'FAIL'}" for e, v in g["engines"].items()))
    lines.append("- 排除的计时：" + (", ".join(f"{e} {why} ×{n}" for (e, why), n in sorted(excluded.items())) or "无"))
    lines.append(f"- 分叉：共 {len(divs)} 处，候选劣于 E0 超出噪声 {sum(bool(d.get('invalidates')) for d in divs)} 处，"
                 f"E0 自身偏离 {sum(bool(d.get('e0_deviated')) for d in divs)} 处")
    lines.append("")

    usage_f = HERE / "runs" / "usage.json"
    if usage_f.exists():
        lines += ["## Agent 用量（来自 Codex 会话记录）", "",
                  "| 运行 | 时长 h | 模型请求 | 工具调用 | 工具时间占比 | 输入 token（缓存） | 输出 token | 思考 token "
                  "| 未接电源 min | 合盖 min | 有人操作 min | 非常温 min |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for u in json.loads(usage_f.read_text()):
            m = u.get("machine") or {}
            lines.append(f"| {u['run']} | {u['wall_s'] / 3600:.2f} | {u['requests']} | {u['tool_calls']} | "
                         f"{u['tool_time_s'] / max(u['wall_s'], 1):.0%} | {u['input_tokens']:,} ({u['cached_input_tokens']:,}) | "
                         f"{u['output_tokens']:,} | {u['reasoning_output_tokens']:,} | {m.get('minutes_not_on_ac', '—')} | "
                         f"{m.get('minutes_lid_closed', '—')} | {m.get('minutes_user_active', '—')} | "
                         f"{m.get('minutes_thermal_not_nominal', '—')} |")
        lines += ["", "模型请求数按 token_count 事件计，是近似值。工具时间是工具调用从发出到返回的时间之和；"
                      "长命令可能被拆成多次轮询，所以会偏低。其余时间（模型时间）包括推理、流式输出和网络，"
                      "不等于纯思考时间。“有人操作”指 HID 空闲不到 2 分钟的分钟数。", ""]
    (res / "nav_report.md").write_text("\n".join(lines))
    (res / "nav_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
