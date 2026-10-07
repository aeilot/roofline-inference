"""Pre-registered v3 analysis (PLAN.md sections 1 and 7). Frozen before data collection.

Primary: E2 vs E0, regime idle. For each scenario and metric, the statistic is
the mean paired log ratio log(t_E0 / t_E2); 95% CI by bootstrap stratified by
session (10 000 resamples, fixed seed); geometric-mean ratio = exp(mean).
One-sided p for "GM > 1.02" = share of bootstrap means <= log 1.02; Holm
correction within each family (TTFT, decode, request) per regime.
Decision (only with >= 6 pairs in total and >= 3 per session present; otherwise
"insufficient": a bootstrap over 1-2 pairs has no meaningful spread):
  invalid    any E2-vs-E0 generated-sequence divergence that is not an FP32 near-tie
  faster     Holm-adjusted p < 0.05, and every session's own estimate > 1
  equivalent 95% CI inside [0.98, 1.02]
  undecided  otherwise
Secondary (no decisions): E1 vs E0 (scheduling), E2 vs E1 (kernels), regime warm
for the same rules, ablations (t_x / t_E2), carryover (prev=E0 vs prev=E2).
Exclusions: only gate timeouts and worker crashes; all counted in the report.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

B = 10_000
BOOT_SEED = 7
MARGIN = math.log(1.02)
MIN_PAIRS, MIN_PER_SESSION = 6, 3
FAMILIES = {
    "TTFT": ("ttft_s", ["ttft16", "ttft64", "ttft256", "ttft1024", "ctx4096", "ctx16384", "ctx32768"]),
    "decode": ("decode_s_median", ["ctx512", "ctx4096", "ctx16384", "ctx32768"]),
    "request": ("request_completed_s", ["req256x256", "req1024x512", "req4096x128"]),
}
PLANNED = {"ttft16": 40, "ttft64": 40, "ttft256": 30, "ttft1024": 12, "ctx512": 12, "ctx4096": 12,
           "ctx16384": 6, "ctx32768": 6, "req256x256": 12, "req1024x512": 12, "req4096x128": 12}


def boot(logr_by_session):
    """Stratified bootstrap of the mean log ratio. Returns (mean, lo, hi, p_le_margin, per-session means)."""
    rng = np.random.default_rng(BOOT_SEED)
    strata = [np.array(v) for v in logr_by_session.values() if len(v)]
    allv = np.concatenate(strata)
    means = np.zeros(B)
    n = len(allv)
    for s in strata:
        idx = rng.integers(0, len(s), size=(B, len(s)))
        means += s[idx].sum(axis=1)
    means /= n
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = (1 + np.sum(means <= MARGIN)) / (B + 1)
    return float(allv.mean()), float(lo), float(hi), float(p), {k: float(np.mean(v)) for k, v in logr_by_session.items() if v}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def load(results):
    runs, excluded = [], []
    for f in sorted((results / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        u = r.get("unit") or {k: r.get(k) for k in ("phase", "regime", "scenario", "engine", "session")}
        (excluded if r.get("excluded") else runs).append(dict(r, unit=u))
    return runs, excluded


def paired(runs, phase, regime, a, b, metric, scenario, key_extra=None):
    """log(t_a / t_b) per session, pairing by (session, block)."""
    t = defaultdict(dict)
    for r in runs:
        u = r["unit"]
        if u["phase"] == phase and u["regime"] == regime and u["scenario"] == scenario and r.get(metric):
            if key_extra and not key_extra(u):
                continue
            t[(u["session"], u["block"])][u["engine"] if not u.get("role") else u["prev"]] = r[metric]
    out = defaultdict(list)
    for (s, _), d in t.items():
        if a in d and b in d:
            out[f"s{s}"].append(math.log(d[a] / d[b]))
    return dict(out)


def fmt_ci(m, lo, hi):
    return f"{math.exp(m):.3f} [{math.exp(lo):.3f}, {math.exp(hi):.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    results = Path(a.results)
    runs, excluded = load(results)
    plan = json.loads((results / "plan.json").read_text()) if (results / "plan.json").exists() else {}
    planned = plan.get("pairs", PLANNED)
    divs = json.loads((results / "divergence_records.json").read_text()) if (results / "divergence_records.json").exists() else []
    manifests = [json.loads(f.read_text()) for f in sorted(results.glob("manifest-s*.json"))]
    gates = {f.stem: json.loads(f.read_text()) for f in sorted(results.glob("gate-s*.json"))}
    sessions = sorted({r["unit"]["session"] for r in runs})
    failed = sorted({e for g in gates.values() for e, v in g["engines"].items() if not v["pass"]})
    all_runs = runs        # before dropping gate-failed engines (used for the exclusion audit)
    runs = [r for r in runs if r["unit"]["engine"] not in failed or r["unit"]["engine"] in ("E0", "E2")]
    summary = dict(sessions=sessions, primary={}, secondary={}, ablation={}, carryover={},
                   exclusions={}, divergences=dict(total=len(divs), not_near_tie=sum(not d["near_tie"] for d in divs)))

    lines = ["# v3 验证结果", "",
             f"方案：{plan.get('profile', 'full')}；时段：{sessions}；有效计时 {len(runs)} 次，排除 {len(excluded)} 次。由 analyze.py 自动生成（判据见 PLAN.md 第 1、7 节）。", ""]
    if failed:
        lines += [f"> 未通过正确性门槛、已从分析中剔除的引擎：{failed}" + ("（E0/E2 不可剔除，结论无效）" if {"E0", "E2"} & set(failed) else ""), ""]
    if len(sessions) < 2:
        lines += ["> **只有一个时段的数据：** 按方案，结论需两个时段方向一致；当前所有\"更快\"均为暂定。", ""]
    if manifests and len({json.dumps(m["files"], sort_keys=True) for m in manifests}) > 1:
        lines += ["> **警告：两个时段的被测代码不一致（冻结失败）。**", ""]
    if any(m.get("git_dirty") for m in manifests):
        lines += ["> 注意：采集时 v2/v3 目录有未提交的修改（manifest 中 git_dirty=true）。", ""]

    for regime in ("idle", "warm"):
        label = "主分析（idle）" if regime == "idle" else "补充（warm）"
        lines += [f"## E2 vs E0 — {label}", "",
                  "| 族 | 场景 | 对数（计划） | 几何平均比 [95% CI] | 各时段 | Holm p | 结论 |", "|---|---|---|---|---|---|---|"]
        for fam, (metric, sids) in FAMILIES.items():
            rows = []
            for sid in sids:
                lr = paired(runs, "main", regime, "E0", "E2", metric, sid)
                if not lr:
                    continue
                rows.append((sid, lr, *boot(lr)))
            adj = holm([r[5] for r in rows]) if rows else []
            for (sid, lr, m, lo, hi, p, per), padj in zip(rows, adj):
                bad = any(d["regime"] == regime and d["scenario"] == sid and d["engine"] == "E2" and not d["near_tie"] for d in divs)
                n = sum(len(v) for v in lr.values())
                if bad:
                    verdict = "作废（非近似平局的分叉）"
                elif n < MIN_PAIRS or min(len(v) for v in lr.values()) < MIN_PER_SESSION:
                    verdict = "样本不足"
                elif padj < 0.05 and all(v > 0 for v in per.values()):
                    verdict = "更快" + ("（单时段，暂定）" if len(per) < 2 else "")
                elif math.exp(lo) >= 0.98 and math.exp(hi) <= 1.02:
                    verdict = "等价"
                else:
                    verdict = "未定"
                per_s = ", ".join(f"{k} {math.exp(v):.3f}" for k, v in sorted(per.items()))
                lines.append(f"| {fam} | {sid} | {n} ({planned.get(sid, "—")}) | {fmt_ci(m, lo, hi)} | {per_s} | {padj:.4f} | {verdict} |")
                summary["primary" if regime == "idle" else "secondary"].setdefault(regime, {})[f"{fam}/{sid}"] = dict(
                    n=n, gm=math.exp(m), ci=[math.exp(lo), math.exp(hi)], per_session={k: math.exp(v) for k, v in per.items()},
                    p_holm=padj, verdict=verdict)
        lines.append("")

    lines += ["## 收益来自哪里（idle，次要分析，不作判定）", "",
              "E0/E1 > 1 表示公平调度本身带来的加速；E1/E2 > 1 表示 v2 kernel 在相同调度下的加速。", "",
              "| 族 | 场景 | E0/E1 [95% CI] | E1/E2 [95% CI] |", "|---|---|---|---|"]
    for fam, (metric, sids) in FAMILIES.items():
        for sid in sids:
            l1 = paired(runs, "main", "idle", "E0", "E1", metric, sid)
            l2 = paired(runs, "main", "idle", "E1", "E2", metric, sid)
            if l1 and l2:
                b1, b2 = boot(l1), boot(l2)
                lines.append(f"| {fam} | {sid} | {fmt_ci(*b1[:3])} | {fmt_ci(*b2[:3])} |")
                summary["secondary"].setdefault("attribution", {})[f"{fam}/{sid}"] = dict(
                    e0_over_e1=math.exp(b1[0]), e1_over_e2=math.exp(b2[0]))
    lines.append("")

    lines += ["## 消融（idle）：去掉一项优化后变慢多少", "", "比值 = t(去掉该项) / t(E2)，> 1 表示该项有贡献。", "",
              "| 场景 | 指标 | " + " | ".join(e for e in ["E2-noskinny", "E2-nogemv", "E2-nogqa", "E2-noprefuse"]) + " |",
              "|---|---|---|---|---|---|"]
    for sid in ["ttft64", "ttft1024", "ctx512", "ctx4096"]:
        for metric in (["ttft_s"] + (["decode_s_median"] if sid.startswith("ctx") else [])):
            cells = []
            for x in ["E2-noskinny", "E2-nogemv", "E2-nogqa", "E2-noprefuse"]:
                lr = paired(runs, "abl", "idle", x, "E2", metric, sid)
                if lr:
                    bb = boot(lr)
                    cells.append(fmt_ci(*bb[:3]))
                    summary["ablation"][f"{sid}/{metric}/{x}"] = dict(gm=math.exp(bb[0]), ci=[math.exp(bb[1]), math.exp(bb[2])])
                else:
                    cells.append("—")
            if any(c != "—" for c in cells):
                lines.append(f"| {sid} | {metric} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## 延续性检验（进程隔离是否足够）", "",
              "同一 prompt 上测目标引擎，比较紧挨着的前一个进程是 E0 还是 E2：比值 = t(前为 E0) / t(前为 E2)。"
              "两个方向都测（目标 E2、目标 E0）。判断：区间不含 1 → 检出影响；区间落在 [0.98, 1.02] 内 → 无可测影响；"
              "否则无法排除，并给出本样本量下可检出的最小效应（区间半宽）。", "",
              "| 目标 | 场景 | 指标 | 比值 [95% CI] | 可检出的最小效应 | 判断 |", "|---|---|---|---|---|---|"]
    for target in ("E2", "E0"):
        for sid in ["ttft64", "ctx4096"]:
            for metric in (["ttft_s"] + (["decode_s_median"] if sid.startswith("ctx") else [])):
                lr = paired(runs, "carry", "idle", "E0", "E2", metric, sid,
                            key_extra=lambda u, t=target: u.get("role") == "probe" and u["engine"] == t)
                if not lr:
                    continue
                m, lo, hi, _, _ = boot(lr)
                n = sum(len(v) for v in lr.values())
                mde = (math.exp(hi) - math.exp(lo)) / 2
                if n < MIN_PAIRS:
                    judge = "样本不足"
                elif lo > 0 or hi < 0:
                    judge = "检出影响"
                elif lo >= -MARGIN and hi <= MARGIN:
                    judge = "无可测影响"
                else:
                    judge = f"无法排除（小于 ±{mde:.1%} 的影响检不出）"
                lines.append(f"| {target} | {sid} | {metric} | {fmt_ci(m, lo, hi)} (n={n}) | ±{mde:.1%} | {judge} |")
                summary["carryover"][f"{target}/{sid}/{metric}"] = dict(
                    n=n, gm=math.exp(m), ci=[math.exp(lo), math.exp(hi)], mde=mde, verdict=judge)
    lines.append("")

    # Are exclusions independent of the engine? (main phase; permutation test of the chi-square statistic)
    main_all = [r for r in all_runs + excluded if r["unit"].get("phase") == "main"]
    engines = sorted({r["unit"]["engine"] for r in main_all})
    if main_all and excluded:
        lab = np.array([engines.index(r["unit"]["engine"]) for r in main_all])
        exc = np.array([bool(r.get("excluded")) for r in main_all])

        def chi2(x):
            tot = 0.0
            for e in range(len(engines)):
                sel = lab == e
                for val in (True, False):
                    obs = np.sum(x[sel] == val)
                    exp_ = sel.sum() * np.mean(x == val)
                    tot += (obs - exp_) ** 2 / exp_ if exp_ else 0.0
            return tot

        rng = np.random.default_rng(BOOT_SEED)
        stat = chi2(exc)
        perm = np.array([chi2(rng.permutation(exc)) for _ in range(5000)])
        p_exc = float((1 + np.sum(perm >= stat)) / (1 + len(perm)))
    else:
        p_exc = None
    by_engine = {e: (sum(1 for r in main_all if r["unit"]["engine"] == e and r.get("excluded")),
                     sum(1 for r in main_all if r["unit"]["engine"] == e)) for e in engines}
    summary["exclusions_by_engine"] = dict(counts=by_engine, permutation_p=p_exc)
    lines += ["## 排除记录与引擎的关系（主实验）", "",
              "| 引擎 | 排除 / 总数 |", "|---|---|"] + [f"| {e} | {a} / {b} |" for e, (a, b) in by_engine.items()] + [
              "", ("没有排除记录。" if p_exc is None else
                   f"排除与引擎独立性的置换检验（χ² 统计量，5000 次）：p = {p_exc:.3f}。"
                   + (" **p < 0.05：排除可能与引擎相关，主结论需要结合排除记录解读。**" if p_exc < 0.05 else "")), ""]

    reasons = defaultdict(int)
    for r in excluded:
        reasons[(r["unit"].get("phase"), r.get("excluded"))] += 1
    summary["exclusions"] = {f"{k[0]}/{k[1]}": v for k, v in reasons.items()}
    lines += ["## 正确性与数据质量", "",
              "门槛（每个时段一次）：" + "; ".join(
                  f"{k}: " + ", ".join(f"{e}={'通过' if v['pass'] else '未通过'}" for e, v in g["engines"].items())
                  for k, g in gates.items()), "",
              f"生成序列分叉：{len(divs)} 处，其中非 FP32 近似平局 {summary['divergences']['not_near_tie']} 处"
              f"（阈值 = E0 的 p99 margin 噪声）。", "",
              "排除记录：" + (", ".join(f"{k[0]}/{k[1]}: {v}" for k, v in reasons.items()) or "无"), "",
              "计时后的热状态分布：" + ", ".join(f"{s}: {n} 次" for s, n in sorted(
                  {str(k): sum(1 for r in runs if r.get("thermal_after") == k)
                   for k in {r.get("thermal_after") for r in runs}}.items())), ""]
    if manifests:
        m0 = manifests[0]
        lines += [f"环境：git {m0['git_head'][:10]}，MLX {m0['mlx']}，mlx-lm {m0['mlx_lm']}，macOS {m0['macos']}，"
                  f"AC 电源 {[m['on_ac_power'] for m in manifests]}，探针基准 {[round(m.get('probe_ref_ms') or 0, 2) for m in manifests]} ms。", ""]
    (results / "report.md").write_text("\n".join(lines))
    (results / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
