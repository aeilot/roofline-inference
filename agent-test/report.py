"""Render paired measurements, including uncertainty and regressions."""
import json
import math
import random
import statistics
from pathlib import Path


def interval(ratios):
    rng = random.Random(0)
    logs = [math.log(x) for x in ratios]
    samples = sorted(math.exp(statistics.median(rng.choices(logs, k=len(logs))))
                     for _ in range(10000))
    return samples[249], samples[9749]


def main():
    lines = ["# 最终默认配置的受控对照", "",
             "每对在同一进程内对照，共享 BF16 权重和 GPU stream，相同 token 输入、greedy、生成长度、"
             "KVCache、chunk 大小和逐块分区；AB/BA 交错。默认只启用 decode 投影融合。", "",
             "单位 ms；加速比是 baseline / agent，大于 1 表示 agent 更快。"
             "括号中是逐对加速比中位数的配对 bootstrap 95% 区间；"
             "不足四对时只列观测范围。重复数小、温度未锁定，区间仅反映这些观测样本的波动。", "",
             "| Prompt | 对数 | TTFT mlx-lm → agent | TTFT 配对加速比 [95%] | Decode mlx-lm → agent | Decode 配对加速比 [95%] |",
             "| --- | --- | --- | --- | --- | --- |"]
    for path in [Path("agent-test/results.json"), Path("agent-test/results-long.json")]:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for summary in data["summary"]:
            L = summary["length"]
            rows = [r for r in data["runs"] if r["length"] == L]
            if not all(r["tokens_equal"] for r in rows):
                raise AssertionError("Do not report invalid-output performance")
            cells = [str(L), str(len(rows))]
            for metric in ("ttft_s", "decode_s_median"):
                stats = summary[metric]
                ratios = [r["measurements"]["mlx-lm"][metric] /
                          r["measurements"]["agent"][metric] for r in rows]
                lo, hi = interval(ratios) if len(rows) >= 4 else (min(ratios), max(ratios))
                label = "" if len(rows) >= 4 else "观测范围 "
                cells.extend([f"{stats['baseline']*1000:.2f} → {stats['agent']*1000:.2f}",
                              f"{stats['paired_speedup_median']:.3f} [{label}{lo:.3f}, {hi:.3f}]"])
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "每组生成 token 必须完全一致，12 组固定分块 logits 检查必须逐位相同。"
              "原始 JSON 保留所有计时、顺序、负载、热状态信息和源码校验值。", "",
              "仅凭超过 1 的一个点估计不能宣称稳定收益；区间覆盖 1 的点视作未证实提升。"
              "长上下文只测两对，用于覆盖验证，不用于宣称稳定提升。"
              "不能把历史原始基线或未通过数值检查的 tail merge 数据混入这个表。", ""]
    Path("agent-test/report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
