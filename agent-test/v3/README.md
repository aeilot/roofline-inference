# v3 自动化验证：使用说明

方案与判据见 [PLAN.md](PLAN.md)，采集后修订见第 16 节。原冻结脚本为 analyze_preregistered.py；已有数据的修订检验是探索性的，新的独立采集需要先冻结新方案。整个流程无人值守、可以中断后续跑，结束时自动生成报告。

## 运行前

1. **接上电源。** 电池供电时性能策略不同，脚本会拒绝运行（确有需要可加 `--allow-battery`，报告中会记录）。
2. **关掉其他程序**，尤其是浏览器、IDE 的索引、视频等。脚本只记录系统负载，无法控制别的进程。
3. **合盖可以，但不要让机器睡眠。** 脚本会自动调用 `caffeinate` 保持唤醒。
4. **代码保持冻结。** 第一个时段开始时会记录所有被测文件（含语料快照 `corpus/`）的 SHA256；第二个时段如果发现有变化，会拒绝运行。编辑仓库里的文档不受影响：文本 prompt 来自快照，见 PLAN.md 第 12 节。

## 运行

在仓库根目录执行：

```sh
.venv/bin/python agent-test/v3/run.py --session 1
```

隔一天（换一个时段）再执行：

```sh
.venv/bin/python agent-test/v3/run.py --session 2
```

- 默认使用 **quick 配置**：每个时段 269 次计时，**预计约 2–2.5 小时**。与完整方案的区别见 [PLAN.md 第 11 节](PLAN.md#11-quick-配置默认)。
- 完整方案（每个时段约 10 小时，绝大部分时间在等冷却）：在命令后加 `--profile full`。两种配置的结果分别保存在 `results-quick/` 和 `results-full/`。
- **随时可以中断**（Ctrl-C、断电、重启后）：重新执行同一条命令，已完成的单元会跳过，从断点继续。
- 进度和预计剩余时间写在 `agent-test/v3/results-quick/log.txt`，可以用下面的命令查看：

```sh
tail -f agent-test/v3/results-quick/log.txt
```

## 结果

也可以一条命令跑完两个时段：先跑第一个，在它结束 2 小时后自动跑第二个，等待期间保持唤醒。中断后重新执行会从断点继续，等待时间按第一个时段记录的结束时间计算。

```sh
agent-test/v3/run_both.sh
```

间隔可以用环境变量调整，例如 `GAP_HOURS=12 agent-test/v3/run_both.sh`。注意：同一天、间隔几小时的两个时段，复现条件弱于分在两天（见 PLAN.md 第 8 节）。

两个时段都跑完后：

- `results-quick/report-corrected.md`：修订分析（条件性中位数检验／未定／作废）、边际消融、延续性描述、数据质量；
- `results-quick/summary-corrected.json`：同样的内容，机器可读；原 report.md / summary.json 保留。
- `results-quick/runs/*.json`：每一次计时的原始记录（包括被排除的）；
- `results-quick/manifest-s*.json`：环境、代码哈希、探针基准；
- `results-quick/gate-s*.json`：正确性门槛的结果；
- `results-quick/divergence_records.json`：每一处生成序列分叉及其 FP32 判定。

只跑完一个时段时也会生成描述报告，但不作跨时段的“更快”判定。

## 流程

```
run.py（父进程，不加载模型）
 ├─ 前置检查：电源、冻结、caffeinate
 ├─ 探针校准：等冷却后测一个固定矩阵乘负载，作为本时段的基准
 ├─ gate.py：单元测试 + 全部 7 个引擎对 FP32 的数值检查（E2 未通过则停止）
 ├─ 逐个单元启动 worker.py（每次一个新进程）：
 │    加载模型 → 预热 → 等 thermalState=0 且探针恢复 → idle 空等 2 s / warm 预热负载 → 计时 → 同步引擎的 stream → 写 JSON
 │    顺序：main/idle → main/warm → 延续性检验 → 消融；每个区组内随机，种子固定
 ├─ divergence.py：对生成序列的每一处分叉，用 FP32 前向判断是否为近似平局
 └─ analyze.py：修订分析，生成 report-corrected.md 与 summary-corrected.json
```

冒烟测试（约 30–40 分钟，输出到 `results-smoke-quick/`，不进入正式分析）：

```sh
.venv/bin/python agent-test/v3/run.py --smoke
```

## 新的三组验证

下一次采集使用 `confirm`；原生 mlx-lm / agent-engine / 官方 vLLM Metal。
方案、解释边界和启动命令见 [CONFIRM.md](CONFIRM.md)。不得复用历史 results-simple。
