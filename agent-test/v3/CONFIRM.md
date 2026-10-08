# 三引擎前瞻验证方案（2026-10-07）

目标是估计这台 M4、Qwen3-0.6B、BF16、batch=1、固定长度贪心生成条件下的性能差异。
不是模型质量非劣检验，不是并发服务吞吐测试，也不能代表其他硬件。

- E0：直接调用官方 mlx-lm `generate_step`，保持默认生成路径；E1 不进入主比较。
- E2：agent-test/v2 的 agent-engine。
- V0：官方 vLLM 0.30.0 + vllm-metal 0.30.0，离线 `LLMEngine` API。
  Metal 插件使用 MLX 后端，不是 CUDA vLLM，也不是另一个项目 vllm-mlx。
  明确使用 BF16、单请求、4096 分块、禁用前缀缓存和推测解码、关闭文本反解码。Metal 内存比例固定 0.40，限制默认约 15 GiB 的 KV 预留；记录实际分配，要求足够最大长度。
  这测量的是各引擎从 token IDs 到 token IDs 的完整路径，包含各自调度开销。

全部计时、gate、分析用 `.venv-vllm-metal/bin/python`，统一 Python 3.12、MLX 0.32.1
（官方预编译扩展的兼容版本）、mlx-lm 0.32.0。原 `.venv` 保留。
模型必须来自同一已缓存 snapshot，记录权重 SHA256、版本、源码 SHA256 和计划。
新目录与历史 results-simple 分开；两个时段之间版本、权重和方案改变即拒绝续跑。

每场景固定 24 个配对，分到两个时段各 12 个，随机/text 各半。三引擎在每对内随机顺序，
独立进程，**每个引擎都单独检查热状态**。预热完整 prompt 长度（包括 32k）和 8 个生成 token，
使用不同内容，完成预热后再冷却。固定生成数、忽略 EOS；模型加载、预热、冷却不计入请求耗时。
TTFT 在首个可用 token 时打点，decode 为排除前两个间隔后的 token 间隔中位数，
request 为全部 token 生成并同步后的完成时间。V0 若一次返回多个 token 则拒绝 per-token 计时。

先做 E0/E2 FP32 数值筛查；V0 在 4 个独立 random/text 短中上下文 fixture 上要求
与 E0 的 32 个生成 token 相同，或首次分叉的 FP32 logit 差小于
同环境 E0 筛查预先产生的 p99 margin 噪声阈值，失败停止。正式生成的首次分叉也按此规则审查。
这只筛查首次分叉，不保证后续自回归序列完全等价，不得按结果自动放宽阈值。
**分叉判定改为有方向（采集前登记，2026-10-07 20:25）：** 只有当被测引擎（E2 或 V0）选的 token 在 FP32 下比 E0 的选择低、且差距超过阈值时，才判作废或 gate 失败；如果引擎选的是 FP32 更认可的 token，记为"E0 偏离"，不影响判定。起因：第二次 smoke 中，4k 随机 prompt 的第 9 个 token 上，E2 与 vLLM 都选了 FP32 的首选（29.564），只有 mlx-lm 选了次选（29.312），差距 0.252 > 0.224；按原来无方向的规则，E2 会因为基线自身的 BF16 误差被判作废。
这些都是经验筛查，不足以证明模型任务质量等价或所有上下文正确。

主假设提前固定为 E0/E2 配对时间比的中位数 > 1.02；单侧精确符号检验，按既定指标族 Holm。
判定"更快"需同时满足：Holm 校正后 p < 0.05；两个时段都有数据；两个时段各自的几何平均比都 > 1（跨时段方向一致）。
GM 和 bootstrap 区间仅描述，不参与判定。只完成第一个时段时，报告标注"等待第二个时段"，不作判定。V0/E0、V0/E2 均列出描述性时间比，不作显著性判定，
所以不会通过增加对照组暗中增加主检验的假阳性机会。
24 对不代表能检测 2% 的差异：功效取决于超过 1.02 的配对比例及噪声；不显著写“未定”。
旧方案部分场景只有 6 对，即使全部同向，原始最小 p=1/64，7 项 Holm 之后也无法达到 0.05。
新的正式判定要求收齐计划配对和两个冻结一致的时段。失配 prompt、模型、输出数及重复记录拒绝分析。
**重试规则（采集前登记）：** 每个时段所有单元跑完后、分叉判定和分析之前，被排除的单元（门控超时、worker 崩溃）各重跑一次；
首次尝试的记录移到 `runs/retried/` 保留，不进入分析，报告给出重试数。重试仍被排除的单元保持排除，该场景按"样本不足"处理。
重试发生在时段末尾，与原区组顺序不同，这一点在报告中如实说明。
**电源与睡眠（采集前登记）：** 每次计时前后记录电源状态与内核的 sleep/wake 时间戳；计时期间不在插电状态、或系统睡眠过（含合盖后的 DarkWake），该次计时排除（再按上条重试一次）。调度器在每个单元前检查电源，用电池时暂停等待，不在电池模式下计时。
起因：2026-10-07 的 confirm smoke 在 14:43 合盖并拔电后，计时阶段一直处于合盖睡眠与 DarkWake 循环中，32k 的 TTFT 因此慢了约 3 倍，而 thermalState 始终为 0、门控照常通过。该 smoke 的计时数值全部无效，只证明流程可以跑通。

**运行条件：** 全程插电；不要合盖（除非接外接显示器）；保持登录。按未睡眠时的耗时估算，每个时段约 4–5 小时（32k 每个单元约 3–4 分钟，共 36 个），两个时段间隔 2 小时，从 01:00 开始约在次日中午前后结束。

检验仍以给定时段内配对独立为条件；热状态门槛和随机顺序不能证明独立。
凌晨开始不是科学控制变量。两小时休息只是两个时段，不能写成跨日复现；
更强的跨日结论需另一天重复，并以独立日期为推断单位。

先运行短 smoke（不是证据）：

```sh
HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
.venv-vllm-metal/bin/python agent-test/v3/run.py --profile confirm --smoke \
  --out agent-test/v3/results-smoke-confirm
```

实际启动：2026-10-07 晚上由 launchd 一次性任务（加载即运行，`~/Library/LaunchAgents/com.roofline.v3-now.plist`）立即运行下面"同夜两个时段"的命令（`GAP_HOURS=2`），原定的 01:00 定时任务已撤销；日志 `~/Library/Logs/roofline-v3-run.log`。开始时间不是科学控制变量。
手动运行第一时段的等价命令：

```sh
.venv-vllm-metal/bin/python agent-test/v3/run.py --profile confirm --session 1 \
  --out agent-test/v3/results-confirm-20261008
```

第二天同环境、同目录运行 `--session 2`。如只需要同夜两个时段：

```sh
BENCH_PYTHON=.venv-vllm-metal/bin/python \
BENCH_OUT=agent-test/v3/results-confirm-20261008 GAP_HOURS=2 \
agent-test/v3/run_both.sh --profile confirm
```

总计 648 次正式请求，加完整形状预热和逐次冷却；不能沿用 simple 的“两小时完成”估计。
结果输出 `report-corrected.md` 和 `summary-corrected.json`；原始 JSON 全部保留。

参考：[官方安装](https://docs.vllm.ai/projects/vllm-metal/en/stable/installation/)，
[官方 Metal 后端](https://docs.vllm.ai/projects/vllm-metal/en/stable/)。
