# Qwen3 BF16 单流推理引擎实验

按 `../docs/design-notes.md` 的 A 级约束实现：Apple M4 GPU、Qwen3-0.6B、BF16 权重与 KV、batch=1。自有 transformer forward 与异步 greedy 调度，复用 MLX 算子及 mlx-lm 的权重加载、RoPE、KVCache。不是从零重写 Metal GEMM，也不调用 mlx-lm 的 generate 来运行新引擎。

最终默认配置的数据汇总见 [report.md](report.md)，原始记录见 `results.json` 与 `results-long.json`。当前实测尚未证实稳定超过 mlx-lm，不能宣称全面更快。

## 运行

在仓库根目录运行，使用现有 `.venv`：

```sh
.venv/bin/python agent-test/engine.py --prompt 'Explain the roofline model in two sentences.' --max-tokens 128
.venv/bin/python -m pytest -q agent-test/test_engine.py
.venv/bin/python agent-test/benchmark.py --lengths 16 64 512 1024 4096 --gen 65 --repeats 6 --cooldown 2 --out agent-test/results.json
python3 agent-test/report.py
```

需要 macOS Metal GPU 访问权限。当前验证环境 MLX 0.32.3、mlx-lm 0.32.0。模型使用同一次 `load("Qwen/Qwen3-0.6B")` 加载的权重对象。

## 实现与优化变量

- **Decode 投影合并**：Q/K/V 三次 GEMV 合成一次，gate/up 两次合成一次；每层减少三次矩阵乘调用。prefill 保留原投影，因为拼接 GEMM 和保持 M/N/K 的 batched GEMM 均未通过严格的数值检查；后者在 16-token 测试中最大 logit 差为 0.2734375，已撤回。
- **可选残差 + RMSNorm 融合**（`--fused-norm`）：每层 attention 残差与 post norm 合并；MLP 残差与下一层 input norm 合并，共 55 处。自定义 Metal kernel 保留 BF16 残差舍入、float32 累加，以及 norm 乘权重前的 BF16 舍入。1024 以外的维度自动使用原算子。
- **实验性 Prefill 尾部合并**（`--merge-prompt-tail`，默认关闭）：mlx-lm 将最后一个 prompt token 留作独立前向；引擎将它留在最后一个 prefill 块内，只对最后一行计算词表投影，减少一次模型前向和同步。两边 `prefill_step_size=2048` 相同；实际尾块长度是引擎调度的处理变量，JSON 中分别记录。默认与基线逐块长度也相同。尾部合并在 L=512 的生成序列检查失败，不能把其计时当合格加速结果。
- **Token-only greedy**：直接取 logits argmax，省掉不向调用方输出的全词表 logprobs；保留同样的一步异步 lookahead，包括最后一个在途 token。
- **Allocator 调度**：中间 prefill 块只求值 KV，立即丢弃无用 hidden graph，避免保留旧 KV 版本；不在每块调用 `mx.clear_cache()`。

不量化，不改 KV 精度，不做投机、batching 或近似 attention。仅支持 dense Qwen3、batch 1、greedy；不是通用 serving 引擎。packed 投影需要额外权重内存，JSON 记录其字节数。

## 控制变量

默认实验唯一处理变量是引擎实现（包括内部融合和调度），prefill 逐块分区也完全相同。固定设备、进程、共享权重、dtype、输入 token、prompt 长度、生成长度、chunk 大小、采样、GPU stream、KVCache 类型及 lookahead 数量。

每个长度先各自预热，然后六组交错 AB/BA；每次开始前同步 GPU、清空 allocator 缓存、相同等待时间。输入由固定 seed=10 产生并记录 SHA256；每组要求生成 token 序列完全一致，否则保存原始数据并终止。不会拿旧 `results/raw/` 的历史基线作加速比分母。

主实验分两次运行完成；`results.json` 的 `phases` 保留各次时间与长度，合并前核对源码校验值和所有公共参数完全相同。每一对的基线与 agent 均在同一进程内运行。

输出包含每次 TTFT、所有 decode 间隔、最后一个 token 的返回时间、同步完最后在途工作的完成时间、峰值 MLX 内存、系统负载、`pmset -g therm` 输出、版本和源码 SHA256。decode 按仓库 baseline 的约定剔除最初两个间隔；汇总同时报告中位数之比和逐对加速比的中位数/范围。

**热状态没有锁定。** 相同等待与交错顺序只能减轻漂移，无法保证实际温度、频率或后台负载完全相同。`pmset` 的“没有记录警告”也不能证明没有降频。原始波动必须与结果一起看，不把一次最快值当稳定提升。

基线是安装版本原生 `mlx_lm.generate.generate_step`，两边共同输出 token ID，均不计 detokenizer、终端打印或加载时间。因此与仓库的 `stream_generate` 基线绝对数值不是同一计时边界。基线提供 logprobs 的额外工作属于 API 实现差异；若要提供 logprobs，新引擎也必须付出相应成本。

## 验证与消融

`test_engine.py` 检查：融合 kernel 的逐位一致性、12 组全位置 logits、分块 causal mask、KV 扩容边界、真实中英文/代码 prompt 的 greedy 序列、无效输入、零生成长度和 EOS，以及 prefill 图和 CLI detokenizer 生命周期回归检查。已执行 12 项测试通过。测试覆盖范围内的逐位一致性不构成任意 prompt 的全局证明。

```sh
# 消融 decode 投影融合 / Metal residual norm 融合
.venv/bin/python agent-test/benchmark.py --no-pack --out agent-test/results-no-pack.json
.venv/bin/python agent-test/benchmark.py --fused-norm --out agent-test/results-fused-norm.json
# 长上下文保持两边各项设置相同；成本较高
.venv/bin/python agent-test/benchmark.py --lengths 16384 32768 --gen 65 --repeats 2 --out agent-test/results-long.json
```

`results-strict-before-tail.json` 为启用两个融合的六组正式对照（精确 L-1+1 分区），包含变慢的点。默认只开启 decode 投影融合，归一化融合的消融未显示稳定收益，因此改为可选。`results-tail-rejected.json` 保存了尾部合并在 512-token prompt 上生成序列不一致的失败实验，不能用其更快的点证明合格加速。`results-before-prefill-release.json` 和 `results-long-before-prefill-release.json` 记录释放无用 hidden graph 前的版本，后者未完成 32k；均不属于最终结果。`pilot.json` 为早期、不完整预实验；`tuning.json` 来自 `tune.py` 的同步、teacher-forced forward 消融，其计时方式与生成实验不同，只用于诊断。

规范化 BF16 的舍入顺序参考 [MLX RMSNorm Metal 实现](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal)。相关 Apple MIT 许可保存在 `LICENSE-MLX` 和 `LICENSE-MLX-LM`。

最后修正了 CLI 的流式显示：`tokenizer.detokenizer` 每次访问返回副本，因此必须保留一个实例。实验使用的原源码保存在 `benchmarked-engine.py`，SHA256 与 JSON 记录相符；当前 `Engine` 类的 AST 与快照完全一致，仅 CLI 显示部分不同。
