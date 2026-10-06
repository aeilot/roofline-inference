# v2：Qwen3-0.6B BF16 单流引擎与 mlx-lm 的受控对照

2026-10-06 · MacBook Air M4 24 GB · MLX 0.32.3 / mlx-lm 0.32.0 · A 级（BF16 权重与 KV 不变，只改实现）

## 结论

1. **在测试范围内，v2 的 TTFT 和 decode 都快于 mlx-lm `generate_step` 的默认配置。** prompt 16–4096 各 6 对，配对加速比的 95% 区间全部高于 1；16k/32k 各只有 2 对，两对都更快，但不足以给出统计结论。
2. **只有短 prompt 的 TTFT 是明显提升**（1.26–1.72 倍）。decode 快 6–7%，中长 prompt 的 TTFT 快 4–12%。
3. **decode 的 A 级上界在这台机器上基本是紧的。** 上界是下界性质的论证，现在有一个实际实现达到了它的 96–99%（以校准带宽 102.7 GB/s 为准）。A 级下 decode 剩余空间约 1–4%。
4. **长 prompt prefill 与上界仍有约 20–30% 的差距，本轮没有缩小多少。** 后续实验（第 3 节）否定了"softmax 运算未计入上界"这一解释：差距来自 attention 的 MMA 与数据搬运路径（约为 MMA 峰值的 77–80%），加上持续负载下的降频（60 s 内 −12%）。
5. **本轮不能证明上界作为 agent 导航是必要的或有效的**：没有"不使用上界"的对照组（见第 6 节）。

适用范围：一台 M4、一个模型、batch 1、greedy、只输出 token ID；计时用的是随机 token prompt，真实文本只用在数值检查里。

## 1. 主要结果

| Prompt | 对数 | TTFT mlx-lm → v2 (ms) | TTFT 加速比 [95%] | Decode mlx-lm → v2 (ms/token) | Decode 加速比 [95%] | 整个请求 |
|---|---|---|---|---|---|---|
| 16 | 6 | 52.4 → 30.5 | **1.72** [1.42, 2.16] | 12.64 → 11.84 | 1.067 [1.055, 1.074] | 1.074 |
| 64 | 6 | 61.0 → 39.6 | **1.56** [1.46, 1.71] | 12.70 → 11.94 | 1.063 [1.056, 1.066] | 1.075 |
| 256 | 6 | 124.7 → 96.9 | **1.26** [1.25, 1.31] | 12.91 → 12.16 | 1.061 [1.050, 1.079] | 1.069 |
| 1024 | 6 | 389.7 → 347.9 | 1.12 [1.11, 1.13] | 13.82 → 12.91 | 1.069 [1.056, 1.072] | 1.078 |
| 4096 | 6 | 1938 → 1814 | 1.07 [1.04, 1.08] | 17.62 → 16.59 | 1.062 [1.059, 1.077] | 1.064 |
| 16384 | 2 | 15908 → 15364 | 1.04（两对：1.02, 1.05） | 32.13 → 30.78 | 1.044（1.04, 1.05） | 1.036 |
| 32768 | 2 | 64525 → 59805 | 1.08（两对：1.04, 1.11） | 52.48 → 50.49 | 1.040（1.03, 1.05） | 1.078 |

*热态（warm）：每次请求前等 `thermalState` 回到 0，再跑约 150 ms 固定 GPU 负载；AB/BA 交错。生成 129 个 token（16k/32k 为 65 个）。加速比为逐对比值的中位数，区间为配对 bootstrap。原始数据见 [results-warm.json](results-warm.json)、[results-long.json](results-long.json)，其中记录的源码 SHA256 与当前文件一致。*

空闲 2 s 后再发请求，结果方向相同（[results-idle.json](results-idle.json)）：TTFT 依次为 1.60 / 1.34 / 1.23 / 1.10 / 1.07 倍，decode 1.060–1.071 倍。其中 256 的区间很宽（1.004–1.60）。

**基线说明。** 基线是 mlx-lm 的默认路径：prefill 分块 2048，先算 L−1 个 token 再算最后 1 个，每块后调用 `clear_cache`；KV 缓存靠 concat 扩容；额外计算 logprobs。中长 prompt 的部分收益可能来自 v2 避开了这些调度选择，而不是 kernel 更好，本轮没有把两者分开。logprobs 的开销也没有单独测量，因此结论只适用于 token-only greedy。

## 2. 正确性标准与结果

第一轮要求逐位一致，这比 A 级定义更严：A 级允许改变求和顺序。v2 改用 [check.py](check.py) 的容差标准，因此**本轮与第一轮的正确性门槛不同，两轮结论不能直接比较**。

- 以同一组权重上转为 FP32 的模型作参考，在 FP32 自己的 greedy 序列上 teacher-forcing，比较两边真实生成路径的每个位置 logits。
- 中位 RMS 误差、逐位置最大误差 p99、FP32 top-2 差值误差 p99，都不得超过 mlx-lm 的 1.1 倍；与 FP32 top-1 的不一致不得显著更多（配对符号检验 p ≥ 0.05）。
- 自由生成的分叉，只允许出现在两个候选的 FP32 差距小于 mlx-lm 自身 p99 margin 噪声的位置。

结果（17 个 prompt、1632 个位置，[check.json](check.json)）：

| | mlx-lm | v2 |
|---|---|---|
| 中位 RMS 误差 | 0.0416 | 0.0411 |
| 最大误差 p99 | 0.536 | 0.489 |
| top-2 margin 误差 p99 | 0.250 | 0.201 |
| 与 FP32 top-1 不一致 | 8 | 7（符号检验 p = 0.77） |

主实验（热态与空闲态）的 60 对生成序列全部一致；32k 的两对在第 2 个生成 token 分叉，两个候选在 FP32 下只差 0.015（[div32k.json](div32k.json)）。26 项单元测试通过（[test_engine.py](test_engine.py)）。

容差的阈值（1.1 倍、p99）是本轮自定的，不是质量标准。它只能说明 v2 的数值误差不大于 mlx-lm 的 BF16 误差，不涉及下游任务质量。

## 3. 与上界的关系

上界按原报告的校准值计算（DRAM 102.7 GB/s，算力 3.68 TFLOP/s，均为 burst 实测）。

**Decode**

| 上下文 | 上界 (ms) | mlx-lm | v2 |
|---|---|---|---|
| 16–256 | 11.7–12.0 | 92.4–92.7% | 98.4–98.7% |
| 1024 | 12.8 | 92.8% | 99.3% |
| 4096 | 16.3 | 92.3% | 98.0% |
| 16384 | 30.0 | 93.3% | 97.4% |
| 32768 | 48.3 | 92.1% | 95.7% |

*上下文取 prompt 加平均已生成 token 数。*

原报告认为 mlx-lm 的 decode 离上界还差 5–13%，v2 收回了大部分。"上界基本是紧的"这一结论仅相对于校准带宽成立：如果硬件真实可达带宽更高，上界会更低，v2 的比例也会相应下降。

**Prefill**

| Prompt | 上界（约） | mlx-lm | v2 |
|---|---|---|---|
| 1024 | 0.28 s | 71% | 80% |
| 4096 | 1.50 s | 77% | 83% |
| 16384 | 12.3 s | 77% | 80% |
| 32768 | 41.3 s | 64% | 69% |

*上界 = 投影 FLOPs 与 causal attention FLOPs 之和 ÷ 3.68 TFLOP/s，未计 lm_head 与逐元素操作，属粗略估计。*

- **短 prompt：原报告的诊断得到支持。** 原报告把损失归因于"小矩阵乘效率低 + 每块固定开销"。针对这两点修改后（整段前向、skinny GEMM），TTFT 快了 1.56–1.72 倍。
- **长 prompt：差距仍有约 20–30%，本轮只缩小了几个百分点。** 实测纯寄存器 simdgroup MMA 峰值约 3.6 TFLOP/s；MLX GEMM 在 M ≥ 128 时达到其 87–88%，MLX attention 约 77%，v2 的 attention kernel 约 82%。
- **"softmax 运算未计入上界"的假设被否定**（[softmax_cost.py](softmax_cost.py)，[softmax_cost.json](softmax_cost.json)）。在同一个 attention kernel 上做三个变体，只改 softmax，加载、MMA、控制流和启动几何不变；2048 个 query、2k/8k/32k 个 key，5 轮交错：

  | 变体 | 2k | 8k | 32k (TFLOP/s) |
  |---|---|---|---|
  | 完整 | 2.63 | 2.82 | 2.71 |
  | 去掉 exp | 2.70 | 2.82 | 2.71 |
  | 去掉整个 softmax | 2.38 | 2.58 | 2.47 |
  | MLX SDPA | 2.44 | 2.55 | 2.52 |

  去掉 softmax 反而更慢（编译器对寄存器与指令调度的安排变了），去掉 exp 也没有变化。所以 attention 离 MMA 峰值（同轮实测 3.44–3.53）的约 20% 差距，在 MMA 与数据搬运路径本身（threadgroup 加载、V 的类型转换、barrier），而不在 softmax。上界的计算资源项不需要因此修改；这部分差距原则上可以由更好的 kernel 收回，本轮没有做到。
  *注：该实验第一版的"去掉 softmax"变体因脚本切割位置错误、删掉了写输出的代码，测得 581 TFLOP/s，已作废；修正后先校验该变体确实计算 S·V（相对误差 0.23%）再计时。*
- **热降频是长 prompt 的第二个来源。** 持续 60 s：纯 MMA kernel 全程 3.53–3.56 TFLOP/s，不降速；attention kernel 从 2.65 降到 2.33（−12%，热状态升到 1）。attention 比纯 MMA 多了大量片上数据搬运，功耗更高。32k 请求正好持续约 60 s。
- 粗略拆解 32k（上界 41.3 s）：attention 按 burst 效率约 45 s，计入 60 s 内平均约 7% 的降频约 48 s，投影 GEMM 约 9 s，合计约 57 s，与实测 59.8 s 接近。这只是量级核对，不是严格分解。

因此，"prefill 还有 20–70% 空间"：短 prompt 被证实并已大部分收回；长 prompt 的空间确实存在，位置在 attention kernel 的数据路径与持续负载下的降频。

## 4. 改动内容

| 定位到的瓶颈 | 改动 | 证据类型 |
|---|---|---|
| TTFT 需要两次前向、两次同步，并在 `clear_cache` 后重新分配缓冲区 | 整段 prompt 一次前向；只对最后一行算词表投影；最后一层只对最后一行算 attention/MLP | 微基准 |
| M ≤ 56 时 MLX GEMM 效率低 | 自写 skinny MMA GEMM，按形状调优后查表启用（[tune_skinny.py](tune_skinny.py)） | 微基准：L=16 的 prefill 从 22 ms 降到 17 ms |
| decode GEMV 只有 74–90 GB/s | 自写 BF16 GEMV，并融合 RMSNorm、残差、SwiGLU | 微基准：gate/up 96 GB/s |
| decode 每层约 17 个 kernel | 每层 7 个；K/V 合并成一块缓存，每步只写一次；KV 按请求长度预分配 | 仅整体结果 |
| prefill attention 效率 | GQA 配对 flash attention；行最大值不变时跳过重缩放（乘 1.0，结果精确） | 微基准：2.81 → 2.94 TFLOP/s |
| prefill 的逐元素操作与转置 | qk-norm+RoPE 融合并直接写出 attention 需要的布局；残差 + RMSNorm 融合；SwiGLU 编译 | 阶段剖析：L=2048 时约 50 ms |

**正式基准只比较了完整引擎，没有做逐项消融**，表中各项收益来自单独的微基准，不能相加，也不等于它们在整机中的贡献。

内存方面：引擎另外物化了合并后的 QKV 与 gate/up 权重，约 587 MB。实验进程中它们与原分项权重同时驻留；若加载后释放原分项权重，总字节数与原模型相同，但本轮没有实现这一步。

代码：[engine.py](engine.py)、[kernels.py](kernels.py)、[attention.py](attention.py)。

## 5. 试过但未采用的路径

以下结论只针对本轮的实现方式与这台机器，不是对这些方向的一般性否定。

1. **CPU 矩阵单元与 GPU 协同做 prefill。** CPU 经 Accelerate 跑 FP32 SGEMM 可达 1.9 TFLOP/s，数值与 GPU 相同（BF16→FP32 是精确转换）。两边无依赖并发的 burst 测试中，GEMM 吞吐合计 4.39 TFLOP/s（GPU 单独 3.1）。但是：
   - 按 GEMM 粒度切分时，MLX 跨流依赖每次交接多付约 1–2 ms；改用 numpy 线程直接写统一内存后，每个 GEMM 仍有约 1 ms 的同步与回传开销，L=2048/4096 的净收益约为 0（快照见 [experimental_cpu_split_engine.py](experimental_cpu_split_engine.py)，不能在当前 kernels.py 上运行）。
   - 一次 24 s 的持续负载测试中，GPU attention 吞吐降低约 5%，CPU SGEMM 降到 0.67 TFLOP/s，热状态升到 1。只测了一次，没有重复。
   - 没有尝试的：按层或按 token 区间的粗粒度划分、自写的低开销同步机制。
2. **attention kernel 的其他变体**（K 转置存入 threadgroup memory、V 预转 FP32、更大的 BK/NSG）：单次扫描，差异在噪声范围内或更慢。
3. **Split-K skinny GEMM**（针对 o/down）：规约开销超过收益。

## 6. 本轮证明不了什么

- **上界作为导航的价值。** 改进方向是结合上界和逐段剖析找到的，但没有对照"只用剖析、不看上界"的做法，所以不能说上界是必要的，也不能量化它的帮助。现有证据只到这一步：它正确给出了 decode 的天花板，也正确指出长 prompt 的损失在 attention。对照实验的设计见 [../../docs/navigation-experiment.md](../../docs/navigation-experiment.md)。
- **各项优化的独立贡献**：见第 4 节，正式基准没有消融。
- **跨设备、跨模型、真实文本负载、持续负载**下是否成立。
- **长 prompt 的统计显著性**：16k/32k 只有 2 对。

## 7. 测量方法上的观察

- **热状态影响很大。** 连续负载后，同一个 4096 prompt 的 TTFT 从 1.9 s 变成 3.1 s。`pmset -g therm` 在这台机器上什么也不记录，所以第一轮报告中"没有热警告"不能说明没有降频。v2 用 `NSProcessInfo.thermalState`（[therm.swift](therm.swift)、[thermal.py](thermal.py)）做门控，并记录每次请求前后的状态。不过 `thermalState` 是粗粒度指标，状态为 0 不等于频率完全一致。
- **GPU 空闲降频会给 TTFT 加约 30–40 ms**，两个引擎都受影响（L=16，空闲 0 s 与 2 s 对比，每档 5 次）。所以热态和空闲态分开报告。

## 8. 下一步

1. **长 prompt：** softmax 假设已否定（第 3 节）。下一步是改 attention 的数据路径（例如 Q/K/V 片段直接从 device 读入寄存器、双缓冲 K/V），并把持续负载下的算力作为上界的第二个校准点。
2. **在正式基准中做逐项消融**（整段 prefill、skinny GEMM、融合 GEMV、attention kernel、预分配 KV），并把 mlx-lm 的 `clear_cache` 和分块大小作为对照变量。
3. **检验导航价值：** 按 [对照实验设计](../../docs/navigation-experiment.md) 执行。
4. **放宽到 B/C 级**（量化、投机解码）：decode 在 A 级已接近上界，要再快成倍，只能改变计算本身。需要先定质量预算，基线也要换成同配置的量化版本。

## 复现

```sh
cd agent-test/v2
swiftc -O therm.swift -o thermstate
../../.venv/bin/python -m pytest -q test_engine.py
../../.venv/bin/python check.py --gen 96 --out check.json
../../.venv/bin/python benchmark.py --regime warm --lengths 16 64 256 1024 4096 --out results-warm.json
../../.venv/bin/python benchmark.py --regime idle --lengths 16 64 256 1024 4096 --out results-idle.json
../../.venv/bin/python benchmark.py --regime warm --lengths 16384 32768 --repeats 2 --gen 65 --out results-long.json
../../.venv/bin/python div32k.py
../../.venv/bin/python softmax_cost.py        # 第 3 节的 softmax 与持续负载实验
```

`skinny_table.json` 是这台 M4 上的调优结果，换设备需重新运行 `tune_skinny.py`。
