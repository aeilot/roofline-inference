# 项目思路与设计笔记

本文记录项目的目标、范围、建模方法和待办。
2026-10-05 从头重新确定了范围(第 1、2 节),同日把 prefill 加入范围,其余各节随之调整。最后与代码和实测同步:2026-10-05。

## 1. 项目目标

**给定 model spec 与 device spec,算出 LLM inference 的性能上界,并作为 agent 定制 inference engine 时的导航。**

- 输入:**model spec**(模型结构)+ **device spec**(设备规格及系统底层信息,含实测校准值)。
- 输出:
  - 与实现无关的**上界**(decode 每 token 时间、prefill 的 TTFT、整次请求时间);
  - 给定实现方案时的**方案代价**,以及现有引擎与上界之间的 **gap 分解**:差距落在哪些因素上、各占多少;
  - 每个约束的 slack,以及"放开约束 X 之后上界或代价变为多少"。
- **用途是导航,不是打分。** 上界和代价模型告诉 agent 瓶颈在哪、改哪里收益最大。
  因此可分解、可按约束参数化比把上界压紧更重要。`achieved / bound` 可以作为参考指标,但不作为设计目标。
- 现有引擎(mlx-lm、llama.cpp)代表"现在能做到的",是 gap 的另一端。

> 这不是"表征现有 kernel 落在 roofline 哪里"的分析工具。实测用于校准 device spec 和拆分现有引擎的 gap。

### 1.1 放宽程度:分阶段

上界的含义取决于允许实现方改动什么。分三级,**当前只做 A**:

| 级别 | 允许的改动 | 上界的定义 | 状态 |
|---|---|---|---|
| **A** | 只改实现:融合、调度、引擎分配;与原模型数学等价 | 唯一确定,不需要质量模型 | **当前** |
| B | A + 改精度:量化权重与 KV | 位宽成为参数,需要质量约束 | 以后 |
| C | B + 改算法:投机解码、KV 压缩、剪枝等 | 每个被接受 token 的时间 T(knobs),需要质量预算 | 以后 |

B、C 下不加质量约束时上界没有意义(权重压到 1 bit 就可以任意快)。届时的设想是:T(knobs) 只算物理时间,
质量损失另用实测表("knob 取值 → 质量损失",如 Δppl、投机解码接受率 α)表示,scenario 给质量预算。
现在不做,但接口要能扩展到 knobs。

## 2. 当前范围(2026-10-05 确定)

| 维度 | 决定 |
|---|---|
| 放宽程度 | A(只改实现) |
| 场景 | 单流(batch 1)decode 与 prefill,两者并行推进;batching 以后再做 |
| decode | 上下文长度 512 ~ 32k 扫描 |
| prefill | prompt 长度 16 ~ 32k 扫描 |
| 输出指标 | decode 每 token 时间;TTFT(L);整次请求 T_request(L, N)(5.3) |
| 设备 | MacBook Air M4 24GB,GPU 单引擎为主 |
| 模型 | Qwen3-0.6B,BF16 权重与 KV |
| A 级上界 | = 物理上界(第 5.1 节) |
| 基线引擎 | mlx-lm、llama.cpp(Metal) |

## 3. 当前目标硬件

**MacBook Air M4(Mac16,13)**

| 项目 | 值 |
|---|---|
| CPU | 4P + 6E |
| GPU | 10 核 |
| 内存 | 24GB 统一内存 |
| CPU L2 | P 簇 16MB,E 簇 4MB |
| L1D | 64KB,cache line 128B |
| 环境 | Python 3.14,MLX 0.32.3(`.venv`) |
| GPU 推荐工作集 | 约 17.8 GiB(MLX `max_recommended_working_set_size`) |
| GPU 可见片上容量 | 实测约 12 MiB(GPU L2 + SLC,未公开;见第 8 节) |

## 4. Apple Silicon 的建模要点

1. **统一内存,带宽由 CPU/GPU/ANE 共享。** 理论总带宽约 120 GB/s(LPDDR5X-7500,128-bit)。每个引擎单独可达的带宽低于总量,同时使用多个引擎时总量封顶。spec 里需同时有"总带宽"和"各引擎可达带宽上限"。
   实测:GPU 单独读 102.7 GB/s;CPU 单独 46~54 GB/s;CPU + GPU 同时读 89.6~92.0 GB/s,不超过 GPU 单独,所以 GPU 一个引擎就能吃满 DRAM。
2. **offload 成本近似为 0。** 没有 HBM/PCIe 之分,算子可按引擎自由分配,但引擎之间抢同一份带宽。在 A 级下,引擎分配是实现方可改的一项;按上面的实测,decode 把一部分读交给 CPU 不会更快。
3. **decode 基本被带宽锁死,prefill 基本被算力锁死。** decode 上界约为 `带宽 / (权重字节 + KV 字节)`;prefill 超过约 48 个 token 后受算力限制。在 A 级下字节数和 FLOPs 都是定死的,实现方只能逼近上界;量化、KV 压缩、投机解码属于 B/C 级。
4. **容量约束。** 24GB 中 GPU 可用 wired 内存约 16~18GB,模型加 KV cache 必须装得下。Qwen3-0.6B 在 32k 上下文下约 5 GB,不构成约束。
5. **无风扇,存在热降频。** spec 需区分 `burst` 与 `sustained` 两档。
6. **三套计算引擎。** GPU(实测 BF16 3.68、FP32 3.65 TFLOP/s)、CPU(NEON + SME)、ANE(宣称约 38 TOPS,INT8 为主)。按引擎和精度分表。

> 上述数值部分来自公开资料,部分为估算,**不能直接当 ground truth**。以 device spec 的 `measured` 字段和第 8 节为准。

## 5. 上界与方案代价模型

### 5.1 A 级上界 = 物理上界

只算任何实现都绕不开的量(第 6.1 节):

```
T >= max( FLOPs / P_peak, Bytes_dram / B_dram, ... )
```

其中 `Bytes_dram` 取**最优融合下的最小数据移动**:权重必读一次、KV 必读一次、新 K/V 必写一次、中间激活留在片上。

理由:理论上整个模型可以做成一个常驻 GPU 的 kernel,采样也在 GPU 上完成,token 异步交给 CPU。
这样 dispatch、commit、同步都可以被摊薄到接近 0,所以它们不进上界,而是作为方案代价(5.3)。

Qwen3-0.6B BF16 单流 decode(带宽 102.7 GB/s):

```
T_phys(ctx) = 11.6 ms + ctx × 1.12 µs
```

| context | T_phys | 约 tok/s | KV 占读流量 | compute slack |
|---|---|---|---|---|
| 512 | 12.2 ms | 82 | 5% | 34 |
| 4096 | 16.2 ms | 62 | 28% | 28 |
| 32768 | 48.2 ms | 21 | 76% | 20 |

权重每 token 读 1.19 GB(含 tied 的 lm_head);KV 每个上下文 token 114,688 B(28 层 × K/V × 8 头 × 128 × 2 B)。
上下文越长,瓶颈越从读权重转向读 KV。

### 5.2 prefill 上界

Qwen3-0.6B BF16 单流 prefill(算力 3.68 TFLOP/s,带宽 102.7 GB/s),attention FLOPs 已按 causal 减半:

| prompt | T_phys | 瓶颈 | 另一项 slack | attention 占 FLOPs |
|---|---|---|---|---|
| 16 | 11.6 ms | DRAM | compute 2.96 | 0% |
| 32 | 11.6 ms | DRAM | compute 1.50 | 0% |
| 64 | 15.5 ms | compute | DRAM 1.33 | 1% |
| 512 | 131 ms | compute | DRAM 11 | 6% |
| 1024 | 278 ms | compute | DRAM 22 | 12% |
| 4096 | 1.50 s | compute | DRAM 92 | 35% |
| 32768 | 41.3 s | compute | DRAM 846 | 81% |

- **分界约 48 个 token。** BF16 下每个 token 对每字节权重做 1 FLOP,机器平衡点约 36 FLOP/B;
  短 prompt 和一次 decode 一样,主要时间是读一遍权重,之后受算力限制。
- **长 prompt 由 attention 主导。** 32k 时 attention 占 81% FLOPs,上界主要取决于 attention。
- **softmax 的 exp。** 每个 attention score 一次 exp,对应 QK 与 PV 共约 4 × head_dim = 512 FLOP。
  按 exp 吞吐为 FMA 的几分之一估算,约占 1~2%,作为第 1 层的一项资源列出(6.1),预计不会成为瓶颈。

### 5.3 方案代价 T_A(plan)

方案(plan)指一种具体实现:kernel 怎么切、怎么提交、在哪里同步。代价模型:

```
T_A(plan) = Σ_k (t₀ + bytes_k / B_eff)       每个依赖 kernel:固定成本 + 读流量
          + n_cb × c_commit                  每个 command buffer 的提交
          + n_sync × c_sync                  GPU→CPU 同步
          + extra_bytes / B_dram             未融合中间结果的额外流量
```

- **t₀**:每个**依赖** kernel 的全部固定成本,包括 dispatch、kernel 之间的 barrier,以及启动、爬坡和收尾。
  decode 的 kernel 前后依赖,中间有 barrier,所以按这种方式计。
  不再单列 c_dispatch:如果 BW(bytes) 是单独跑一个 kernel 测的,其中已经含有启动开销,再加 c_dispatch 会重复计算。
- **B_eff**:kernel 在稳态时的读带宽,预计接近 B_dram。decode 的 GEMV 很小(如 k_proj 2 MB,约 20 µs),
  t₀ 相对于 b / B_eff 不可忽略,这就是小 kernel 的带宽损失。整个模型一个 kernel 时 T_A 退化为 T_phys。
- **按实际使用方式拟合,不单独测各项。** 在一个 command buffer 里串起 N 个有依赖的 kernel,每个读 b 字节;
  每次同步前提交 K 个这样的 command buffer;测总时间并拟合

  ```
  T(N, b, K) = K · N · (t₀ + b / B_eff) + K · c_commit + c_sync
  ```

  变 N 和 b 分出 t₀ 与 B_eff,变 K 分出 c_commit 与 c_sync(只变 N、b 时两者总是一起出现)。
  目前只有合测的往返 106 µs(第 8 节)。若拟合残差随 b 呈非线性(如几百 KB 以下并行度不足),
  再把 B_eff 换成 B_eff(b)。
- **独立 kernel 可能重叠。** 同一层的 q/k/v 投影互不依赖,没有 barrier 时可以并发,t₀ 可能小于依赖情形;需要时另测。
- **extra_bytes**:decode 下中间激活只有 KB 级,预计可忽略;operator 粒度的流量见 6.5。
- 各项是相加还是可重叠,需逐项确定(第 10 节)。上式先按相加写,是保守估计。
- 引擎分配:CPU + GPU 同时读不比 GPU 单独快,第一版只考虑 GPU。

上式是 decode 的形状。prefill 的 kernel 都很大,固定开销可以忽略,主项换成算力效率:

```
T_A(plan) = Σ_k FLOPs_k / P(shape_k)         GEMM 与 attention kernel 的算力效率
          + 读写流量 / B_dram(仅在短 prompt 或不融合时可见)
          + 固定开销(同 decode,通常可忽略)
```

- **P(shape)**:给定 GEMM 形状 (m, n, k) 或 attention 形状 (L, head_dim, 头数) 时实际能达到的算力。
  `mx.matmul` 方阵约为峰值的 85%;0.6B 的 GEMM 又窄又小(如 L × 1024 × 1024),tile 和波次量化可能让效率更低。
  P(shape) 在 prefill 侧的角色对应 decode 侧的 t₀ 与 B_eff。
- **分块 prefill(chunked prefill)** 与原计算等价,属 A 级可选方案:块越小,每块重复读权重的流量越多,块越大占用越多内存。

### 5.4 导航仪的输出

1. 上界 T_phys 及资源 slack(已有)。指标:
   - decode:每 token 时间,随上下文长度变化;
   - **TTFT(L)**:prefill L 个 token,含最后一个 token 的 lm_head 与采样;
   - **T_request(L, N) = TTFT(L) + Σ_{i=1}^{N-1} T_decode(L + i)**:整次请求,显示 prefill 与 decode 各占多少。
2. 现有引擎的 **gap 分解**:

   ```
   T_baseline − T_phys = 每 kernel 固定成本(n_kernel × t₀) + 稳态带宽损失 + commit / sync + 额外流量 + 残差
   ```

   前几项用 5.3 的模型和基线的 kernel trace 计算;prefill 中前两项换成算力效率损失(按 P(shape) 计算),残差是模型解释不了的部分;残差大说明模型漏了因素。
3. 对给定方案的 T_A(plan) 及同样的分解,回答"把 kernel 数减半 / 去掉每层同步 / 换一个 attention kernel 之后能快多少"。
4. 结构化输出(JSON),每个数值带来源(datasheet / measured)和置信度。

## 6. 底层系统因素

指模型结构和峰值算力之外,所有影响性能的硬件与系统细节。按"是否物理上不可绕开"分三层。
第 1 层决定上界;第 2 层是方案代价,决定实现离上界多远。

### 6.1 第 1 层:物理上界(进入上界)

| 因素 | 说明 | 获取方式 | 现状 |
|---|---|---|---|
| 可达 DRAM 带宽 | 标称约 120 GB/s,实际只能达到一部分;decode 上界与之成正比 | 实测(STREAM 风格 copy / read) | 已测,GPU 读 102.7 GB/s,已用于上界 |
| 读写带宽差异 | 只读、只写、读写混合速度不同;decode 以读为主 | 实测 | 已测(写 94.9、copy 95.0),上界目前读写共用一个带宽 |
| 各引擎单独可达带宽 | GPU / CPU / ANE 各自能用到的带宽 | 分引擎实测 | GPU 已测;CPU 46~54 GB/s 只记在 note,未成字段;ANE 未测 |
| 带宽争抢 | 多引擎同时访存时总量封顶,不叠加 | CPU + GPU 并发实测 | 已测,总量取 GPU 单独值 |
| GPU 可用内存上限 | 24GB 中 GPU 默认约 16~18GB(`iogpu.wired_limit_mb`,0 表示系统默认);**硬约束** | 读系统参数 | 已读,17.76 GiB,已用于容量检查 |
| 各精度峰值算力 | FP32 / FP16 / BF16 / INT8 分引擎分精度 | 估算,再用 GEMM 实测 | GPU FP32/FP16/BF16 已测(FMA 峰值);INT8 与其他引擎未测 |
| 特殊函数吞吐 | softmax 的 exp 等;prefill attention 中每个 score 一次,是独立于 FMA 的资源 | 实测 exp 峰值吞吐 | 未测;按估算约占长 prompt attention 时间的 1~2%(5.2) |

### 6.2 第 2 层:方案代价(进入 T_A(plan),agent 可调)

| 因素 | 说明 | 获取方式 | 现状 |
|---|---|---|---|
| 依赖 kernel 固定成本 t₀ 与稳态带宽 B_eff | 每个 kernel 的 dispatch + barrier + 爬坡;决定 kernel 切分的代价 | 拟合 T(N, b, K)(5.3),b 取 64 KB ~ 64 MB | **未测**,A 级 decode 预计最大的一项 |
| GEMM 算力效率 P(shape) | 给定 (m, n, k) 时 GEMM 能跑到峰值的多少;0.6B 的形状窄小,可能远低于方阵 | 按模型实际形状与 L 扫描,mx.matmul 与自写 kernel | 只有方阵数据(约 85%);**未测**,A 级 prefill 预计最大的一项 |
| attention kernel 效率 | 长 prompt 下的主项;取决于分块、causal 跳块、softmax 实现 | 按 L 扫描 MLX 的 attention kernel | 未测 |
| commit 开销 | 每个 command buffer 的提交成本 | 同上拟合,变 K | 未拆分 |
| CPU-GPU 同步 | 采样通常需每 token 一次 GPU→CPU 往返;GPU 上采样可绕开 | 同上拟合 | 与上一项及 t₀ 合测为 106 µs |
| 片上存储容量 | GPU L2 + SLC、threadgroup memory(32KB);决定中间激活能否不回写 DRAM | 部分公开,SLC 需实测 | 已测约 12 MiB,已用于 `operator` 上界(6.5) |
| 访问粒度 | cache line 128B、页 16KB;小块零散访问浪费带宽 | 已知 | 未建模 |
| 热降频 | 无风扇,长时间负载降频;spec 分 `burst` / `sustained` | 连续跑数分钟记录性能曲线 | 未测;现有实测值都是 burst |

B 级时再加入:量化元数据开销(如 group size 64 + FP16 scale/bias,4-bit 实际约 4.5 bit/权重)、反量化 ALU 开销。

### 6.3 第 3 层:测量噪声(不进入模型,仅在实测时控制)

| 因素 | 控制方式 |
|---|---|
| 后台程序占用 CPU / GPU / 带宽 | 关闭无关程序,测前检查负载 |
| 首次运行编译(shader、pipeline) | 预热后计时 |
| Python / 框架开销 | 用 GPU 时间戳计时,单独列出。目前 benchmarks 用墙钟计时,靠大样本(GB 级或 >50 ms)把启动开销压到约 1%;几十 µs 的小 kernel 必须改用 GPU 时间戳 |
| P/E 核调度 | 记录 QoS,CPU 测试固定线程数 |
| 动态调频、电源状态 | 插电、关闭低电量模式、预热后再测 |
| 内存压缩与换页 | 监控 pageout;建模中以容量约束保证不发生 |
| 屏幕显示、模型从 SSD 加载 | 影响小或只影响启动,不计入稳态 |

### 6.4 代码现状

`--bound` 有两个取值:

- `physical`:第 1 层,片上存储无限、理想融合。即 A 级上界,decode 与 prefill 都已支持。
- `operator`:第 1 层 + "有限片上容量 + 每个 GEMM 为独立算子"(6.5),`--on-chip-mib` 可放松 M。
  它给出的是某一类方案(不跨 GEMM 融合)的流量下界,属于方案代价的一个分量。

T_A(plan)、TTFT / T_request 与 gap 分解尚未实现;exp 吞吐尚未作为资源接入。

### 6.5 有限片上存储:red-blue pebble game

物理上界假设片上存储无限(中间激活全部留在片上)。`--bound operator` 把它换成有限容量 M,
实现见 `src/roofline/pebble.py`。

- **假设:每个 GEMM 是一个独立算子。** 激活输入 A 从 DRAM 读(最多 M 字节可以还在片上),
  输出 C 写回 DRAM(同样最多 M 字节可以留在片上);逐元素运算免费融合进 GEMM。
  这取决于融合方案,属第 2 层。A 由上游算子在片上直接产生(跨 GEMM 融合)时下面的定理不适用,
  整图融合下的界是开放问题(SOAP, SPAA'21)。
- **每个 GEMM 的读下界**(元素数):`max(W, W + A - M, 2mnk/√M - mn - 2M)`。
  第三项来自 Smith et al. 2019(TOMS)推论 3.8,首项系数是紧的。写下界:`max(新 K/V, C - M)`。
- **M 取上沿估计。** M 越大界越弱,所以取片上容量的上沿仍然是合法下界。
  2026-10-04 实测:读带宽 2.4 MiB 以内约 230 GB/s,11.3 MiB 起降到 DRAM 速度,取 12 MiB。
- **M4 上的结论:这一项不会成为瓶颈。** pebble 项把 GEMM 的算术强度上限压到约 √M/2 FLOP/byte,
  M = 12 MiB 时约 1250,远高于机器平衡点 3.68e12 / 102.7e9 ≈ 36。
  所有 Qwen3-0.6B 场景下 operator 上界与物理上界的时间相同;
  只有 M 缩小到约 32 KiB 时,DRAM slack 才降到 2 左右(prefill 1024 为 1.98)。
  它的作用是量化"不做跨 GEMM 融合"的流量代价(如 prefill b8×4096:物理 5.0 GB,operator 34 GB),
  并在机器平衡点更高的设备上给出更紧的界。decode 下 GEMM 的 m = 1,这一项可以忽略。
- **未覆盖:** attention 的 pebble 下界(Saha & Ye 2024,Θ(N²d²/M),M ≥ d² 时由 FlashAttention 达到)
  只给出了阶,没有显式常数,暂不计入。

## 7. 计划

已完成:

1. [x] `configs/hardware/m4_mba_24g.yaml`:device spec,实测字段已填(2026-10-04)。
2. [x] `configs/models/qwen3_0.6b.yaml`:model spec schema。
3. [x] `src/roofline/`:decode 与 prefill 的 `physical` / `operator` 上界,输出瓶颈和 slack。
4. [x] `benchmarks/`:`bandwidth.py`、`compute.py`、`contention.py`,用 MLX 自定义 Metal kernel 校准 spec。

下一步(A 级、单流,decode 与 prefill 并行推进):

5. [ ] **依赖 kernel 链拟合**(decode):一个 command buffer 里串 N 个有依赖的读 kernel,每个读 b 字节(64 KB ~ 64 MB),
   每次同步前提交 K 个 buffer;拟合 `T(N, b, K)` 得到 t₀、B_eff、c_commit、c_sync(5.3)。
6. [ ] **拟合检验**:看残差是否随 b 非线性(需要 B_eff(b));必要时另测无依赖 kernel 并发时的 t₀。
7. [ ] **算力效率曲线 P(shape)**(prefill):按 Qwen3-0.6B 的实际 GEMM 形状与 L 扫描;attention kernel 按 L 扫描;exp 峰值吞吐。
8. [ ] **上界补全**:exp 作为资源接入;输出 TTFT(L) 与 T_request(L, N)。
9. [ ] **基线**:mlx-lm 与 llama.cpp(Metal),decode 在 context 512 ~ 32k、prefill 在 prompt 16 ~ 32k 上跑;
   一次 trace 同时抓两段,取 kernel 数、每个 kernel 的耗时、字节数与 FLOPs。llama.cpp 需另配环境。
10. [ ] **gap 分解**:实现 T_A(plan),按 5.4 把 `T_baseline − T_phys` 拆成各项与残差,decode 与 prefill 分别给出。

以后:

11. [ ] sustained 档:连续负载数分钟,测热降频后的带宽与算力。长 prompt prefill(32k 约 41 s)本身就会进入降频区间。
12. [ ] B 级(量化)、C 级(投机解码、KV 压缩)、batching、多引擎分配、MoE。

## 8. 实测结论(2026-10-03 ~ 10-04)

脚本在 `benchmarks/`,原始结果在 `results/raw/`(不进 git),写入 device spec 的 `measured` 字段。
共同做法:预热后计时;墙钟包住 `mx.eval`;**取最快一次**(中位数记在 spec 的 note);每次只跑几秒,都是 burst 值。

| 量 | 结果 | 方法 |
|---|---|---|
| GPU 读带宽 | 102.7 GB/s(中位数 102.2) | `bandwidth.py`:自写 Metal kernel,grid-stride 读 1 GiB(uint4),4096 group × 256 线程,best of 20 |
| GPU 写 / copy | 94.9 / 95.0 GB/s | 同上;写需 16384 group 才跑满(1024 group 仅约 36) |
| 总带宽 | 102.7 GB/s(取 GPU 单独值) | `contention.py`:CPU、GPU 各自独立进程,同时开始读 3 s,best of 3 |
| CPU 单独读 | 46~54 GB/s(1~4 进程) | 同上,MLX CPU stream 上 `mx.sum` |
| CPU + GPU 同时读 | 89.6~92.0 GB/s(GPU 部分降到 63~67) | 同上;不超过 GPU 单独的 95.0 |
| 片上容量 M | 12 MiB | `bandwidth.py --sweep-per-octave 4`:2.4 MiB 内约 230 GB/s,11.3 MiB 降到 DRAM 速度;取上沿 |
| GPU 峰值算力 | BF16 3.68 / FP16 3.71 / FP32 3.65 TFLOP/s | `compute.py`:每线程 64 条独立 FMA 链;128 条时 FP32 寄存器溢出 |
| `mx.matmul` | BF16 3.0~3.1 / FP32 2.6~2.8 TFLOP/s(峰值的约 85% / 76%) | 同上,方阵 n = 1024~8192;现有库水平,不进上界 |
| 启动 + 同步往返 | 106 µs(中位数 135) | 1 MiB 读 kernel,best of 200;dispatch / commit / sync 未拆分 |

由此得到的 Qwen3-0.6B BF16 物理上界见 5.1(decode)与 5.2(prefill)。

注意:106 µs 的往返如果每 token 一次,约占 4096 上下文物理上界 16.2 ms 的 0.7%;
若每层一次(28 层)则约 3 ms,接近 20%。但同一 command buffer 内的 dispatch 应远便宜于完整往返,
所以"每层一次"的估计偏悲观,需要计划第 5 项的拟合把它拆开。

## 9. 已确认

- 项目定位:agent 的导航仪,上界用于指方向和回答放松约束的问题,不以 reward 为设计目标(2026-10-05)。
- 放宽程度先做 A;B、C 以后做(2026-10-05)。
- A 级上界 = 物理上界;dispatch / commit / sync / 小 kernel 带宽损失计入方案代价,不进上界(2026-10-05)。
- 场景:单流 decode(context 512 ~ 32k)与 prefill(prompt 16 ~ 32k),并行推进(2026-10-05)。
- 指标:decode 每 token 时间、TTFT(L)、T_request(L, N)(2026-10-05)。
- 设备 M4 MBA 24GB、模型 Qwen3-0.6B BF16,先固定(2026-10-05)。
- 基线引擎:mlx-lm、llama.cpp(Metal)(2026-10-05)。
- 总带宽封顶值取 GPU 单独实测值(第 8 节)。
- 片上容量 M 取实测拐点上沿(6.5)。

## 10. 待确认

- T_A(plan) 各项的组合方式:相加(串行)还是可重叠(取 max),需逐项确定。例如 CPU 端 encode 下一个 buffer 可与 GPU 执行重叠。
- 上界与代价模型使用 burst 还是 sustained 实测值。长 prompt prefill 持续数十秒,burst 值可能偏乐观。
- 基线 kernel trace 的获取方式:Metal System Trace、MLX 的 Metal capture,还是自己插 GPU 时间戳。
- gap 分解的残差多大可以接受,超过时怎么定位漏掉的因素。
