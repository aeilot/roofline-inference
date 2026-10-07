# 项目思路与设计笔记

本文记录项目的目标、范围、建模方法和待办。
2026-10-05 从头重新确定了范围(第 1、2 节),同日把 prefill 加入范围,其余各节随之调整。2026-10-06 加入计算图 + pebble game → roofline 的方法(6.7),并补出 attention I/O 下界的常数(6.6)。最后与代码和实测同步:2026-10-06。

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
  目前只有合测的往返 106 µs(第 8 节)。(2026-10-06:已拟合,并发现 CPU 编码必须单列、相加式不成立,见 6.9。)若拟合残差随 b 呈非线性(如几百 KB 以下并行度不足),
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

`--bound` 有三个取值:

- `physical`:第 1 层,片上存储无限、理想融合。即 A 级上界,decode 与 prefill 都已支持。
- `operator`:第 1 层 + "有限片上容量 + 每个 GEMM 为独立算子"(6.5),`--on-chip-mib` 可放松 M。
  它给出的是某一类方案(不跨 GEMM 融合)的流量下界,属于方案代价的一个分量。
  每层 attention 也作为独立算子计入(6.6)。

- `hierarchy`:多层多处理器下界(6.8),在计算图上同时给出整图界与算子边界界,输出每条存储边界的流量、AI 上限与时间。

`graph.py`:decode 与 prefill 的算子级计算图(6.7),每个收缩记乘积形状 (m, n, k),attention 记 score 数。
FLOPs 与必读必写流量都与 `workload.py` 完全一致(2026-10-06 修正:tied embedding 下 `workload.py` 原先另计了
embedding gather,每 token 2 KB,而被 gather 的行本来就在 LM head 的整表读取之内,属于多算,已去掉;
另加 token id 的读写,每个 4 B)。

`iobound.py`:存储层级上的 I/O 下界(6.8)。`toy.py`:元素级小图、精确红蓝 pebbling 求解器与显式方案的 I/O,用于验证。
`scripts/plot_roofline.py`:roofline 图与 JSON。

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
- **attention 的界单独讨论,见 6.6。**

### 6.6 attention 的 I/O 界

Saha & Ye 2024(arXiv 2402.07443,目前只有 v1)给出 attention 的 I/O 复杂度:标准矩阵乘法下为 Θ(min(N²d/√M, N²d²/M))(定理 3.1);
允许任意矩阵乘法算法时为 Ω(min(N²d²/(M log²N), N²))(定理 4.14,限定 Q, K ∈ {0,1})。
定理 4.14 带 log 因子且要求输入为 0/1,只说明"换快速矩阵乘法也绕不开",不用于估算。
下面只用定理 3.1 的 large cache 一支(引理 3.2),并从它的证明里补出常数(2026-10-06)。

**显式常数。** 论文的证明只给出 O(·),逐步写出常数如下。记 S = 2M。由 Hong–Kung,Q ≥ M · (P(2M) − 1),
P(2M) 是 2M-partition 的最少份数。在任一份 V' 中:

1. 含 dominator 顶点的求和树至多 S 棵,含 minimum 顶点的至多 S 棵;
2. 其余的树(论文称 extra 树)的根在 V' 中,它的 2d 个输入必须全部在 dominator 集里。
   设 C 棵 extra 树涉及 I 个不同的行、J 个不同的列,则 (I + J)·d ≤ S 且 IJ ≥ C。
   由 I + J ≥ 2√C 得 C ≤ S²/(4d²)(论文用 max(I, J) ≥ √C,松 4 倍);
3. 每棵求和树有 2d − 1 个 level-1 顶点(d 个乘积、d − 1 个加法)。

全图共 E·(2d − 1) 个 level-1 顶点,E 为实际算出的 score 个数,所以 P(2M) ≥ E / (M²/d² + 4M),

```
Q ≥ E·d² / (M + 4d²) − M        (元素,读 + 写)
```

即 `N²d²/(cM)` 中 **c = 1 + 4d²/M**,M ≫ d² 时 c ≈ 1。式子对任意 M 都成立,M < d² 时偏松(那里真实的阶是 E·d/√M)。

**推广:** 计数只用到"E 个不同的 (行, 列) 对",没有用到 N × N,所以下面几种情形直接成立:

- 矩形:n 行 query、L 个 key,E = nL;
- causal:E 取实际算出的 score 个数;
- GQA:共享同一 KV 头的 g 个 Q 头的行拼在一起,n = gN;
- 多头、多层、batch 内多条序列一起算:C ≤ Σ I_h·J_h ≤ (Σ I_h)(Σ J_h),界仍成立;
- 多核:总有一个核至少算 E/P 个 score,对它用该核的私有容量求界;
- 求和顺序:论文的求和树结构任意,结论不依赖求和顺序。

**限制:**

- 只对标准矩阵乘法(经典算法,求和顺序任意)成立。
- 证明要求 Q、K 是计算图的输入,即 attention 是独立算子(Q/K/V 来自 DRAM)。整图里 Q、K 由投影、q/k-norm、RoPE 产生,
  dominator 可以截在上游。直觉上截在上游不会更便宜(每行上游宽 1024 > d = 128,权重是共享的输入),
  但需要 Menger 式的不相交路径论证,尚未证明。这是 6.7 整图合成问题的一个具体形式。

**算术强度上限。** FLOPs = 4Ed(QK^T 与 PV),BF16 下 AI ≤ 2(M + 4d²)/d FLOP/byte(E 很大、−M 项可忽略时):

| M | c | AI 上限(FLOP/byte) |
|---|---|---|
| 12 MiB(片上容量上沿,DRAM 层) | 1.01 | 99,328 |
| 32 KiB(只算 threadgroup memory,正好 = d² 个元素,两个区间的分界) | 5 | 1,280 |
| 256 KiB(每核寄存器 + threadgroup + L1 的假设值,未公开) | 1.5 | 3,072 |

机器平衡点约 36。即使取 32 KiB 一行,片上到核心的总带宽也要低于 3.68e12 / 1280 ≈ 2.9 GB/s 才会卡住算力,不可能发生。

- **可达一侧(方案代价,第 2 层):** FlashAttention 外层按 query 分块,每块 B 行,每块把 K、V 完整读一遍,
  I/O ≈ 2nLd/B + 2nd 个元素,强度 ≈ **B** FLOP/byte(B 计入共享同一次 K/V 读取的所有 GQA 头的行)。
  这里决定 B 的是单个 threadgroup 的 tile,不是 SLC 的 M:Q tile 占 B × 256 B,
  B = 36 需 9 KiB,B = 64 需 16 KiB,都放得进 32 KiB threadgroup memory。
  要保持算力瓶颈,需要有效的 B ≳ 36。并发 threadgroup 若同时扫同一段 K/V,
  会经由 SLC 共享读取,有效 B 大于 tile 的 B。32k 时一个 KV 头一层的 K/V 为 16 MiB,超过 SLC,
  这种共享取决于调度,只能实测。
- **decode:** n = g = 2,E 很小,上式为负;强度为 2 FLOP/byte,由"KV 必读一次"决定(5.1)。
- **代码:** `pebble.py` 的 `operator` 界把每层 attention 当作一个独立算子,流量取 max(KV cache 读, 上式),
  E 含所有头和 batch。prefill 32k、M = 12 MiB 时这一项为 0.89 GB(28 层合计),远小于 GEMM 部分。
  attention 自身的 Q 读、O 写和 prefill 中新 K/V 的回读仍未计入(约 344 KB/token,32k 时 11.3 GB,约 0.11 s,
  远小于 41.3 s 的算力时间)。少算只会让下界更弱,不影响正确性。

**结论:** 在 M4 + Qwen3-0.6B 上,attention 的 I/O 下界在所有存储层、decode 和 prefill 下都不会影响上界。
真正需要关注的是可达一侧的 B,它属于 P(shape) 和方案代价(5.3),应通过实测确定。

### 6.7 下一步方法:计算图 + 多层多处理器 pebble game → roofline(2026-10-06)

**思路:**

1. 把一次 decode / prefill 写成计算图;
2. 用多层存储、多处理器的 red-blue pebble game,求每一层存储边界的 I/O 下界 Q_ℓ;
3. 得到算术强度上界 AI_ℓ = FLOPs / Q_ℓ;
4. 乘以该层的峰值带宽 B_ℓ,得到性能上界 min(P_peak, AI_ℓ · B_ℓ);
5. 先画第一张 roofline 图。

**做法:**

- **计算图:** 算子级,不做元素级(decode 约 6 亿个顶点)。节点带迭代空间与形状(矩阵乘、逐元素、行归约、attention),
  边是张量及其大小。另做一个玩具规模的元素级图生成器,只用于验证。
- **每个节点的下界:** 查现成结论(下表);逐元素、norm、RoPE、softmax 只算必读必写。
- **多层:** Savage 的 memory hierarchy game:第 ℓ 层边界的 I/O ≥ 快存储取"第 ℓ 层以上容量之和"时的两层下界。
- **多处理器:** GPU 10 个核心。寄存器、threadgroup memory、L1 每核私有,L2 / SLC / DRAM 共享。
  私有层:总有一个核至少承担 W/P 的计算,对它用私有容量求界;共享层按总流量。
- **整图合成:** 唯一需要自己证明的部分。各子图下界相加并不天然成立:子图的输入可能由上游在片上直接产生;
  允许重算时,dominator 可能截在上游(如 down_proj 的输入可由 X 重算)。
  候选工具:Elango et al. POPL'15 的分解定理、IOLB(PLDI'20)、SOAP(SPAA'21),需逐条核对前提;
  确认不了的部分退回一定成立的 max(必读必写, 各子图的 S-partition 项),并在输出中标注。
- **图:** 横轴 AI、纵轴 FLOP/s,都取对数。屋顶为 BF16 3.68 TFLOP/s 与 DRAM 102.7 GB/s(标称 120 GB/s 画虚线对照)。
  点为 decode(ctx 512 / 4k / 32k)和 prefill(L 16 / 64 / 512 / 4k / 32k)的整图点,外加各算子的点
  (qkv、o_proj、mlp、lm_head、attention)。按定义这些点都在屋顶上,信息在横坐标;以后把基线实测画在点的下方。
- **验证:** 在玩具规模的元素级图上暴力求最优 pebbling,检查下界 ≤ 最优;两层结果须复现 6.5 与 5.1 / 5.2。

**现成结论**(除 Saha & Ye 外,定理原文与常数尚待逐条核对):

| 部分 | 结论 | 用法 |
|---|---|---|
| 矩阵乘,两层 | Smith et al. 2019(TOMS)推论 3.8,紧常数 | `pebble.py` 已用 |
| 矩阵乘,多核 | Irony–Toledo–Tiskin 2004:每个处理器约 ≥ n³/(2√2·P·√M) − M;COSMA(SC'19)给出紧常数 | 每核私有层 |
| 矩阵乘,多层 + 多核 | Ballard–Demmel–Holtz–Schwartz 2011(SIMAX),基于 Loomis–Whitney,只数乘法,不依赖求和顺序 | 多层、多核 |
| 多层归约 | Savage 1995 的 memory hierarchy game | 任意计算图 |
| attention | Saha & Ye 2024,显式常数见 6.6 | 算子边界下严格成立 |
| 逐元素 / norm / RoPE / softmax | 必读必写 | — |

**预期:**

- DRAM 层大概率不改变物理上界。decode 的 AI ≈ 1,在平衡点 36 左侧。prefill 只算必读必写时 AI 约随 L 增长;
  pebble 项把单个 GEMM 的 AI 压到约 √M/2 ≈ 1250(M = 12 MiB),L 超过约 1250 后点的横坐标停住,
  但仍远大于 36,瓶颈不变。
- 新信息来自内层和多核:threadgroup 32 KiB 对应的 GEMM AI 上限约 64 FLOP/byte,是否成为瓶颈取决于尚未实测的片上带宽。
  (实测与结果见 6.8:在 core 边界也不卡。)

**交付顺序:**

1. 计算图 + 两层下界 + 第一张 DRAM 层 roofline(只用已有实测值);
2. 多层 + 多处理器 + 分层 roofline(需先补测 L2/SLC → 核心、threadgroup memory 的带宽,否则内层屋顶只能标为估算)。

两步均已完成(2026-10-06),见 6.8。

### 6.8 多层多处理器 I/O 下界与 roofline(2026-10-06 实现)

实现在 `src/roofline/iobound.py`,图由 `scripts/plot_roofline.py` 生成(`results/figures/`)。

**存储层级**(device spec 的 `hierarchy`,从核心向外):

| 层 | 份数 | 每份容量 | 外侧边界带宽 | 来源 |
|---|---|---|---|---|
| core(寄存器 + threadgroup + cache,M3 起共用一块 SRAM) | 10 | 512 KiB | L2 → 核心 770 GB/s | 容量为估计上沿;带宽实测(第 8 节) |
| chip(GPU L2 + SLC) | 1 | 12 MiB | DRAM 102.7 GB/s | 均为实测 |

按 Savage 的 memory hierarchy game,某层外侧边界的快存储取"该层及以内所有存储之和":

- core 边界:每核 S = 512 KiB,P = 10;
- DRAM 边界:S = 12 MiB + 10 × 512 KiB = 17 MiB,P = 1。

容量取上沿,只会让下界更弱,不影响正确性。每条边界 b 给出 Q_b,于是 AI_b = FLOPs / Q_b,时间下界为 Q_b / B_b,
总时间下界为 max(FLOPs / P_peak, max_b Q_b / B_b)。

**两种界:**

1. **整图界(fused)**:选择 ③ 下任何实现都成立。目前只证明了必读必写:`Q_b ≥ 必读必写 − P·S`
   (每个用到的输入元素至少跨边界一次,每个输出一次,减去开始时可能已在快存储里的 P·S)。
   与物理上界的差别只有这个暖缓存项(DRAM 边界 17 MiB,约 0.17 ms)。
2. **算子边界界(operator)**:每个收缩(共享输入的合并为一个 GEMM:qkv、gate_up)和每层 attention 是独立算子,
   激活输入来自边界外侧,输出写回边界外侧(至多 P·S 可留下);逐元素、行归约免费融合;残差流的流量不计(少算只让下界更弱)。
   每个算子取下面两项中较大的一项,对所有算子求和,最后统一减一次 P·S:
   - 必读必写:`W + (A − P·S)⁺ + max(新 K/V, C − P·S)`;attention 为 `KV cache + (Q + 新 K/V − P·S)⁺ + (O − P·S)⁺`;
   - pebble:GEMM `2mnk/√S − 3P·S`,attention `E·d²/(S + 4d²) − P·S`。
   整图界对这类方案同样成立,所以报告两者中较大的一个。

**求和为什么成立。** 按算子执行时段划分。时段 o 内的 I/O 至少是该算子自己的 pebble 项,
也至少是"它用到的数据减去时段开始时已在快存储里的部分"。开始时已在快存储里的数据,要么在整步开始时就在(≤ P·S),
要么是更早时段多读进来的。后者本身也计入那些时段的 I/O,而 Smith 的分段论证中,不属于本算子的读取只占读预算,
不增加本算子可完成的 FMA 数。所以总量 ≥ Σ_o max(pebble_o, 冷启动必读必写_o) − P·S。
权重的暖缓存只能全局减一次,不能每个算子各减一次:否则 decode 的小 GEMM 各自"假设权重已在片上",总和会低于物理上界。

**多处理器的 GEMM 项。** Smith et al. 引理 3.4(一个读代价为 R 的段内至多 ((M+R)/3)^{3/2} 个不同的 FMA)
对任意一组 FMA 成立,不要求是整个矩阵乘,所以可以逐核应用:

- 核 p 的读 ≥ 2W_p/√S − 2S − T_p,其中 T_p 是该核新建的累加器个数;
- 这些累加器必须离开该核,所以写 ≥ T_p − S;
- 两式相加得 I/O_p ≥ 2W_p/√S − 3S,对核求和得 2mnk/√S − 3P·S。

Smith 的模型不允许重算,论文论证这不失一般性。attention 项对每个核单独做 S-partition 论证再求和(6.6)。

**为什么整图界只有必读必写(2026-10-06 已部分解决,见 6.12):**

- 子图的输入可以在片上产生,而且允许重算,所以 dominator 可以截在上游。
  例:down_proj 的输入 act 可由 h2 的行(每行 1024)和 Wg/Wu 的列(每列 2048)重算。
  行数 |R|、列数 |J| 都大时,2048|J| + 1024|R| < |R||J|,截在上游比截在 act 本身便宜。
  attention 的 Q 同理:h 的一行可以生成全部 16 个头的 Q。
- 行私有的输入只有 token id 和 embedding 行。重复的 token 让"每行一组不相交路径"在输入处合并,
  Menger 式论证不能直接套用。
- 可行方向:分段计数时,把"重算所需的权重"计入每段 ≤ 2S 的 dominator 预算,对每类收缩得到形如
  N_A ≤ |D_A| + (2S)²/(4αβ) 的界(α、β 分别为重算路线每行、每列的权重代价)。核心层 S 小,修正项很小,
  有望得到接近算子边界界的整图界;DRAM 层修正项大。尚未完成。

**验证**(`tests/test_iobound.py`、`src/roofline/toy.py`):

- 精确求解器:在元素级小图上(matmul、attention,十几个顶点)做 0-1 BFS,求允许重算的最优红蓝 pebbling,
  检查必读必写项与各 pebble 项都不超过最优值。规模太小,pebble 项在这里是平凡的,这一项主要检验游戏规则和必读必写项。
- 显式方案(真实规模):对一组形状、S、P,下界都不超过分块 GEMM 和 FlashAttention 式分块的 I/O。
  GEMM 项在 8192³、S = 64K 时与分块方案只差 5% 以内,Smith 的常数是紧的;
  attention 项比 FlashAttention 低约 5 倍,与理论比值 4(1 + 4d²/S) 一致。

**结果**(Qwen3-0.6B BF16,单流;AI 单位 FLOP/byte;机器平衡点:DRAM 边界 36,core 边界 4.8):

| 场景 | 界 | 时间下界 | 瓶颈 | AI(core 边界) | AI(DRAM 边界) |
|---|---|---|---|---|---|
| decode 512 | 两者相同 | 12.01 ms | DRAM | 1.1 | 1.1 |
| decode 4k | 两者相同 | 16.01 ms | DRAM | 1.3 | 1.3 |
| decode 32k | 两者相同 | 48.03 ms | DRAM | 1.8 | 1.8 |
| prefill 16 | 两者相同 | 11.45 ms | DRAM | 12.1 | 12.3 |
| prefill 64 | 两者相同 | 15.53 ms | 算力 | 47.9 | 48.4 |
| prefill 512 | fused / operator | 130.8 ms | 算力 | 387 / 378 | 390 / 390 |
| prefill 4k | fused / operator | 1.50 s | 算力 | 3340 / 402 | 3365 / 1729 |
| prefill 32k | fused / operator | 41.3 s | 算力 | 30740 / 1124 | 30818 / 3522 |

- **上界不变。** 两条边界、两种界下,时间下界都等于物理上界(只差暖缓存的 ≤ 1.5%)。
  decode 被 DRAM 锁住;prefill 从 64 个 token 起被算力锁住。
- **算子边界的代价在长 prompt 下显现。** prefill 32k 的 DRAM 边界 AI 从 30818 降到 3522,
  主要来自激活回写(o_proj 的 A 有 67M 个元素,远超 S),其次是 GEMM pebble 项;仍远大于 36,不卡。
- **core 边界。** L2 → 核心 770 GB/s,机器平衡点只有 4.8。decode 在这里的 AI 为 1.1–1.8,时间约为 DRAM 边界的 1/7.5;
  prefill 的算子边界界 AI 为 378–1124,不卡。
- **对每核容量估计不敏感。** 把每核容量从 512 KiB 降到 32 KiB(16 倍),core 边界时间在所有场景下仍不到瓶颈时间的 1/7。
- **各算子**(图 1 右):decode 下各组 AI ≈ 1(attention 为 2);prefill 4k 下各组在 1000–4500;
  lm_head 只算最后一行,AI ≈ 1,但只占 0.3 GB。

**结论:** 在 M4 + Qwen3-0.6B 上,多层多处理器的 pebble 下界不改变任何场景的上界和瓶颈。
它的用处在导航上:给出不同融合方案(整图 vs 算子边界)在各层边界上的流量,以及哪一层离卡住最近
(decode 的 DRAM;其次是 decode 的 core 边界,差 7.5 倍)。

### 6.9 方案代价的校准、基线与 gap 分解(2026-10-06)

**依赖 kernel 链拟合**(`benchmarks/kernel_chain.py`、`scripts/fit_kernel_chain.py`;计划第 11、12 项)。
一个 command buffer 里串 N 个有依赖的读 kernel(每个读 b 字节、从未读过的数据,保证来自 DRAM),
每次同步前提交 K 个 buffer,N = 1~256、b = 64 KiB~64 MiB、K = 1~16。

- 5.3 原来的相加式 `K·N·(t₀ + b/B_eff) + K·c_commit + c_sync` **不成立**:K = 1 时 GPU 要等 commit 后才开始,
  CPU 编码与 GPU 执行相加;K > 1 时编码下一个 buffer 与执行当前 buffer 重叠。所以 CPU 编码要单列:

  ```
  CPU   每个 kernel 编码 e,每个 buffer 提交 c_commit       (async_eval 返回前的时间)
  GPU   每个依赖 kernel g(b) = t₀ + b / B_eff
  T     buffer k 在 k·(N e + c_commit) 时提交,GPU 按顺序执行,最后加 c_sync
  ```

- 拟合结果:e = 2.46 µs,c_commit = 17.4 µs,t₀ = 2.44 µs,B_eff = 93.9 GB/s,c_sync = 148 µs。
  流水线模型的预测误差中位数:K = 1 为 3.5%,K > 1 为 5.5%;最差的几个都是 N = 1(首个 kernel 的抖动)。
- g(b) 在 64 KiB~64 MiB 内对 b 线性,残差在 ±15% 以内(256 KiB 一点为 +15%,其余 ≤ 7%),不需要 B_eff(b)。
- 无依赖的 kernel 能重叠:64 KiB 时只要依赖链时间的 0.75,1 MiB 起约为 1。
- 原来合测的 106 µs 往返 ≈ e + c_commit + g + c_sync,与拟合一致。

**P(shape)**(`benchmarks/compute_shapes.py`;第 13 项)。用 MLX 自己的 kernel(mlx-lm 实际调用的)在 Qwen3-0.6B 形状上测:

| 项 | 结果 |
|---|---|
| prefill GEMM(q/k/v/o/gate/up/down 分开,mlx-lm 的做法) | M ≥ 512 时 3.15~3.26 TFLOP/s(峰值的 86~89%);M = 256 时 2.4~3.1;M = 64 时 2.2~3.0;M = 16 时 0.6~1.2 |
| decode GEMV(M = 1,40 份同时发出) | 60~94 GB/s;lm_head(311 MB)94 GB/s |
| SDPA prefill(2048 行一块,causal) | 2.8 TFLOP/s(76%);L = 512 时 1.9 |
| SDPA decode(1 个 query) | 49~94 GB/s(KV 越长越高) |
| 小 kernel(RMSNorm、RoPE、add、SwiGLU、cache 写入,M = 1,依赖链) | 每个 4~7 µs(含 CPU 编码约 2.5 µs) |
| exp 峰值(`fast::exp`) | 0.46 T/s |

- 多份同时发出的 GEMV 会互相争 DRAM,q_proj 只有 60 GB/s;按它逐层相加得到 14.1 ms/token,反而大于 mlx-lm 实测的整步时间。
  所以孤立测得的 decode kernel 时间**不能直接相加**,gap 分解改用下面的消融实测。
- 小 kernel 的"独立多份"数字不可用:cache 写入那一项在独立模式下会把整个 cache 复制一遍(测量写法问题),只用依赖链的数字。

**上界补全**(第 14 项)。

- exp 作为独立资源接入物理上界(`engines.gpu.special.exp`,与 FMA 时间取 max,不相加):每个 attention score 一次 exp。
  prefill 32k 时 exp 的 slack 远大于 10,不会成为瓶颈。
- `bound.ttft(L)` 与 `bound.request_time(L, N)`,CLI `python -m roofline ... request --prompt-len L --gen N`。
  例:L = 4096、N = 512 时 T_request ≥ 9.92 s,其中 prefill 占 15%。

**基线**(`benchmarks/baseline_mlx.py`、llama-bench;第 15 项)。随机 token 的 prompt,贪心采样,BF16,单流,burst:

| 场景 | 物理上界 | mlx-lm 0.32 | llama.cpp(brew 0.6.0,Metal,FA auto) |
|---|---|---|---|
| decode ctx 512 | 12.2 ms | 12.8 ms(95%) | 14.0 ms(87%) |
| decode ctx 4k | 16.2 ms | 17.9 ms(91%) | 19.6 ms(83%) |
| decode ctx 16k | 30.0 ms | 32.5 ms(92%) | 35.0 ms(86%) |
| decode ctx 32k | 48.2 ms | 53.6 ms(90%) | 未测 |
| TTFT L 16 | 11.6 ms | 35 ms(33%) | 22 ms(53%) |
| TTFT L 512 | 131 ms | 197 ms(67%) | 186 ms(70%) |
| TTFT L 4k | 1.50 s | 1.92 s(78%) | 2.26 s(67%) |
| TTFT L 16k | 12.3 s | 15.8 s(78%) | 23.6 s(52%) |
| TTFT L 32k | 41.3 s | 55.8 s(74%) | 470 s(9%) |

- llama.cpp 在 32k 处断崖:16k 为 695 tok/s,32k 只有 70 tok/s。FA 设为 auto 时实际开着(-fa 1 与 auto 相同,-fa 0 为 533 tok/s),
  所以断崖发生在它的 flash-attention 路径内,原因未查。decode 32k 需要先做一次 32k 预填充(约 8 分钟),这次没测。
- GGUF 由 `scripts/convert_qwen3_gguf.py` 从 HF 权重转出(不依赖 PyTorch;1 维 norm 权重按上游做法存 F32),
  生成结果正确("The capital of France is Paris. ...")。
- mlx-lm 每个 decode step 有 536 个 GPU kernel(197 个 matmul、113 个 RMSNorm、56 个 RoPE、56 个 cache 写入、
  56 个残差加、28 个 SDPA、28 个 SwiGLU、gather 与 argmax),Python 建图 0.53 ms/token。
  prefill 把前 L − 1 个 token 分成 ≤ 2048 的块(每块后求值 cache 并清空分配器缓存),最后一个 token 单独算 logits。

**gap 分解**(`benchmarks/ablation_mlx.py`、`src/roofline/gap.py`、`scripts/gap_report.py`;第 16 项)。
孤立测得的 kernel 时间不能直接相加(见上),所以 mlx-lm 的时间用**消融**拆开,各段实测、首尾相接,加起来正好等于实测总时间:

- decode:只有 197 个 matmul(保留依赖)→ 再加 SDPA → 完整 forward(加小 kernel 与 argmax 反馈)→ mlx-lm `stream_generate`;
- prefill:按 mlx-lm 的分块方式,只有 GEMM → 再加 causal SDPA → 完整 forward → `stream_generate` 的 TTFT。

每段与它在物理上界中的理想份额(字节 / B_dram 或 FLOPs / P_peak)相减,得到该段的损失。
各变体交错跑 3 轮取最小值(L > 4096 时只跑 1 轮),以免热状态漂移只落在某一个变体上。
`model` 列是用校准参数预测的同一段,各段之和即 mlx-lm 这个方案的 T_A(plan)。图:`results/figures/gap_mlx.png`。

| 场景 | 上界 | 实测 | 效率 | matmul 损失 | attention 损失 | 小 kernel 损失 | engine 损失 | T_A(model) |
|---|---|---|---|---|---|---|---|---|
| decode 512 | 12.18 ms | 13.69 ms | 89% | 0.61 | 0.70 | 0.11 | 0.09 | 15.13 |
| decode 4k | 16.18 ms | 18.36 ms | 88% | 0.95 | 0.50 | 0.60 | 0.12 | 19.77 |
| decode 16k | 29.91 ms | 34.36 ms | 87% | 1.07 | 2.74 | 1.70 | −1.05 | 34.47 |
| decode 32k | 48.20 ms | 50.67 ms | 95% | 0.95 | 1.91 | 0.61 | −1.01 | 53.90 |
| prefill 16 | 11.6 ms | 30.7 ms | 38% | 9.5 | 0.3 | 6.0 | 3.2 | 36.5 |
| prefill 64 | 15.5 ms | 45.8 ms | 34% | 17.0 | 3.2 | 5.9 | 4.2 | 42.9 |
| prefill 512 | 131 ms | 192 ms | 68% | 34.7 | 7.4 | 14.0 | 5.5 | 189 |
| prefill 1k | 278 ms | 383 ms | 73% | 56.4 | 17.5 | 17.0 | 14.2 | 372 |
| prefill 4k | 1.50 s | 1.99 s | 75% | 216 | 189 | 53 | 30 | 1912 |
| prefill 16k | 12.3 s | 18.8 s | 65% | 977 | 3798 | 2022 | −262 | 15748 |
| prefill 32k | 41.3 s | 70.7 s | 58% | 2683 | 22479 | 3489 | 733 | 53430 |

(decode 单位 ms/token;prefill 损失单位 ms。负值是噪声:同一段在不同变体里测两次,差值在实测总时间的 ±3% 以内。)

- **decode:mlx-lm 已到上界的 87~95%。** 剩下 1.5~4.5 ms/token:
  - matmul 0.6~1.1 ms,比模型预测的 1.57 ms(t₀ 固定成本 0.48 + 按 B_eff = 93.9 GB/s 算的带宽损失 1.09)小。
    MLX 的 GEMV 实际流速约 97 GB/s,快于合成的读 kernel。
  - attention 0.5~2.7 ms,来自 decode SDPA 只跑到 49~94 GB/s;上下文越长,这一项越大。
  - 小 kernel 0.1~1.7 ms;模型按 311 × t₀ 估 0.76 ms。把它们全部融合,最多省这么多。
  - engine 约为 0:mlx-lm 的 async_eval 流水线把每 token 的 CPU(建图 0.53 ms + 编码约 1.3 ms)和同步(148 µs)都藏在 GPU 时间后面。
- **短 prompt prefill(16、64):只有 34~38%。**
  - matmul 损失占 31~37%:M = 16/64 的 GEMM 只有 0.6~3 TFLOP/s,读权重也读不满带宽。
  - 小 kernel 占 13~20%(约 6 ms),engine 占 9~11%(3~4 ms:每块两次求值、清空分配器缓存、采样同步)。
  - 模型对后两项低估(3.3~3.8 ms 与 0.3 ms):每块的额外开销没有进入模型。
- **中等长度(512~4k):68~75%。** GEMM 效率损失 11~18%(模型与实测差 30% 以内),attention 5~9%,小 kernel 3~7%,engine 1~3%。
  T_A(model) 与实测在 64~4k 范围内相差 7% 以内。
- **长 prompt(16k、32k):58~65%,attention 主导**(占实测时间的 20% 与 32%)。
  - 实测的 attention 损失(3.8 s、22.5 s)远大于按孤立 SDPA 表预测的值(2.5 s、10.3 s)。
  - 原因至少有两个,尚未分开:分块的 causal SDPA 在真实调用里的效率,以及长时间运行的热降频。
    同一 32k TTFT,本次(连续测了约 40 分钟之后)为 70.7 s,早先较冷时的基线为 55.8 s,差 27%。
- **模型的整体表现:** decode 的 T_A 高估 0.3~11%:孤立代价假设没有重叠,而实际上小 kernel、相邻 matmul 互相重叠。
  prefill 的 T_A 在 16k、32k 低估 16~24%(attention 与热状态)。
- **llama.cpp 没有做分解**(无法对它做同样的消融),只列总数(上表的基线一节)。它在 decode 上比同为较冷时测的 mlx-lm 基线低 5~8 个百分点;
  prefill 在 4k 以上落后更多,32k 断崖。

**对导航的含义:**

- decode 的上界已基本达到,剩下的大头在长上下文的 attention 带宽。
- prefill 有两段可改:短 prompt 的固定开销(小 GEMM 效率、小 kernel、分块与同步开销),和长 prompt 的 attention。
- 热状态对长 prompt 的影响与 attention 损失同量级,第 17 项(sustained 档)应提前。

`decode_plan_time(...)` 可以对假想方案估 T_A,例如"融合全部小 kernel""把 q/k/v 合成一个 matmul"。
在 decode 下,这类改动的模型收益上限约为 (省掉的 kernel 数) × t₀。

### 6.10 性能上界 vs 快存储容量 S(2026-10-06)

传统 roofline 问"这个实现的 AI 是多少、瓶颈在哪";这里问"这张计算图在给定存储与硬件下,任何实现能达到的最好
计算 / 通信权衡是什么"。所以 AI 不再是自变量,而是由计算图与快存储容量 S 推出的派生量:

```
AI_max(S) = FLOPs / Q_min(S)
perf(S)  ≤ FLOPs / max(FLOPs / P_peak, Q_min(S) / B)
```

`iobound.traffic_at(graph, S, P)` 对任意 S 给出 Q_min(S)(整图界与算子边界界);图由 `scripts/plot_vs_capacity.py` 生成
(`results/figures/perf_vs_capacity.png`):两层模型,一个处理器,S 从 1 KiB 到 4 GiB,DRAM 102.7 GB/s,
竖线标出 M4 各级片上存储(threadgroup 32 KiB、每核 SRAM 估计 512 KiB、GPU L2 约 3 MiB、L2 + SLC 12 MiB)。

- **decode(ctx 4k):** AI ≈ 1.3,从 1 KiB 到约 256 MiB 几乎不变;直到 S 能放下大部分权重与 KV(约 1 GiB 以上)才上升,
  S ≥ 权重 + KV 时稳态不再有 DRAM 流量。任何片上存储大小都不能让单流 decode 变成算力瓶颈。
- **prefill,算子边界界:** AI 随 S 约按 √S 增长(GEMM 的 pebble 项),在 S ≈ 5~10 KiB 处越过机器平衡点 36;
  M4 的 threadgroup memory(32 KiB)已经在这之上,所以 prefill 在任何一级都是算力瓶颈。
- **prefill,整图界:** 加入 6.12 的融合 pebble 界后,小 S 时也约按 √S 上升,在必读必写占上风处变平
  (L 512 约 16 KiB,4k 约 300 KiB,32k 约 1 MiB);在所有 S 下都高于机器平衡点。

下一步:第二个维度用可用并行度 / 在途访存请求数(带宽随并发而变),需要在 Q_min(S) 之外加一个并发约束。

### 6.11 第二维:可用并行度(2026-10-06)

6.10 的上界假设带宽与算力总能跑满。实际上二者都要足够的并发:按 Little 定律,
延迟为 L 的存储系统在 n 个请求在途时最多提供 `B(n) = min(B_max, n · 16 B / L)`。
于是上界变成两个变量的函数:

```
perf(S, T) ≤ FLOPs / max( FLOPs / P(T),  Q_min(S) / B(T) )
```

T 为并发线程数。B(T)、P(T) 由 `benchmarks/concurrency.py` 实测(每线程 1 个或 4 个 16 B 读在途;每线程 64 条 FMA 链)。
图由 `scripts/plot_capacity_parallelism.py` 生成(`results/figures/perf_capacity_parallelism.png`),每个场景按自己的最好上界归一化。

| 量 | 结果 |
|---|---|
| Little 定律拟合(每线程 1 个读在途) | L = 290 ns(T = 32~1024 各点在 282~303 ns),B_max = 103.6 GB/s |
| 跑满 DRAM 所需在途量 | 约 29 KiB(约 1.8k 个 16 B 请求) |
| 指针追逐延迟 | 39 ns(≤ 128 KiB)、98 ns(1 MiB)、207~225 ns(8~64 MiB)、368 ns(256 MiB,含 TLB 缺失) |
| 带宽到 50% / 90% | T ≥ 512 / 1024(每线程 4 个读在途) |
| 算力到 50% / 90% | T ≥ 2048 / 32768;T ≤ 256 时只有一个 threadgroup,只用到一个核 |

- **decode(ctx 4k):** 图上是水平色带,与 S 基本无关(直到 S 接近能放下权重的 100 MiB 以上)。
  只要 T ≥ 约 1k(约 29 KiB 在途)就到最好上界的 90%。
  对方案代价的含义:每个依赖 kernel 都要尽快把在途量拉到约 29 KiB;
  barrier 之后的爬坡与收尾正是 6.9 中 t₀ 的一部分。
- **prefill(L 4k):** S 只在约 4 KiB 以下起作用(算子边界下的 GEMM pebble 项);
  到 90% 需要 T ≥ 约 16k~32k 个并发线程,比带宽多一个数量级。
- **算力比带宽难喂。** 跑满带宽要约 1k 线程,跑满算力要约 32k 线程。
  所以小 GEMM(prefill 短 prompt,M = 16~64)效率低,并不是带宽问题,而是给不出足够的并行线程(6.9 的 P(shape))。
- **局限:** 并行度目前是机器侧的量(实现保持了多少并发)。计算图本身对可用并行度的限制
  (例如 decode 的依赖链、GEMV 的输出维度)还没有接入 Q_min(S) 的推导。

### 6.12 整图界超出必读必写的部分:融合 + 重算下的 pebble 界(2026-10-06,计划 10a)

实现:`iobound.fused_contractions`、`recompute_capacity`、`fused_pebble_elements`;整图界现在取 max(必读必写, 本节的界)。
**以下是本项目自己的推导,未经同行评审。**

**设定。** 选择 ③ 下的任意实现:融合、求和顺序、softmax 写法、重算都自由。
把一个处理器的执行按 I/O 次数切成段,每段 R 次 I/O。一段能用的"外来值"集合 D(段开始时在快存储里的,加上段内读进来的)至多
X = S + R 个,并且 D 支配段内算出的一切(Hong–Kung 的分段论证允许重算)。

**计入的乘积。** 激活输入位于残差更新之后的矩阵乘:各层的 gate_up 与 down,以及第 1 层起的 qkv。
o_proj、attention、lm_head、第 0 层 qkv 不计入;少计乘积只让下界更弱。

**每段的乘积上限。** 收缩 c 的一个有用乘积所属的输出,要么在段内由段内算的乘积全部完成("完成":需要整行 A 与整列 W,
完成的输出可以直接在片上被下游消费,**不必写回**,这正是融合的好处),要么以部分和的形式跨过段边界
(段开始时在 D 里,或段结束时在快存储里 / 被写出)。于是

```
c 的乘积 ≤ sqrt(α_c β_c γ_c) + α_c β_c / K_c
```

(未完成的部分用 Loomis–Whitney)。其中 β_c 是 D 中 c 的权重,γ_c 是 D 中 c 的部分和加上离开的部分和
(所有收缩合计 ≤ S + R),α_c 是段内可用的 c 的 A 值。

**重算的代价。** 一个 A 值可用,要么它私有链上的某个顶点在 D 里,要么在段内从头重算。从头重算需要:

| 收缩 | 每个 A 列需要的上游权重 ω | 每行的前置条件 |
|---|---|---|
| qkv_l | Wd_{l−1}[:,k] 与 Wo_{l−1}[:,k],共 5120 | act_{l−1}[i,:](每列 j:一个私有顶点,或 Wg/Wu[:,j] 共 2048)与 attn_{l−1}[i,:](每个 V 列:≥ 行数个 V 值,或 Wv 一列 1024) |
| gate_up_l | Wo_l[:,k],2048 | attn_l[i,:](同上) |
| down_l | Wg_l[:,j] 与 Wu_l[:,j],2048 | h2_l[i,:](每列 k:一个私有顶点,或 Wo_l[:,k] 共 2048) |

所以 α_c ≤ X + rec_c(X),rec_c(Y) = max_r r · min(K, (Y − cost(r))/ω) 是预算 Y 最多能从头重算的 A 值个数
(行数 × 列数;cost(r) 分段线性,对连续松弛精确求最大)。省略的前置条件只会让 rec_c 变大,不影响正确性。

**合成。** 不同收缩的权重、部分和是不同的顶点(合计 ≤ X),由 Cauchy–Schwarz,一段内所有计入乘积至多

```
F(X) = X · sqrt(X + r*) + (X + r*) · X / K_min,     r* = max_c rec_c(X)
```

N 为计入乘积总数,P 个处理器各自分段再求和,则对任意 R,流量 ≥ R · (N / F(S + R) − P),对 R 取最大。

**验证**(`tests/test_iobound.py`):

- rec_c 的解析最大值与整数暴力搜索一致;
- 对 L = 4k / 32k、S = 4K~2M 个元素、P = 1 / 10,下界都不超过逐算子分块 GEMM 这个可行方案的流量;
- 下界随 S 单调不增;decode 下可以忽略(< 万分之一的权重流量)。

**结果**(Qwen3-0.6B,prefill;core 边界每核 S = 512 KiB、P = 10;DRAM 边界 S = 17 MiB):

| L | core 边界:必读必写 | core 边界:本节的界 | DRAM 边界:本节的界 | core 边界:算子边界界 |
|---|---|---|---|---|
| 4k | 1.66 GB | 1.21 GB | 0.03 GB | 13.8 GB |
| 8k | 2.13 GB | **2.43 GB** | 0.05 GB | 29.3 GB |
| 16k | 3.07 GB | **4.87 GB** | 0.07 GB | 60.8 GB |
| 32k | 4.95 GB | **9.74 GB** | 0.09 GB | 135 GB |

- **整图界第一次超过了必读必写**:在 core 边界、prompt 约 8k 以上,32k 时约为 2 倍
  (core 边界的整图 AI 上限从 30740 降到 15612 FLOP/byte)。
- **DRAM 边界不起作用。** 17 MiB 的快存储足以放下一整块重算所需的权重(r* 达到 m·K),界退化到接近 0。
- **与算子边界界仍差约 14 倍。** 真实的最小流量在两者之间;差距来自本节较松的地方:
  完成的输出不必写回、所有收缩共用一个预算、α 允许把 D 全部算作私有 A 值。
- **性能 vs S 图(6.10)随之改变:** 整图界在小 S 时也按约 √S 上升。例如 prefill 4k:
  S = 1 KiB 时 AI 上限 109(算子边界 17),32 KiB 时 723(95),512 KiB 时 3330(354)。
  prefill 在所有 S 下仍是算力瓶颈。
- **对 M4 + Qwen3-0.6B 的结论不变:** core 边界 9.74 GB / 770 GB/s ≈ 13 ms,远小于 41.3 s 的计算时间。

**局限与下一步:**

- attention 与 o_proj 的乘积还没计入(需要处理 Q/K/V 的上游重算,6.6 末尾的 Menger 问题)。
- 按收缩分别分配预算(而不是共用 max)可能把界收紧数倍。
- 每行前置条件只往上游推了一到两层;推得更深可以让 DRAM 边界也有非平凡的界。

### 6.13 KV 项的结构下界与 KV 存储方案(2026-10-06)

实现:`src/roofline/kvpolicy.py`;`workload.decode(..., kv=policy)`;CLI `decode --kv ... --kv-bits ...`;
测量 `benchmarks/key_clusters.py`;测试 `tests/test_kvpolicy.py`。

**来源。** 四篇关于流式注意力的下界 / 算法论文:

| 论文 | 结论 | 用处 |
|---|---|---|
| Haris & Onak 2025(arXiv 2502.15955) | d = Ω(log n) 时,对每步注意力输出做 (1±η) 近似也要 Ω(nd) bits;99% 的 score 近 0 但位置未知时仍然成立;固定滑窗 W 为 Θ(dW);访问模式预先固定的算法每步要 Ω(nd) 时间,自适应的(检索)可做到 O(d√n) | 主要依据。Qwen3 的 head_dim 128 ≫ ln(32k) ≈ 10.4,处在 Θ(nd) 区间,低维的 e^d 界不相关 |
| SubGen(Zandieh et al. 2024,arXiv 2402.06082) | keys (m, δ) 可聚类且 ‖q‖ ≤ r 时,内存与每步时间 O(d(mt + s)),t = ε⁻² e^{2δr} log n,s = ε⁻² d log n | 可聚类这一档的形状;需要实测 m 与 δr |
| Chen et al. 2025(arXiv 2503.11108) | 空间界推广到 tensor attention;存预计算量与现算之间的时间—内存取舍 | 引出下面的"KV 存储方案"检查 |
| van den Brand et al. 2023(arXiv 2304.02207) | 动态维护注意力(逐元素更新)的摊还更新 / 查询时间,条件最优 | 只有 ω 指数、无常数;对当前 estimator 基本不用,prefix 编辑 / KV 复用时再看 |

**KV 策略(B/C 级的 knob)。** 每个策略给出:每步每个 KV head 读多少个缓存 key(`k_read(n)`)、多少个 value(`v_read(n)`)、
常驻多少(`stored(n)`)、每元素位数(`bits`)、是否自适应。`full()` 即 A 级(默认,数值与原公式完全一致)。

| 策略 | 依据 | 形状 |
|---|---|---|
| `full(bits)` | 无结构:Ω(nd) bits | 随 n 线性;只能靠位数改斜率(B 级)。"不知道稀疏位置"的压缩改变不了增长 |
| `window(W, sink)` | 固定访问模式:Θ(dW) | n > W + sink 后饱和 |
| `subgen(m(n), δr, ε)` | SubGen 定理(常数取 1) | 有可证保证;下面实测表明对 Qwen3-0.6B 不适用 |
| `budget(k, v)` | 经验做法(k-center + 最近 token、H2O 等) | 无保证,质量另测 |
| `retrieval(k, page)` | 自适应:读页摘要 + top-k | **读的少于存的**:内存仍随 n 线性,每步流量不是 |

空间下界约束的是常驻状态(内存容量),每步流量由时间下界约束:访问模式固定的方法每步仍 Ω(nd),
所以 `retrieval` 单独建模为 stored > read,避免把省下的内存误当成省下的带宽。
Ω/O 的常数未知,这些策略只定 KV 项的**形状**,不是数值下限;误差模型是每步输出的 (1±η),与任务质量预算不同,C 级仍需质量模型。

M4 + Qwen3-0.6B,context 32k 的 decode 物理上界(`python -m roofline ... decode --context 32768 --kv ...`):

| 策略 | 上界 | kv_read | KV 常驻 |
|---|---|---|---|
| A 级(full, BF16) | 48.2 ms | 3.76 GB | 3.76 GB |
| `--kv-bits 4` | 20.8 ms | 0.94 GB | 0.94 GB |
| `--kv window:4096` | 16.2 ms(= 4k 上下文的 A 级上界) | 0.47 GB | 0.47 GB |
| `--kv retrieval:512`(page 16) | 14.5 ms | 0.29 GB | 3.99 GB |
| `--kv budget:2048` | 13.9 ms | 0.23 GB | 0.23 GB |

**SubGen 的常数在 32k 以内无效。** ε = 0.1 时 s = 100 · 128 · ln n > n(n ≤ 32k),可证版本保留全部 value,
不比精确注意力便宜;取 m = 64、δr = 0.5,ε = 0.3 时要 n ≈ 10⁶ 才把 KV 降到 3%,ε = 0.5 时 32k 降到 27%。

**实测:Qwen3-0.6B 的 key 不可聚类(在 softmax 有意义的尺度上)。** `benchmarks/key_clusters.py`:
prefill 后取每层每个 KV head 的 key(k-norm 与 RoPE 之后),greedy farthest-first(Gonzalez,半径在最优的 2 倍内)求覆盖半径;
r = 该 KV head 下两个 query head 在所有位置上的 max ‖q‖ · head_dim^-½(缩放并入 r)。
仓库语料(text)与随机 token(random),n = 1k ~ 8k,centers 最多 n/2:

| | r(224 个 head) | k = n/2 时 δr 最小 / 中位 | k = 64 覆盖 90% key 的 δr 中位 | δr ≤ 2 的 head |
|---|---|---|---|---|
| text,n = 1k → 8k | 1.5 ~ 4.3,中位 2.2 | 19.9 → 18.6 / 36.7 → 35.5 | 51 → 60 | 0 |
| random,n = 1k → 8k | 同上 | 18.1 → 17.1 / 34.6 → 33.4 | 46 → 52 | 0 |

- SubGen 需要 δr = o(log n),实用上 δr ≲ 2(e^{2δr} 进 t);实测即使一半 token 当中心,最好的 head 也是 δr ≈ 17(e^{34}),
  且不是少数离群点造成的:覆盖 50% / 90% key 的半径与最大半径同量级。
- 覆盖 90% 的半径随 n 增大(k 固定为 64 时 51 → 60),没有饱和;text 与 random 几乎一样,说明 key 的分布主要由 RoPE 与 k-norm 决定,不是由内容聚类。
  (SubGen 在 Llama-2 上用 t-SNE 看到的"可聚类"是 2 维投影的直观印象,不等于定理要求的尺度。)
- 结论:对本模型,**可证的**亚线性 KV 只剩固定访问模式(窗口)与降位数两条;`budget` / `retrieval` 只能作为带质量代价的 C 级 knob。

**A 级的补充检查:KV 存储方案。** "缓存 KV"只是一个实现选择。缓存每层的输入 hidden state、每步重算 K/V,与原模型数学等价
(每个乘积项至少算一次,9 节的 A 级定义允许重算),所以与实现无关的上界应当对存储方案取最小。
Qwen3-0.6B 每层每 token:K+V 2048 个元素,hidden 1024 个,存 hidden 让 KV 流量减半;代价是重算 K/V 投影,每省 1 字节多 2048 FLOP,
而 M4 的机器平衡点约 4.0e12 / 120e9 ≈ 33 FLOP/B,差约 60 倍。32k 时省约 15.7 ms 的读取,多约 0.96 s 的计算。
只缓存 token id、整段重算更差。`kvpolicy.best_storage_plan` 在 512 / 4k / 32k 上都选 `kv`,所以**上界数值不变**,但这一排除现在有代码和测试。
省下的字节与多出的 FLOPs 都与 batch × context 成正比,比值 2·hidden·2·kv_dim / ((2·kv_dim − hidden)·字节数) 与 batch 无关;只有机器平衡点接近这个比值的设备(或 hidden ≪ 2·kv_dim 的模型)才需要重新看。

## 7. 计划

已完成:

1. [x] `configs/hardware/m4_mba_24g.yaml`:device spec,实测字段已填(2026-10-04)。
2. [x] `configs/models/qwen3_0.6b.yaml`:model spec schema。
3. [x] `src/roofline/`:decode 与 prefill 的 `physical` / `operator` 上界,输出瓶颈和 slack。
4. [x] `benchmarks/`:`bandwidth.py`、`compute.py`、`contention.py`,用 MLX 自定义 Metal kernel 校准 spec。
5. [x] attention 的 I/O 下界补出显式常数,并接入 `operator` 界(6.6,2026-10-06)。

下一步(当前优先):计算图 + pebble game → roofline(6.7):

6. [x] **计算图**(`src/roofline/graph.py`):decode 与 prefill 的算子级图,FLOPs 与 `workload.py` 一致,
   必读必写字节的差别见 6.4(2026-10-06)。
7. [x] **I/O 下界与合成**(`src/roofline/iobound.py`):算子边界界的求和规则已证明;
   整图界目前只有必读必写,超出部分的难点与方向见 6.8(2026-10-06)。
8. [x] **roofline 图**:`results/figures/roofline_dram.png`(DRAM 边界,整图界与算子边界界,外加各算子)(2026-10-06)。
9. [x] **验证**:元素级小图的精确求解器、显式方案上界、整图界退化为物理流量(`tests/test_iobound.py`)(2026-10-06)。
10. [x] **内层实测与分层 roofline**:`benchmarks/hierarchy.py`;`results/figures/roofline_hierarchy.png`(2026-10-06)。
10a. [x] **整图界超出必读必写的部分**:融合 + 重算下的 pebble 界(6.12,2026-10-06)。core 边界、长 prompt 时超过必读必写约 2 倍;DRAM 边界不起作用。
10b. [ ] **收紧 10a**:按收缩分别分配预算、计入 attention 与 o_proj、把每行前置条件往上游推得更深。

之后(A 级、单流,decode 与 prefill 并行推进):

11. [x] **依赖 kernel 链拟合**(decode;6.9,2026-10-06):一个 command buffer 里串 N 个有依赖的读 kernel,每个读 b 字节(64 KB ~ 64 MB),
    每次同步前提交 K 个 buffer;拟合 `T(N, b, K)` 得到 t₀、B_eff、c_commit、c_sync(5.3)。
12. [x] **拟合检验**(6.9):看残差是否随 b 非线性(需要 B_eff(b));必要时另测无依赖 kernel 并发时的 t₀。
13. [x] **算力效率曲线 P(shape)**(prefill;6.9):按 Qwen3-0.6B 的实际 GEMM 形状与 L 扫描;attention kernel 按 L 扫描;exp 峰值吞吐。
14. [x] **上界补全**(6.9):exp 作为资源接入;输出 TTFT(L) 与 T_request(L, N)。
15. [x] **基线**(6.9;llama.cpp 的 decode 32k 未测):mlx-lm 与 llama.cpp(Metal),decode 在 context 512 ~ 32k、prefill 在 prompt 16 ~ 32k 上跑;
    一次 trace 同时抓两段,取 kernel 数、每个 kernel 的耗时、字节数与 FLOPs。llama.cpp 需另配环境。
16. [x] **gap 分解**:实现 T_A(plan),按 5.4 把 `T_baseline − T_phys` 拆成各项与残差,decode 与 prefill 分别给出(6.9;mlx-lm 用消融实测,llama.cpp 只有总数)。
16c. [x] **性能上界 vs 快存储容量 S**(6.10,2026-10-06)。
16d. [x] **第二维:可用并行度 / 在途访存请求数**(6.11,2026-10-06)。
16e. [ ] **计算图侧的可用并行度**:依赖链与每个算子的并行宽度对 T 的上限,接入 (S, T) 上界。
16a. [ ] **长 prompt attention 的损失来源**:把分块 causal SDPA 的真实调用效率与热降频分开(固定温度条件,或交错的冷/热两组)。
16b. [ ] **prefill 每块开销**:短 prompt 下小 kernel 与 engine 被模型低估(每块两次求值、清空分配器缓存),补进 T_A。

以后:

17. [ ] sustained 档:连续负载数分钟,测热降频后的带宽与算力。长 prompt prefill(32k 约 41 s)本身就会进入降频区间。
18. [ ] B 级(量化)、C 级(投机解码、KV 压缩)、batching、多引擎分配、MoE。
18a. [x] **KV 项的结构下界**(6.13,2026-10-06):KV 策略 knob(full / bits / window / subgen / budget / retrieval)、KV 存储方案检查、Qwen3-0.6B 的 key 可聚类性实测(不可聚类)。
18b. [ ] **KV knob 的质量表**:`budget` / `retrieval` / 降位数在 v3 prompts 上的输出偏差(与 v3 的 divergence 指标同口径),补成 C 级的"knob → 质量损失"表。

## 8. 实测结论(2026-10-03 ~ 10-06)

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
| L2 → 核心 | 700~774 GB/s(0.4~2.9 MiB),4 MiB 起降到 SLC(480 → 150),约 16 MiB 到 DRAM 速度 | `hierarchy.py`(2026-10-06):160 group,每个 group 读整个 buffer(起点错开),一次 launch,best of 9 |
| 每核可见 cache | 1 group/核时 128~176 KiB 以内约 1.35 TB/s;4 group/核时降到 48 KiB 以下 | 同上,10 / 40 group;说明寄存器与 cache 共用每核 SRAM(Dynamic Caching) |
| threadgroup memory | 690~890 GB/s(合计) | 同上,每 group 16 KiB,反复读 |
| 依赖 kernel 链 | e = 2.46 µs,c_commit = 17.4 µs,t₀ = 2.44 µs,B_eff = 93.9 GB/s,c_sync = 148 µs | `kernel_chain.py` + `fit_kernel_chain.py`(2026-10-06),见 6.9 |
| exp 峰值 | 0.46 T exp/s | `compute_shapes.py`,每线程 8 条 exp + fmul 链 |
| mlx `matmul` 在 Qwen3 形状上 | M ≥ 512:3.15~3.26 TFLOP/s;M = 16:0.6~1.2 | `compute_shapes.py`,见 6.9 |

由此得到的 Qwen3-0.6B BF16 物理上界见 5.1(decode)与 5.2(prefill)。

注意:106 µs 的往返如果每 token 一次,约占 4096 上下文物理上界 16.2 ms 的 0.7%;
若每层一次(28 层)则约 3 ms,接近 20%。但同一 command buffer 内的 dispatch 应远便宜于完整往返,
所以"每层一次"的估计偏悲观,需要计划第 11 项的拟合把它拆开。

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
- 多处理器指 GPU 的 10 个核心(私有寄存器 / threadgroup / L1,共享 L2 / SLC / DRAM),不是 CPU、GPU、ANE 三个引擎(2026-10-06)。
- 第一张 roofline 图同时画整图界与算子边界界;整图界中尚未证明的部分用保守的 max 并标注(2026-10-06)。
- A 级"数学等价"取"只固定乘积项":每个收缩(矩阵乘、QK^T、PV)的每个乘积项 a_ik·b_kj 至少算一次,
  求和顺序、softmax 写法(含 online softmax)、融合、重算都自由。I/O 下界只能用乘积结构与必读必写推出(2026-10-06)。
  其他候选:固定图且不许重算(重算的实现可能 I/O 更低,上界不成立);固定图允许重算(排除 split-K、online softmax);
  再加代数重写(如 P(VWo)、权重吸收;Qwen3 上被 RoPE、q/k-norm 挡住或使 FLOPs 与 KV 变大,无收益);
  任意算法(如 Strassen;pebble 项基本消失,退回物理上界,归 C 级)。

## 10. 待确认

- T_A(plan) 各项的组合方式:CPU 编码与 GPU 执行已确认在多 buffer 时重叠(6.9 的流水线模型);
  相邻无依赖 kernel 之间、小 kernel 与 matmul 之间也有重叠,孤立测量相加会高估(decode 高估 0.3~11%),尚未建模。
- 热状态:同一 32k TTFT 在不同时刻相差 27%(6.9)。上界与基线比较应在相同热状态下进行。
- 上界与代价模型使用 burst 还是 sustained 实测值。长 prompt prefill 持续数十秒,burst 值可能偏乐观。
- 基线 kernel trace 的获取方式:Metal System Trace、MLX 的 Metal capture,还是自己插 GPU 时间戳。
- gap 分解的残差多大可以接受,超过时怎么定位漏掉的因素。
