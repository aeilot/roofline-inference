# 项目思路与设计笔记

本文整理自项目初期的讨论,记录项目的真实目标、设计含义和待办。
最后与代码和实测同步:2026-10-04。

## 1. 项目目标

**预估 LLM inference 性能的理论上限。**

- 输入:**model spec**(模型结构)+ **device spec**(设备规格及系统底层信息)。所有能得到的信息都可以使用。
- 输出:与具体实现无关的理论性能上限(decode 的 tokens/s、prefill 的 TTFT、吞吐等),以及瓶颈归因。
- **现有算子是下界**("现在能做到的"),roofline 上限是上界,二者之间的 gap 就是优化空间。
- **最终用途**:让 agent 在比现有算子**更宽松的约束**下搜索最优方案,**定制 inference engine**。

> 注意:这不是"表征现有 kernel 落在 roofline 哪里"的分析工具。实测只用来校准下界和设备参数,不是核心。

## 2. 当前目标硬件

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
| GPU 可见片上容量 | 实测约 12 MiB(GPU L2 + SLC,未公开;见第 7 节) |

## 3. Apple Silicon 的建模要点

1. **统一内存,带宽由 CPU/GPU/ANE 共享。** 理论总带宽约 120 GB/s(LPDDR5X-7500,128-bit)。每个引擎单独可达的带宽低于总量,同时使用多个引擎时总量封顶。spec 里需同时有"总带宽"和"各引擎可达带宽上限"。
   实测:GPU 单独读 102.7 GB/s;CPU 单独 46~54 GB/s;CPU + GPU 同时读 89.6~92.0 GB/s,不超过 GPU 单独,所以 GPU 一个引擎就能吃满 DRAM。
2. **offload 成本近似为 0。** 没有 HBM/PCIe 之分,算子可按引擎自由分配,但引擎之间抢同一份带宽。这是 agent 放宽约束后的最大自由度。
3. **decode 基本被带宽锁死。** 上限约为 `带宽 / (权重字节 + KV 字节)`。例如 8B 模型 4-bit 约 4.5GB,上限约 26 tok/s。主要优化杠杆是量化位宽、KV 压缩、投机解码。
4. **容量约束比带宽更早出现。** 24GB 中 GPU 可用 wired 内存约 16~18GB,模型加 KV cache 必须装得下,属硬约束。
5. **无风扇,存在热降频。** spec 需区分 `burst` 与 `sustained` 两档。
6. **三套计算引擎。** GPU(FP32/FP16 约 4 TFLOPS 量级;实测 BF16 3.68、FP32 3.65 TFLOP/s)、CPU(NEON + SME)、ANE(宣称约 38 TOPS,INT8 为主)。按引擎和精度分表。

> 上述数值部分来自公开资料,部分为估算,**不能直接当 ground truth**。以 device spec 的 `measured` 字段和第 7 节为准。

## 4. 设计含义

1. **输入只有两份 spec,不依赖实测。** `benchmarks/` 仅用于校准下界与设备参数。
2. **上限是多资源约束的 min,而非单一 roofline:**
   ```
   T >= max( FLOPs/P_peak, Bytes_hbm/B_hbm, Bytes_link/B_link, ... )
   ```
   并需考虑**最优融合下的最小数据移动**:权重必读一次、KV 必读一次、中间激活理论上可留在片上。这不同于按算子独立统计流量。
3. **约束要显式参数化。** agent 放宽约束,模型必须能回答"放开 X 之后上限变成多少"(可参数化项:融合范围、片上容量、精度、并行度、引擎分配等)。
4. **输出对 agent 友好。** 除数字外,给出瓶颈归因、各约束的松弛度(slack)、结构化结果,而非只出图。`achieved / bound` 作为效率评分和 reward。
5. **spec 数值分两类并标注来源与置信度:**
   - **datasheet 值**:官方/公开规格(核数、缓存、标称带宽),直接写入。
   - **微基准实测值**:带宽(STREAM 风格)、GEMM FLOP/s、各引擎可达带宽,在本机实测后覆盖。

## 5. 计划

1. [x] `configs/hardware/m4_mba_24g.yaml`:按上述结构编写 device spec。实测字段已填(2026-10-04)。
2. [x] `configs/models/`:定义 model spec schema。目前只有 `qwen3_0.6b.yaml`,MoE 参数尚未使用。
3. [x] `src/roofline/`:实现 decode 与 prefill 上限计算,输出瓶颈归因和各约束 slack。
   已有 `physical`(第 1 层)与 `operator`(第 1 层 + 有限片上容量,见 6.5)两种上界。
4. [x] `benchmarks/`:`bandwidth.py`、`compute.py`、`contention.py`,用 MLX 自定义 Metal kernel 校准 spec。

下一步(未开始):

5. [ ] 可达上界:把 kernel 提交开销、CPU-GPU 同步、反量化、热降频接入上界计算(见 6.4)。
6. [ ] sustained 档:连续负载数分钟,测热降频后的带宽与算力。
7. [ ] 量化模型、多引擎分配、MoE、投机解码。

## 6. 底层系统因素

指模型结构和峰值算力之外,所有影响性能上限的硬件与系统细节。按"是否物理上不可绕开"分三层。

### 6.1 第 1 层:物理上限(必须建模)

任何实现都绕不开,由这一层得到**纯物理上界**。

| 因素 | 说明 | 获取方式 | 现状 |
|---|---|---|---|
| 可达 DRAM 带宽 | 标称约 120 GB/s,实际只能达到一部分;decode 上限与之成正比 | 实测(STREAM 风格 copy / read) | 已测,GPU 读 102.7 GB/s,已用于上界 |
| 读写带宽差异 | 只读、只写、读写混合速度不同;decode 以读为主 | 实测 | 已测(写 94.9、copy 95.0),上界目前读写共用一个带宽 |
| 各引擎单独可达带宽 | GPU / CPU / ANE 各自能用到的带宽 | 分引擎实测 | GPU 已测;CPU 46~54 GB/s 只记在 note,未成字段;ANE 未测 |
| 带宽争抢 | 多引擎同时访存时总量封顶,不叠加 | CPU + GPU 并发实测 | 已测,总量取 GPU 单独值 |
| GPU 可用内存上限 | 24GB 中 GPU 默认约 16~18GB(`iogpu.wired_limit_mb`,0 表示系统默认);**硬约束** | 读系统参数 | 已读,17.76 GiB,已用于容量检查 |
| 各精度峰值算力 | FP32 / FP16 / BF16 / INT8 分引擎分精度 | 估算,再用 GEMM 实测 | GPU FP32/FP16/BF16 已测(FMA 峰值);INT8 与其他引擎未测 |
| 量化元数据开销 | 每组权重额外存 scale / bias,如 group size 64 + FP16 scale/bias,4-bit 实际约 4.5 bit/权重 | 由量化格式推导 | 未实现(目前只有 BF16) |

### 6.2 第 2 层:可达上限(参数化,agent 可调)

取决于实现方案,agent 改方案即可改变。第 1 层 + 第 2 层得到**可达上界**。

| 因素 | 说明 | 获取方式 | 现状 |
|---|---|---|---|
| 反量化开销 | 4-bit 权重需解包为 FP16 再计算;M4 无 tensor core,解包占用通用 ALU,可能使 decode 不再纯带宽受限 | 实测量化 matmul 与普通 matmul 差距 | 未测 |
| kernel 提交开销 × 每 token kernel 数 | 每次提交 command buffer 有固定成本,融合越少 kernel 越多;构成每 token 的固定时间下限 | 实测空 kernel 往返;kernel 数由融合方案决定 | 往返已测(106 µs,含同步),未接入上界 |
| CPU-GPU 同步 | 采样通常需每 token 一次 GPU→CPU 往返 | 实测 | 与上一项合测,未单独拆分 |
| 片上存储容量 | GPU 侧 L2 + SLC、threadgroup memory(32KB);决定中间激活能否不回写 DRAM,即融合上限 | 部分公开,SLC 需实测 | 已测约 12 MiB,已用于 `operator` 上界(6.5) |
| 访问粒度 | cache line 128B、页 16KB;小块零散访问浪费带宽 | 已知 | 未建模 |
| 特殊函数吞吐 | softmax 的 exp 等慢于 FMA,长上下文 attention 中有影响 | 实测 | 未测 |
| 热降频 | 无风扇,长时间负载降频;spec 分 `burst` / `sustained` | 连续跑数分钟记录性能曲线 | 未测;现有实测值都是 burst |

### 6.3 第 3 层:测量噪声(不进入模型,仅在实测时控制)

| 因素 | 控制方式 |
|---|---|
| 后台程序占用 CPU / GPU / 带宽 | 关闭无关程序,测前检查负载 |
| 首次运行编译(shader、pipeline) | 预热后计时 |
| Python / 框架开销 | 用 GPU 时间戳计时,单独列出。目前 benchmarks 用墙钟计时,靠大样本(GB 级或 >50 ms)把启动开销压到约 1% |
| P/E 核调度 | 记录 QoS,CPU 测试固定线程数 |
| 动态调频、电源状态 | 插电、关闭低电量模式、预热后再测 |
| 内存压缩与换页 | 监控 pageout;建模中以容量约束保证不发生 |
| 屏幕显示、模型从 SSD 加载 | 影响小或只影响启动,不计入稳态 |

### 6.4 输出约定

- 同时输出**物理上界**(第 1 层)与**可达上界**(第 1 + 2 层),用参数切换,agent 两者都能看到。
- 第 2 层每一项都是可放松的约束,模型需能回答"放开该项后上限变为多少"。
- 现状:`--bound` 有两个取值。
  - `physical`:第 1 层,片上存储无限、理想融合。
  - `operator`:第 1 层 + 第 2 层中的"有限片上容量 + 每个 GEMM 为独立算子"一项(6.5),`--on-chip-mib` 可放松 M。
    它是可达上界的一个分量,不是完整的可达上界;其余第 2 层项尚未接入。

### 6.5 有限片上存储:red-blue pebble game

物理上界假设片上存储无限(中间激活全部留在片上)。`--bound operator` 把它换成有限容量 M,
实现见 `src/roofline/pebble.py`。

- **假设:每个 GEMM 是一个独立算子。** 激活输入 A 从 DRAM 读(最多 M 字节可以还在片上),
  输出 C 写回 DRAM(同样最多 M 字节可以留在片上);逐元素运算免费融合进 GEMM。
  这是第 2 层约束(取决于融合方案)。A 由上游算子在片上直接产生(跨 GEMM 融合)时下面的定理不适用,
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
  并在机器平衡点更高的设备上给出更紧的界。
- **未覆盖:** attention 的 pebble 下界(Saha & Ye 2024,Θ(N²d²/M),M ≥ d² 时由 FlashAttention 达到)
  只给出了阶,没有显式常数,暂不计入。

## 7. 实测结论(2026-10-03 ~ 10-04)

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
| `mx.matmul` | BF16 3.0~3.1 / FP32 2.6~2.8 TFLOP/s(峰值的约 85% / 76%) | 同上,方阵 n = 1024~8192;现有库水平,属下界,不进上界 |
| 启动 + 同步往返 | 106 µs(中位数 135) | 1 MiB 读 kernel,best of 200 |

在这些参数下,Qwen3-0.6B BF16 的物理上界:decode(batch 1,context 4096)16.2 ms/token,约 62 tok/s,DRAM 受限,
compute slack 约 28;prefill 1024 约 278 ms,compute 受限,DRAM slack 约 22。`operator` 上界与物理上界相同(见 6.5)。

注意:在 decode 场景里,106 µs 的往返如果每 token 发生一次,约占物理上界 16.2 ms 的 0.7%;
若每层一次(28 层)则约 3 ms,接近 20%。这一项是接入可达上界时最先要处理的。

## 8. 已确认

- 第一版范围:GPU 单引擎、dense 模型、decode 与 prefill、batch 可调。
- 第一个模型:Qwen3-0.6B,第一版只做 BF16(权重与 KV cache)。
- MLX 已安装(0.32.3,Python 3.14 可用)。
- 总带宽封顶值取 GPU 单独实测值(第 7 节)。
- 片上容量 M 取实测拐点上沿(6.5)。

## 9. 待确认

- 系统底层因素的范围与上界口径见第 6 节,待确认分层是否合适。
- 上界使用的实测值目前取 best(burst)。是否需要 sustained 档,以及上界该用哪一档。
- 可达上界的组合方式:第 2 层各项是相加(串行开销)还是取 max(可重叠),需逐项确定。
