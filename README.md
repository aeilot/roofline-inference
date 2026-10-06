# roofline-inference

Implementation-independent upper bounds on LLM inference performance, derived
from a model spec and a device spec, plus the gap between today's engines and
those bounds.

Given the model architecture and calibrated hardware numbers, the project
computes the best time any implementation could reach (decode per token,
time to first token, a whole request), names the binding resource, and
splits an engine's measured time into the pieces it loses. The end goal is a
navigator for an agent that customizes an inference engine: where the gap
is, and how much relaxing each constraint would gain.

This is not a tool for placing existing kernels on a roofline plot.
Measurements calibrate the device spec and decompose the engines' gap. The
results report (Chinese) is [docs/report.md](docs/report.md); the full
design and derivations are in [docs/design-notes.md](docs/design-notes.md).

## Results so far (Qwen3-0.6B BF16, MacBook Air M4, single stream)

| | Bound | mlx-lm 0.32 | llama.cpp (Metal) |
|---|---|---|---|
| Decode, context 512 / 4k / 32k | 12.2 / 16.2 / 48.2 ms per token | 95 / 91 / 90% of bound | 87 / 83 / – % |
| Time to first token, prompt 512 / 4k / 32k | 0.13 / 1.50 / 41.3 s | 67 / 78 / 74% | 70 / 67 / 9% |

- Decode is bound by DRAM bandwidth (every token streams 1.19 GB of
  weights); mlx-lm is close to the bound, so faster decode needs algorithmic
  changes (quantization, speculative decoding), not better kernels.
- Prefill is bound by compute; short prompts lose to small-GEMM efficiency
  and per-chunk overheads, long prompts to attention.
- The fanless machine throttles: the same 32k prefill takes 55.8 s cold and
  70.7 s after ~40 minutes of load.

Burst measurements; see [docs/report.md](docs/report.md) and design notes 6.9 for the decomposition and caveats.

## Approach

- **Inputs:** a model spec (layers, hidden size, heads, KV heads, FFN size,
  dtypes) and a device spec (compute and special-function throughput,
  bandwidth, memory hierarchy, kernel overheads). Every device value records
  its source (datasheet / estimate / measured) and confidence.
- **Level-A bound:** any implementation that computes every product term of
  the model's contractions at least once; summation order, fusion,
  recomputation and the softmax formulation are free. Time is bounded by the
  slowest resource:

  ```
  T >= max( FLOPs / P_peak,  Q_b / B_b  for every memory boundary b,  exps / exp throughput )
  ```

- **Traffic Q_b:** from an operator-level computation graph of the step.
  Compulsory traffic (inputs read once, outputs written once), red-blue
  pebble-game bounds per memory level and per core (Smith et al. 2019 for
  GEMM, Saha & Ye 2024 for attention with explicit constants), and a
  fusion- and recomputation-aware bound derived here (design notes 6.12).
  Two variants are reported: fused (any implementation) and operator
  boundary (every GEMM and attention its own operator).
- **Per-plan cost and gap:** a calibrated cost model T_A(plan) (kernel
  fixed cost, streaming bandwidth, CPU encode, commit, sync, P(shape)) and a
  telescoping ablation of mlx-lm that splits measured time into matmul,
  attention, small kernels and engine overhead.
- **Output:** JSON with the bound, the bottleneck, per-resource time and
  slack, and the provenance of every number used; figures in
  `results/figures/`.

## Current scope

- Device: MacBook Air M4 (Mac16,13), 10-core GPU, 24 GB unified memory.
- Model: Qwen3-0.6B, BF16 weights and KV cache.
- Single stream (batch 1): decode at context 512 to 32k, prefill of 16 to
  32k tokens. GPU only.

Quantization, speculative decoding, batching, multi-engine allocation
(CPU / GPU / ANE) and MoE are later stages.

## Repository layout

```
.
├── src/roofline/      # Specs, workload, computation graph, bounds, I/O bounds, gap model
├── configs/
│   ├── hardware/      # Device spec (+ kernel-level calibration .kernels.json)
│   └── models/        # Model architecture specs
├── benchmarks/        # MLX / Metal micro-benchmarks, engine baselines and ablation
├── scripts/           # Fits, calibration export, GGUF converter, figures
├── tests/             # Unit tests (incl. an exact red-blue pebbling solver)
├── docs/              # Design notes and derivations
├── data/              # Input data, converted models (git-ignored)
└── results/
    ├── raw/           # Raw measurements (git-ignored)
    └── figures/       # Generated plots and their data (tracked)
```

## Getting started

Requires macOS on Apple Silicon.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # editable install with MLX, pytest, matplotlib
pytest
```

Bounds (JSON: bound, bottleneck, per-resource time, slack, memory fit,
provenance of every device number):

```bash
python -m roofline --model configs/models/qwen3_0.6b.yaml decode --context 4096
python -m roofline --model configs/models/qwen3_0.6b.yaml prefill --prompt-len 1024
python -m roofline --model configs/models/qwen3_0.6b.yaml request --prompt-len 4096 --gen 512
python -m roofline --bound hierarchy --model configs/models/qwen3_0.6b.yaml prefill --prompt-len 4096
```

`--bound operator` uses the measured on-chip capacity and an operator
boundary at every GEMM (`--on-chip-mib` relaxes it); `--bound hierarchy`
bounds the traffic across every level of the memory hierarchy (per-core
SRAM, L2, DRAM; 10 cores), fused and operator boundary.

Calibration, baselines and figures:

```bash
python benchmarks/bandwidth.py && python benchmarks/compute.py      # DRAM bandwidth, compute peak
python benchmarks/hierarchy.py                                      # on-chip levels
python benchmarks/concurrency.py                                    # latency, bandwidth/compute vs threads
python benchmarks/kernel_chain.py && python scripts/fit_kernel_chain.py
python benchmarks/compute_shapes.py && python scripts/calibrate.py # P(shape) -> configs/hardware/*.kernels.json
python benchmarks/baseline_mlx.py                                   # mlx-lm TTFT and decode
python benchmarks/ablation_mlx.py && python scripts/gap_report.py   # gap decomposition
python scripts/plot_roofline.py                                     # roofline figures
python scripts/plot_vs_capacity.py && python scripts/plot_capacity_parallelism.py
```

llama.cpp needs a GGUF; `scripts/convert_qwen3_gguf.py` converts the HF
safetensors to BF16 GGUF without PyTorch, then `llama-bench` runs it.

## Background

The roofline model bounds attainable performance by

```
P = min(P_peak, I × B_peak)
```

where `P_peak` is peak compute throughput (FLOP/s), `B_peak` is peak memory
bandwidth (bytes/s), and `I` is arithmetic intensity (FLOPs/byte). Here `I`
is not an input: it is derived from the computation graph and the
fast-memory capacity, `I_max(S) = FLOPs / Q_min(S)`, with `Q_min` from the
red-blue pebble game, and the bound is a max over several resources.

- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance
  Model for Multicore Architectures.* CACM, 2009.
- Hong, Kung. *I/O Complexity: The Red-Blue Pebble Game.* STOC, 1981.
- Smith, Lowery, Langou, van de Geijn. *A Tight I/O Lower Bound for Matrix
  Multiplication.* TOMS, 2019.
- Saha, Ye. *The I/O Complexity of Attention, or How Optimal is Flash
  Attention?* ICML, 2024.
