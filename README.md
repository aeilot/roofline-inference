# roofline-inference

Calibrated resource references for LLM inference, derived from a model and
device spec, alongside engine measurements and conditional I/O bounds.

The time formula is a lower bound only when the hardware rates are genuine
upper limits and the graph/storage assumptions hold. Our microbenchmark best
values are calibrations, not certified ceilings; closeness to them does not
prove that an implementation is optimal. The goal is to help an engine
optimization agent form and test hypotheses.

The revised Chinese report is [docs/report.md](docs/report.md); derivations
and historical measurements are in [docs/design-notes.md](docs/design-notes.md).

## Results so far (Qwen3-0.6B BF16, MacBook Air M4, single stream)

| Phase | Corrected calibrated reference (not measured latency) |
|---|---|
| Decode, context 512 / 4k / 32k | 12.0 / 16.0 / 48.0 ms per token |
| TTFT, prompt 512 / 4k / 32k | 0.127 / 1.454 / 39.870 s |

Default prefill counts only useful final-layer rows; complete KV outputs are
retained. DRAM traffic credits the modeled initial on-chip input capacity.
Historical full-graph/cold values and hypothetical rate sensitivities are in
[review_corrections.json](results/figures/review_corrections.json).

The v3 simple experiment observes TTFT ratios of 1.682 / 1.591 / 1.250 at
16 / 64 / 256 tokens. Its 32k TTFT and decode results remain undecided.
[Corrected analysis](agent-test/v3/results-simple/report-corrected.md) is
explicitly post-collection and preserves the original registered analysis.
Small-sample intervals, consecutive sessions and thermal effects limit inference.
KV quality experiments have no measured compressed/sparse decode speed yet.

## Approach

- **Inputs:** a model spec (layers, hidden size, heads, KV heads, FFN size,
  dtypes) and a device spec (compute and special-function throughput,
  bandwidth, memory hierarchy, kernel overheads). Every device value records
  its source (datasheet / estimate / measured) and confidence.
- **Level-A bound:** any implementation that computes every product term of
  the useful graph's contractions at least once (unused final rows are removed); summation order, fusion,
  recomputation and the softmax formulation are free. With valid hardware ceilings,
  resource necessities bound time; the current output is a calibrated reference:

  ```
  T >= max( FLOPs / P_peak,  Q_b / B_b  for every memory boundary b )
  ```

- **Traffic Q_b:** from an operator-level computation graph of the step.
  Compulsory traffic (inputs read once, outputs written once), red-blue
  pebble-game bounds per memory level and per core (Smith et al. 2019 for
  GEMM, Saha & Ye 2024 for attention with explicit constants), and an
  experimental fusion/recomputation candidate (design notes 6.12), excluded
  from default bounds pending proof review.
  Two variants are reported under explicit-KV storage assumptions: fused and operator
  boundary (every GEMM and attention its own operator).
- **Per-plan cost and gap:** a calibrated cost model T_A(plan) (kernel
  fixed cost, streaming bandwidth, CPU encode, commit, sync, P(shape)) and a
  telescoping ablation of mlx-lm reporting ordered marginal effects of matmul,
  attention, small kernels and engine overhead, not independent causal losses.
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

References (JSON: interpretation, conditional bottleneck, resource time, slack, memory fit,
provenance of every device number):

```bash
python -m roofline --model configs/models/qwen3_0.6b.yaml decode --context 4096
python -m roofline --model configs/models/qwen3_0.6b.yaml prefill --prompt-len 1024
python -m roofline --model configs/models/qwen3_0.6b.yaml request --prompt-len 4096 --gen 512
python -m roofline --bound hierarchy --model configs/models/qwen3_0.6b.yaml prefill --prompt-len 4096
```

`decode --kv window:4096` (or `budget:K`, `retrieval:K`, `--kv-bits 4`)
relaxes the KV cache to a level B/C policy whose shape follows the streaming
attention lower bounds (design notes 6.13); the default is level A.

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
python benchmarks/key_clusters.py                                   # key clusterability (SubGen premise)
python benchmarks/kv_quality.py && python scripts/plot_kv_quality.py # KV policy quality vs time bound
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
fast-memory capacity, `I_cap(S) = FLOPs / Q_lower(S)`, where `Q_lower` is a
lower bound from the red-blue pebble game, not an attained minimum, and the bound is a max over several resources.

- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance
  Model for Multicore Architectures.* CACM, 2009.
- Hong, Kung. *I/O Complexity: The Red-Blue Pebble Game.* STOC, 1981.
- Smith, Lowery, Langou, van de Geijn. *A Tight I/O Lower Bound for Matrix
  Multiplication.* TOMS, 2019.
- Saha, Ye. *The I/O Complexity of Attention, or How Optimal is Flash
  Attention?* ICML, 2024.
