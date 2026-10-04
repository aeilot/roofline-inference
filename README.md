# roofline-inference

Implementation-independent upper bounds on LLM inference performance, derived
from a model spec and a device spec.

Given the model architecture and everything known about the hardware, this
project estimates the best performance any implementation could achieve
(decode tokens/s, prefill TTFT, throughput) and reports which resource
bounds it. Existing kernels give the lower bound ("what we can do today");
the gap between the two is the room for optimization. The end goal is to let
an agent search for inference-engine designs under looser constraints than
existing operators, and score them by `achieved / bound`.

This is not a tool for placing existing kernels on a roofline plot.
Measurements are only used to calibrate device parameters and the lower
bound. See [docs/design-notes.md](docs/design-notes.md) for the full design.

## Approach

- **Inputs:** a model spec (layers, hidden size, heads, KV heads, FFN size,
  bit widths) and a device spec (per-engine compute by precision, total and
  per-engine bandwidth, memory capacity, on-chip storage, overheads). Every
  device value records its source (datasheet / estimate / measured) and
  confidence.
- **Bound:** time is bounded by the slowest resource, with minimum data
  movement assuming ideal fusion (weights read once, KV read once,
  activations kept on chip), subject to a memory-capacity constraint:

  ```
  T >= max( FLOPs / P_peak, Bytes / B_peak, ... )
  weights + KV cache + workspace <= usable memory
  ```

- **Two bounds:** a *physical* bound (unavoidable hardware limits) and an
  *achievable* bound that adds parameterized implementation costs
  (dequantization, kernel launch, sync, fusion scope, thermal throttling).
- **Output:** structured results for an agent: the bound, the bottleneck,
  per-resource time and slack, memory fit, and how the bound changes when a
  constraint is relaxed.

## Current scope

- Device: MacBook Air M4 (Mac16,13), 10-core GPU, 24 GB unified memory.
- Model: Qwen3-0.6B, BF16 weights and KV cache.
- Single engine (GPU), dense models, decode and prefill, variable batch size.

Multi-engine allocation (CPU / GPU / ANE), MoE, and speculative decoding are
planned for later.

## Repository layout

```
.
├── src/roofline/      # Core library: device/model specs, workload, bounds, reports
├── configs/
│   ├── hardware/      # Device specs with source and confidence per value
│   └── models/        # Model architecture specs
├── benchmarks/        # Micro-benchmarks to calibrate device specs (MLX / Metal)
├── scripts/           # One-off and pipeline scripts
├── notebooks/         # Exploratory analysis
├── tests/             # Unit tests
├── docs/              # Design notes and derivations
├── data/              # Input data (git-ignored)
└── results/
    ├── raw/           # Raw measurements (git-ignored)
    └── figures/       # Generated plots (tracked)
```

## Getting started

Requires macOS on Apple Silicon.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # editable install with MLX and pytest
pytest
```

Compute a bound (prints JSON: bound, bottleneck, per-resource time, slack,
memory fit, and the provenance of every device number used):

```bash
python -m roofline --model configs/models/qwen3_0.6b.yaml decode --batch 1 --context 4096
python -m roofline --model configs/models/qwen3_0.6b.yaml prefill --prompt-len 1024
```

## Background

The roofline model bounds attainable performance by

```
P = min(P_peak, I × B_peak)
```

where `P_peak` is peak compute throughput (FLOP/s), `B_peak` is peak memory
bandwidth (bytes/s), and `I` is arithmetic intensity (FLOPs/byte). This
project generalizes it to a max over multiple resource constraints plus a
capacity constraint.

- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance
  Model for Multicore Architectures.* CACM, 2009.
