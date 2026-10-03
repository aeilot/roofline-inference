# roofline-inference

Roofline-model analysis of ML inference workloads: characterizing where model
operators and end-to-end inference (prefill, decode, batching) sit relative to
hardware compute and memory-bandwidth ceilings.

## Goals

- Compute analytical FLOPs and memory traffic for model operators (GEMM,
  attention, normalization, etc.) and derive arithmetic intensity.
- Measure achieved throughput on real hardware and place it on the roofline.
- Compare phases (prefill vs. decode), batch sizes, sequence lengths,
  precisions, and hardware targets.

## Repository layout

```
.
├── src/roofline/      # Core library: hardware models, op cost models, plotting
├── configs/
│   ├── hardware/      # Peak FLOPs / bandwidth specs per device
│   └── models/        # Model architecture configs
├── benchmarks/        # Profiling / measurement harnesses
├── scripts/           # One-off and pipeline scripts (run sweeps, make figures)
├── notebooks/         # Exploratory analysis
├── tests/             # Unit tests
├── docs/              # Notes, derivations, write-ups
├── data/              # Input data (git-ignored)
└── results/
    ├── raw/           # Raw measurements and profiler output (git-ignored)
    └── figures/       # Generated plots (tracked)
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
```

_Dependencies and usage instructions will be added as the project develops._

## Background

The roofline model bounds attainable performance by

```
P = min(P_peak, I × B_peak)
```

where `P_peak` is peak compute throughput (FLOP/s), `B_peak` is peak memory
bandwidth (bytes/s), and `I` is arithmetic intensity (FLOPs/byte). Kernels with
`I` below the ridge point `P_peak / B_peak` are memory-bound; those above are
compute-bound.

- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance
  Model for Multicore Architectures.* CACM, 2009.
