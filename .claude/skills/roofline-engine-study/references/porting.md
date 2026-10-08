# Porting to another device, model or engine

The repository's code is general in structure but specific in a few places. This list is what to change.

## New device

| Item | Apple Silicon (current) | Linux + NVIDIA | Other |
|---|---|---|---|
| Calibration kernels | MLX `mx.fast.metal_kernel` benchmarks in `benchmarks/` | rewrite with CUDA/Triton or PyTorch; use cuBLAS/cuDNN for GEMM and MMA peaks | vendor BLAS + a bandwidth stream kernel |
| Device YAML | `configs/hardware/m4_mba_24g.yaml` | new file, same schema; add per-level capacities (SMEM/L1, L2, HBM) | same |
| Thermal gate | `NSProcessInfo.thermalState` via a tiny Swift binary (`agent-test/v2/therm.swift`), plus a ramped probe | `nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks_throttle_reasons.active`; gate on no active throttle reason and stable SM clock | sensors / vendor tool |
| Power gate | `pmset -g batt` must say AC; refuse on battery | usually N/A (servers); laptops: `/sys/class/power_supply/*/online` | |
| Sleep detection | `sysctl kern.sleeptime kern.waketime` before/after each measurement | `CLOCK_BOOTTIME − CLOCK_MONOTONIC` grows across suspend | |
| Keep awake / detach | `caffeinate -dimsu`, launchd one-shot job; lid must stay open on battery-less AC without external display | `systemd-inhibit`, systemd unit or tmux | |
| Stream sync | `mx.synchronize(stream)` per engine | `torch.cuda.synchronize()` (all streams on the device) | |
| Clock control | not available | consider `nvidia-smi -lgc` (needs admin; ask the user) and persistence mode; record either way | |

`scripts/platform_state.py` in this skill implements the thermal/power/sleep snapshot for macOS and
Linux/NVIDIA; wire it into `worker.py` in place of the macOS-only calls.

Other device-specific facts to re-measure, not assume: whether CPU+GPU share bandwidth and power, idle
downclock latency, sustained throttling, the MMA peak per dtype.

## New model

- Model YAML (phase 2). Non-dense architectures need workload/graph extensions first.
- **Engine kernel assumptions** in `agent-test/v2`: `head_dim == 128`, GQA factor 2 (two query heads per KV
  head in the paired attention kernel), hidden size multiple of 256, tied embeddings, per-head q/k RMSNorm,
  non-traditional RoPE with one base, SwiGLU. `Engine.__init__` raises on violations — extend the kernels
  (template the head dim and GQA factor) rather than silently falling back.
- Re-tune shape tables (`tune_skinny.py`) and GEMV configs for the new shapes and device.
- FP32 reference must fit in memory (weights ×2 plus FP32 KV at the longest context); otherwise reduce the
  gate's context lengths and say so.
- New corpus snapshot if the language/domain matters; long contexts need a corpus longer than the context
  (our 44k-token corpus made 32k text prompts near-duplicates).
- Re-derive the divergence noise threshold at the tested context lengths.

## New engine

Write one adapter class (see `agent-test/v3/adapters.py`, `vllm_adapter.py`):

```python
class MyEngine:
    identity = {...}                      # engine, versions, settings that differ from defaults
    def __init__(self, model_path): ...
    def generate(self, prompt_ids, max_tokens):   # yield ints one at a time; exactly max_tokens; ignore EOS
        ...
    def synchronize(self): ...           # wait for ALL device work this engine issued
    def teacher_forced(self, prompt_ids, cont):   # logits per position via the real path (optional; else token screen)
        ...
```

Rules: same weights snapshot as the others (hash it), batch 1 unless batching is the subject, prefix caching
and speculative decoding off unless they are the subject, per-token timing must not see multi-token packets
(raise if the engine emits several tokens per step), no detokenization in the timed path. Record the
engine's own memory reservation (vLLM reserved ~6 GB of KV pool on the M4).

## Before the first formal run on new hardware

Run the smoke profile and compare a few numbers against isolated, hand-run measurements. Every surprise
in this project (heat, clamshell sleep, a too-tight threshold, a version-dependent slowdown) showed up there
first.
