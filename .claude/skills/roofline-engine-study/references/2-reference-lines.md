# Phase 2 — Reference lines from the model graph

## Model YAML

Copy `configs/models/qwen3_0.6b.yaml` and fill from the checkpoint's `config.json`: layers, hidden size,
intermediate size, heads, KV heads, head dim, vocab, tied embeddings, qk-norm, weight and KV dtype.
Architectures outside dense decoder-only transformers (MoE, sliding-window, MLA, hybrid SSM) need
`src/roofline/workload.py` and `graph.py` extended first — say so to the user instead of forcing them
into the dense model.

## Commands

```bash
python -m roofline --model configs/models/<m>.yaml --device configs/hardware/<d>.yaml decode --context 4096
python -m roofline --model ... prefill --prompt-len 4096
python -m roofline --model ... request --prompt-len 4096 --gen 512
python -m roofline --bound hierarchy --model ... prefill --prompt-len 4096      # multi-level I/O bound
python -m roofline --model ... decode --context 32768 --kv window:4096          # B/C KV policies (physical bound only)
```

Output is JSON: per-resource time, the binding resource, slack, memory fit, and provenance of every device
number. Sweep the scenarios fixed in phase 0 and save the table; plots: `scripts/plot_roofline.py`,
`plot_vs_capacity.py`.

## Definitions to state explicitly

- **Level A** = every product term of the *useful* graph's contractions at least once. Outputs: the last
  token's logits and all layers' full K/V. The final layer only needs its last query row through attention,
  O and MLP (`prune_final=True`, default); counting those dead rows inflates prefill W. An engine that skips
  them is doing allowed dead-code removal — do not attribute that to "better kernels".
- **Storage assumption.** KV traffic is compulsory only for an explicit-KV representation; caching hidden
  states and recomputing K/V is mathematically equivalent and was checked (`kvpolicy.best_storage_plan`)
  but not exhaustively for all representations.
- **Initial residency.** DRAM reference credits inputs that might already sit in on-chip memory (17 MiB on
  M4); `initial_resident_bytes=0` is the cold start.

## How to read the result

- Decode is normally bytes-bound: weights once per token plus KV. If the best engine is already at ~90% of
  this line, level-A optimization can give at most ~10%; say so before promising "much faster".
- Prefill is compute-bound beyond small prompts; short prompts are bound by small-M GEMM efficiency and
  fixed per-chunk costs, long prompts by attention.
- The bound is a lower bound only if the rates are true ceilings. Close to it ≠ optimal; far from it may
  also mean the line is loose (our long-prefill gap of ~20–30% remained after a dedicated attention kernel;
  a "softmax ALU missing from the bound" hypothesis was tested and refuted).
- Experimental bounds (fusion/recomputation pebble candidate) stay out of default numbers until proven.
