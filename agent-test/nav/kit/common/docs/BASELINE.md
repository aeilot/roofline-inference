# Baseline — stock mlx-lm on this machine

mlx-lm 0.32.0 `generate_step` with default settings, Qwen3-0.6B BF16, batch 1, greedy, token IDs
only. Medians of 24 process-isolated, thermally gated requests per scenario (random and text prompts).

| Scenario | Prompt | Generated | TTFT | Decode per token | Whole request |
|---|---|---|---|---|---|
| ttft16 | 16 | 8 | 58.5 ms | — | 0.157 s |
| ttft64 | 64 | 8 | 69.5 ms | — | 0.167 s |
| ttft256 | 256 | 8 | 142.8 ms | — | 0.241 s |
| ttft1024 | 1024 | 8 | 403.7 ms | — | 0.509 s |
| ctx512 | 512 | 129 | 224.6 ms | 13.17 ms | 1.923 s |
| ctx4096 | 4096 | 129 | 1900 ms | 17.42 ms | 4.150 s |
| ctx32768 | 32768 | 129 | 53.8 s | 50.26 ms | 60.26 s |
| req256x256 | 256 | 256 | 145.3 ms | 12.96 ms | 3.484 s |

Your engine starts out equal to this (`engine/engine.py` reproduces mlx-lm's behaviour).
