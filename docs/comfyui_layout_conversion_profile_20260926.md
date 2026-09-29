# ComfyUI Spark global-layout conversion cost

Completed 2026-09-26. Raw full-path denominator:
`/autodl-fs/data/h3_experiments/comfyui_layout_conversion_profile_20260926/denominator.json`.
Final 56-pair isolated timing:
`/autodl-fs/data/h3_experiments/comfyui_layout_conversion_profile_20260926/paired_layout_final.json`.
Scripts:
`scripts/profile_comfy_layout_denominator_20260926.py` and
`scripts/profile_comfy_layout_conversion_20260926.py`.

This measures only two **global** video-last↔video-first adapter costs:

1. The net additional time in the existing chunked QKV producer when target
   video rows are moved to the front. The control calls the *same* producer
   with natural-order tokens, matching token shape, 16,384-token chunks,
   projection, RMSNorm/RoPE and Q/K/V output allocation. Q/K/V produced by
   the reordered call are bitwise equal to the control after permutation.
2. The standalone `output.index_select(1, inverse_permutation)` after Spark
   attention. The output projection is excluded.

Input activations and projection weights for the isolated measurements are
synthetic BF16 tensors of the real H3 dimensions; timing uses CUDA events,
alternating paired order, three preliminary warmups, and 56 measured pairs.
It is an isolated net-cost estimate, **not** a subtraction from two full video
generations. The same-code ComfyUI `all_exact` Spark block single-prompt
20-evaluation run supplies the full-denoise denominators; each duration also
has an excluded five-step warmup. Model load, conditioning, VAE and save are
outside Sampler timing.

| Duration | Packed / video tokens | QKV natural / reordered (ms/call) | Net producer layout Δ (ms/call) | Inverse gather (ms/call) | Combined (ms/call) | Estimated full denoise share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5s 768p | 37,897 / 37,296 | 24.268 / 24.352 | 0.078 | 0.749 | 0.827 | 0.648 s / 96.730 s = 0.670% |
| 10s 768p | 73,573 / 72,576 | 47.381 / 47.528 | 0.160 | 1.444 | 1.604 | 1.258 s / 244.374 s = 0.515% |

The estimate applies the per-call combined cost to 784 sparse calls:
16 sparse evaluations × 49 sparse layers. On an active sparse evaluation,
the conversion is approximately **40.5 ms** (5s) or **78.6 ms** (10s).
The observed sparse-step progress cadence was about 3.97 s and 8.54 s,
respectively, giving roughly **1.02%** and **0.92%** of an active sparse step.
The inverse gather contributes about 90% of this estimated global layout
cost; the QKV producer's whole 24–48 ms must **not** be charged to layout.

These figures exclude reblock-plan internal video ordering and all Spark
route/reweight/attention work. The full-denoise percentages combine isolated
measurements with separately timed same-code runs, so they are estimates, not
an end-to-end ablation of a hypothetical video-last kernel.
