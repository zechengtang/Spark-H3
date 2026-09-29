# Diffusers global-reweight component ablation

> **Retired result (2026-09-27):** This numeric run inherited an explicit
> no-compile route helper. The tables below are historical, not comparable to
> the compiled four-arm 50-prompt main benchmark. Raw generated artifacts were
> removed; see `docs/uncompiled_delete_preview_20260927.tsv`.

Completed 2026-09-26; retired protocol and record paths are itemized in the
cleanup preview.

The experiment fixes Spark-H3 TopK10, fanout16, query-granularity tail,
FP32 global anchor, `comfy_fp32` weighted-summary arithmetic, and
`pre_round` log-mass convention. Only two per-key-block components vary:

| Arm | K/V summary | Approximate block log-mass |
| --- | --- | --- |
| `full` | anchor-softmax-weighted K/V | `logsumexp(a·K) − a·weighted_K` |
| `weights_only` | anchor-softmax-weighted K/V | `log(block_length)` |
| `bias_only` | ordinary mean K/V | `logsumexp(a·K) − a·mean_K` |
| `none` | ordinary mean K/V | `log(block_length)` |

Dot products in the table include the usual `1/sqrt(128)` scale. The
`bias_only` shift uses its **active** mean key, so the bias arm corrects block
mass without silently retaining a weighted key. The `none` arm is a matched
within-Spark summary baseline, not the official Sol kernel. With the default
`native_mean` route all four arms retain identical TopK routing/reblock/exact
policy. The full arm remains the production default.

Used the first 10 prompts of the same 25-prompt 10s/768p/seed42/20-grid-step
dataset. Reuse its BF16 conditioning, dense videos, and already generated
`spark_comfy_fp32` full-arm videos. The other three arms were generated on GPU0–5,
excluding one full warmup per arm/GPU from denoise timing. Decoding and scoring
used GPU0–3 after the six-GPU decode load showed sustained I/O waits. PSNR,
SSIM and LPIPS were scored against the same dense reference; VBench is omitted for
this numeric/algorithm component ablation. Runtime of the reused full arm is
from a separate run and must **not** be treated as paired timing against the
new arms. The three newly generated arms are internally paired by prompt/GPU.

Runner: `scripts/diffusers_spark_reweight_components_10prompt_20260926.py`.
It refuses to prepare before the source 25-prompt quality run completes.

| Arm | Denoise mean (s) | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| `full` | 353.108* | 20.1058 | 0.71288 | 0.20578 |
| `weights_only` | 352.874 | 19.9817 | 0.70639 | 0.21543 |
| `bias_only` | 352.689 | 20.0553 | 0.70560 | 0.20937 |
| `none` | 352.578 | 19.9152 | 0.70528 | 0.21078 |

*The `full` runtime is reused from the earlier 25-prompt experiment, not a
paired timing run against the other three arms. All quality values are the
mean of the same 10 matched prompts. `full` leads on all three mean quality
metrics; the 10-prompt sample is too small to claim a stable population-level
advantage. `bias_only` has a larger PSNR gain over `none` than `weights_only`
does, while `weights_only` worsens mean LPIPS versus `none`. These are
component interactions, not an additive decomposition of quality.
