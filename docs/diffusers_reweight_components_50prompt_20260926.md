# Diffusers global-reweight components: 50-prompt extension

> **Retired result (2026-09-27):** This entire numeric ablation used an
> ineffective-compile route helper. Its tables below are retained only as a
> historical record of the protocol error; they are not current benchmark
> evidence. The raw generated artifacts were removed under the
> per-file preview in `docs/uncompiled_delete_preview_20260927.tsv`.

> **Comparability warning (2026-09-27):** This completed run inherited an
> explicit `--no-torch-compile` from its route helper, while the historical
> 2026-09-20 Spark TopK10 and dense reference runs used real per-block
> `torch.compile`. The four arms remain matched **within this run**, but their
> absolute PSNR and runtime must not be compared directly with the compiled
> historical Spark baseline. A four-prompt control restored the current full
> arm from 19.486 to 22.266 dB by enabling compilation; the old Spark mean on
> those same four prompts was 22.776 dB. Pure dense controls also changed
> substantially when compilation was disabled. The full 10/25-prompt
> compile-on validation is separate; do not substitute these 50-prompt
> artifacts for it.

The experiment extends the matched 10-prompt component ablation to the first
50 prompts in the VBench core-five 20% subset. It uses the native
PyTorch/Diffusers pipeline, 10 s / 1344×768 / seed 42 / 20 requested grid
steps (19 transformer evaluations), and GPU0–3 only. It reuses the same
conditioning and dense reference videos. Quality metrics are paired PSNR,
SSIM, and LPIPS against dense, followed by VBench core-five on the dimensions
assigned to each prompt. Raw per-prompt/per-dimension VBench records and video
hashes are retained for audit; the five means use their assigned-prompt
denominators, not an official overall VBench score.

All arms hold Spark-H3 TopK10, fanout16, query-granularity approximate tail,
dense video tail, FP32 global anchor, `comfy_fp32` weighted-summary arithmetic,
and `pre_round` log-mass convention fixed. The sole intervention is the
per-key-block summary:

| Arm | K/V summary | Approximate block log-mass |
| --- | --- | --- |
| `full` | anchor-softmax-weighted K/V | `logsumexp(a·K) − a·weighted_K` |
| `weights_only` | anchor-softmax-weighted K/V | `log(block_length)` |
| `bias_only` | ordinary mean K/V | `logsumexp(a·K) − a·mean_K` |
| `none` | ordinary mean K/V | `log(block_length)` |

`none` is a within-Spark ablation, **not** the official Sol baseline. Two older
50-prompt runs must be kept distinct:

- The canonical 2026-09-20 TopK10 + power-of-two fanout16 + global-reweight
  run (`/autodl-fs/data/h3_experiments/topk_reblock_reweight_50prompt_20260920/`)
  has all 50 prompts, but its saved source uses a BF16 global anchor, BF16
  weighted-summary dot products, and stored/rounded-key log-mass correction.
  It therefore cannot supply this experiment's FP32/`comfy_fp32`/`pre_round`
  `full` arm despite sharing the dataset and broad Spark topology.
- The earlier 2026-09-17 16/16/8 experiment used an arbitrary 16/16/8 reblock
  topology and different numeric settings. Its artifacts have since been moved
  to `/autodl-fs/data/.trash_spark_20260926/old_arbitrary_16_16_8_50prompt/`.

Neither historical run is a controlled comparator for the current component
ablation; neither should be silently substituted for `full` or `none`.

The run reuses 25 `full` artifacts from the completed 25-prompt FP32 precision
experiment and 10 artifacts for each other arm from the completed 10-prompt
component experiment. Sample IDs, prompt SHA-256 hashes, configurations,
conditioning cache, and dense reference entries were checked before reuse.
Thus 55/200 arm–prompt pairs are reused and 145 are newly generated. One
excluded full-denoise warmup per arm/GPU is used. Mixed reused/new wall times
are descriptive and **not** a paired speed comparison. The runner also reports
paired denoising-time differences using only prompt/arm pairs generated in
this run on the same GPU: 25 prompts for comparisons involving `full` and 40
for comparisons among the other three arms.

Runner: `scripts/diffusers_spark_reweight_components_50prompt_20260926.py`.
VBench adapter: `scripts/diffusers_spark_reweight_components_50prompt_vbench_20260926.py`.
The retired protocol and record paths are itemized in the cleanup preview.

## Decoded-video quality and denoising time

All 200 arm–prompt pairs completed denoising and decoding. The following are
the mean of 50 prompt-level paired comparisons against the same dense videos.
The time column uses only the 25 prompts generated **in this run for all four
arms on the same GPU**, rather than mixing historical runtimes.

| Arm | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ | Denoise on shared 25 (s) |
| --- | ---: | ---: | ---: | ---: |
| `full` | 20.2249 | 0.70942 | 0.21386 | 353.132 |
| `weights_only` | 19.6808 | 0.69376 | 0.23408 | 352.511 |
| `bias_only` | 20.0935 | 0.70192 | 0.21724 | 354.085 |
| `none` | 20.0268 | 0.70307 | 0.21713 | 352.458 |

Paired PSNR differences (left minus right, 50 prompts; higher is better):

| Comparison | ΔPSNR (dB) | 95% CI | Wins–losses |
| --- | ---: | ---: | ---: |
| `full` − `none` | +0.1981 | [+0.0719, +0.3244] | 39–11 |
| `full` − `weights_only` | +0.5441 | [+0.3006, +0.7877] | 39–11 |
| `full` − `bias_only` | +0.1314 | [+0.0152, +0.2476] | 37–13 |
| `weights_only` − `none` | −0.3460 | [−0.5810, −0.1110] | 19–31 |
| `bias_only` − `none` | +0.0667 | [−0.0369, +0.1704] | 25–25 |

For `full` versus `none`, paired SSIM changes by +0.00635 (41–9) and LPIPS
changes by −0.00327 (34–16; its 95% interval includes zero). The `full`
denoising-time increment over `none` is +0.674 s on the same 25 newly measured
prompts, or about +0.19% of the `none` runtime. Complete per-prompt metrics,
all SSIM/LPIPS comparisons, and runtime confidence intervals are in
`paired_results.json` in the experiment directory.

These results indicate an interaction: weighted K/V **without** its matched
log-mass bias performs worse than `none`; the bias alone is close to `none`;
the complete pair is the best of the four on mean PSNR and SSIM. This is a
paired numerical observation, not evidence that the bias alone is sufficient
for all prompts or VBench dimensions.

## VBench core-five

The source-assigned VBench evaluation also completed for all four arms. Scores
are percentages; each dimension is averaged only over prompts assigned that
dimension in the core-five source set. The per-arm denominators are 14, 17,
14, 19, and 19 respectively, **not** 50 for each dimension. These are not an
official overall VBench score.

| Arm | Subject ↑ | Background ↑ | Motion ↑ | Imaging ↑ | Aesthetic ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full` | 90.3701 | 94.0553 | 99.0131 | 71.8061 | 68.3146 |
| `weights_only` | 90.5551 | 94.3523 | 99.0135 | 71.3714 | 68.3049 |
| `bias_only` | 90.2757 | 93.9443 | 98.9653 | 71.6077 | 68.0533 |
| `none` | 90.2561 | 94.1195 | 98.9814 | 71.7721 | 68.1230 |

For `full` versus `none`, dimension-level paired wins–losses are subject 9–5,
background 10–7, motion 12–2, imaging 10–9, and aesthetic 10–9; mean score
differences are +0.1140, −0.0642, +0.0317, +0.0340, and +0.1916 percentage
points. Thus the paired PSNR improvement is much clearer than any broad VBench
improvement. Raw per-video/dimension scores and SHA-256 video hashes are in
`vbench/results.json` and `vbench/scores/*.json` under the experiment directory.
