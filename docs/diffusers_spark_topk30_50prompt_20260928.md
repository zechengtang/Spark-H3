# Diffusers Spark-H3 TopK30, 50 prompts (2026-09-28)

> **Sampling note (2026-10-04):** The reused first 25 records are cases 1--25
> of the `20pct` manifest, not the canonical `10pct` subset. Their standalone
> 25-prompt table is not valid benchmark evidence. This document reports the
> completed 50-prompt result, so its full-50 aggregates are the required
> confirmation and are not invalidated by the composition of the intermediate
> first half.

This extends the matched 25-prompt TopK30 run to the full 50-prompt
Table-4 dataset at 10s/768p, seed 42, 20-step grid/19 transformer
evaluations, torch.compile enabled, query-granularity approximate tail,
fanout 16, full global reweight, and the same Dense reference. The first
25 completed records are reused after verifying their prompt/compile
provenance and linking their latents/videos into this standalone result;
cases 26–50 were newly generated on GPU2–5 with one excluded full-denoise
warmup per worker. The 50-prompt manifest records 50 cases. All five VBench
dimensions have 50 per-video scores each; first-half video scores were reused
only after SHA-256 video-match validation. For comparison with historical
Table 4, the table below aggregates only the prompt subset assigned to each
dimension (14/17/14/19/19 videos). The all-50-per-dimension means are a
different diagnostic statistic and must not be mixed into Table 4.

| Method | Mean DiT denoise (s) | PSNR (dB) | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Historical Spark-H3-10pct | 341.7 | 23.30 | 0.80 | 0.14 |
| Historical Spark-H3-20pct | 378.2 | 25.38 | 0.85 | 0.09 |
| New Spark-H3-30pct | 408.59 | 27.07 | 0.8824 | 0.0726 |

| Method | Subject consistency | Background consistency | Motion smoothness | Imaging quality | Aesthetic quality |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical Spark-H3-10pct | 90.77 | 94.24 | 99.01 | 71.81 | 68.22 |
| Historical Spark-H3-20pct | 90.92 | 94.25 | 99.01 | 72.08 | 67.73 |
| New Spark-H3-30pct | 90.90 | 93.99 | 99.02 | 72.01 | 67.72 |

The larger TopK budget improves Dense-paired reconstruction metrics but does
not improve every VBench dimension. The initially reported all-50 means for
TopK30 (89.72/94.62/99.01/71.14/64.08) were **not comparable** with the
historical Table-4 rows, which use dimension-assigned subsets. On the same
all-50 diagnostic protocol, historical Dense, Spark10 and Spark20 aesthetic
means are 63.98, 64.33 and 64.06 respectively; TopK30 is 64.08. Thus the
apparent aesthetic drop was a scoring-population mismatch, not evidence that
TopK30 alone degraded sharply. Historical rows above are the existing blog
Table-4 results, not fresh reruns. Their timing rows come from a frozen
earlier code snapshot, so they do not form a code-version-controlled scaling
experiment. Do not infer significance from their rounded two-decimal
presentation alone.
The new run's raw records and unrounded
summaries are under
`/autodl-fs/data/h3_experiments/diffusers_spark_topk30_50prompt_10s768p_20260928/`,
with decoded videos under the matching `/autodl-fs/data/h3_outputs/` root.

## Attention speedup check

The 30% row's attention speedup was measured separately on September 28 with
GPU0–5: three paired Dense/Spark-H3-30pct prompts (indices 2, 13, 31), 10s/768p,
seed 42, 20 scheduler grid points (19 evaluations), and `torch.compile=True`.
Each worker loaded weights once, excluded one complete denoise as warmup, then
timed all 950 `block.attn.forward` calls in a complete denoise using CUDA events.
The Spark settings match the 50-prompt TopK30 run. This is an inclusive
attention-module measurement, not the complete DiT latency or an estimate
derived from it.

| Prompt | Dense attention (s) | Spark-30 attention (s) | Speedup |
| --- | ---: | ---: | ---: |
| 2 | 493.70 | 320.23 | 1.542× |
| 13 | 490.44 | 316.16 | 1.551× |
| 31 | 491.99 | 312.94 | 1.572× |

The ratio of the two three-prompt means is **1.555×** (rounded to 1.55× in
the blog). Raw per-GPU records, protocol, and status are in
`/autodl-fs/data/h3_experiments/diffusers_spark_topk30_attention_20260928/`;
the reproducible runner is
[`scripts/diffusers_spark_topk30_attention_20260928.py`](../scripts/diffusers_spark_topk30_attention_20260928.py).
The historical 10%/20% attention speedups were measured on an older code
snapshot, so the three table entries are not a strictly same-snapshot Top-K
scaling ablation.

### Matched attention-core retest

The three paired prompts were rerun on GPU0–5 with the same Spark-30 settings,
20 scheduler grid points (19 evaluations), `torch.compile=True`, and one
excluded full-denoise warmup. The timer now starts after QKV construction:
Dense measures its attention dispatch, while Spark measures the full sparse
attention operator including routing, reblock, reweight, exact/sink work, and
the four dense warmup evaluations. Output projections are outside both timers.
All 950 Dense calls and Spark's 215 dense plus 735 sparse calls were checked.

| Prompt | Dense core (s) | Spark-30 core (s) | Speedup |
| --- | ---: | ---: | ---: |
| 2 | 416.31 | 237.78 | 1.751× |
| 13 | 413.49 | 234.67 | 1.762× |
| 31 | 414.34 | 232.58 | 1.782× |

The ratio of the three-prompt means is **1.765×**. The raw records and
protocol are in
`/autodl-fs/data/h3_experiments/diffusers_spark_topk30_attention_core_v2_20260928/`;
the runner is
[`scripts/diffusers_spark_topk30_attention_core_20260928.py`](../scripts/diffusers_spark_topk30_attention_core_20260928.py).
This is an operator-inclusive attention-core result, not the isolated kernel
latency reported by VC-Attention. It also remains a three-prompt/current-code
measurement, whereas the blog's 10%/20% attention-core figures came from an
older code snapshot, so it should not be presented as a strictly controlled
four-variant scaling ablation without remeasuring those rows.
