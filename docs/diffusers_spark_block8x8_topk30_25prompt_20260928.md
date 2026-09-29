# Table-4-aligned Spark tail and TopK ablations (25 prompts)

Both Diffusers experiments use Table 4's first 25 prompt/conditioning pairs,
the identical Dense reference videos, 10s 1344×768, seed 42, the 20-point
denoising setting (19 transformer evaluations), effective `torch.compile`,
BF16 global reweight anchor, threshold route execution, power-of-two fanout
16, and four excluded full-denoise GPU warmups. The historical Spark-H3-10%
query result is the matched comparison arm, not a newly generated candidate.
Every video is scored on **all five** VBench dimensions, matching Table 4.

## TopK10, Q/K block8×8 only in the skipped approximate branch

Selected exact 64×64 blocks, the 64-block TopK10 route, and sink/exact rows
are unchanged. Only the skipped tail averages eight Q rows and uses
anchor-weighted eight-K/V summaries. This is a functional ablation, not an
optimized throughput implementation.

| Metric, first 25 prompts | Historical query | block8×8 | Paired change |
| --- | ---: | ---: | ---: |
| Mean denoise time | historical source; not used as a paired runtime | 754.153s | — |
| PSNR ↑ | 23.225 dB | 21.997 dB | −1.228 dB; 5 win / 20 loss |
| SSIM ↑ | 0.7775 | 0.7412 | −0.0363; 3 win / 22 loss |
| LPIPS ↓ | 0.1454 | 0.1762 | +0.0308; 2 win / 23 loss |
| VBench subject consistency ↑ | 89.891% | 89.867% | −0.024 pp; 12 win / 13 loss |
| VBench background consistency ↑ | 94.351% | 94.248% | −0.102 pp; 13 win / 12 loss |
| VBench motion smoothness ↑ | 98.998% | 98.987% | −0.010 pp; 10 win / 15 loss |
| VBench imaging quality ↑ | 70.510% | 70.356% | −0.154 pp; 12 win / 13 loss |
| VBench aesthetic quality ↑ | 61.993% | 62.013% | +0.020 pp; 10 win / 15 loss |

The sizeable PSNR-family regression has little corresponding change in these
VBench averages. No monotonic quality gain should be assumed from refining
the query/key tail granularity. The observed 754s runtime belongs to this
initial unfused block8×8 implementation; it is not an optimized speed bound.

Artifacts:
`/autodl-fs/data/h3_experiments/diffusers_spark_block8x8_25prompt_10s768p_20260928/`
and its matching `/autodl-fs/data/h3_outputs/` directory. The historical
paired metrics are under `topk_reblock_reweight_50prompt_20260920`; the
first 25 case IDs, prompt hashes, and Dense video hashes were checked before
generation and quality evaluation.

## TopK30, original query-granularity skipped branch

The TopK30 arm retains the original query-granularity approximate branch and
changes only the selected-block ratio from 10% to 30%. All 25 videos and
all 125 VBench dimension/video scores completed. Its 25 denoise records each
state `torch_compile=true`; the matching output denoise records independently
record 50 compile-wrapped transformer blocks. Dense video and conditioning
hashes were checked before generation.

| Metric, first 25 prompts | Historical TopK10 | Historical TopK20 | New TopK30 | TopK30 vs TopK10 paired |
| --- | ---: | ---: | ---: | ---: |
| PSNR ↑ | 23.225 dB | 25.202 dB | 26.871 dB | +3.646 dB; 24 win / 1 loss |
| SSIM ↑ | 0.7775 | 0.8315 | 0.8659 | +0.0883; 24 win / 1 loss |
| LPIPS ↓ | 0.1454 | 0.0986 | 0.0775 | −0.0679; 25 win / 0 loss |
| VBench subject consistency ↑ | 89.891% | 90.055% | 90.078% | +0.187 pp; 15 win / 10 loss |
| VBench background consistency ↑ | 94.351% | 94.605% | 94.534% | +0.183 pp; 14 win / 11 loss |
| VBench motion smoothness ↑ | 98.998% | 99.012% | 99.018% | +0.020 pp; 16 win / 9 loss |
| VBench imaging quality ↑ | 70.510% | 71.073% | 71.098% | +0.588 pp; 18 win / 7 loss |
| VBench aesthetic quality ↑ | 61.993% | 61.930% | 61.873% | −0.120 pp; 13 win / 12 loss |

TopK30 also improves PSNR by 1.669 dB over TopK20 (23 win / 2 loss),
SSIM by 0.0344 (23 / 2), and LPIPS by 0.0212 (24 / 1). The large
pixel-metric improvement is not mirrored by similarly large VBench changes.

Observed mean denoise times were 408.307s for the new TopK30 run and
341.615s / 377.727s for the historical TopK10 / TopK20 first-25 records.
These historical arms used a frozen earlier code snapshot. The times are
reported for orientation, **not** as a code-version-controlled speed ablation
or evidence for a precise TopK scaling law. The new block8×8 run is likewise
an unfused functional implementation and cannot establish an optimized
runtime comparison.

TopK30 artifacts:
`/autodl-fs/data/h3_experiments/diffusers_spark_topk30_25prompt_10s768p_20260928/`
and its matching `/autodl-fs/data/h3_outputs/` directory.
