# Spark-H3 speed verification

This document records the October 1–2, 2026 two-GPU verification of the
current Spark-H3 Top-K 10% implementation on Diffusers and ComfyUI. The purpose
was to check whether the previously reported 10-second speedups—about 1.7x for
the Diffusers blog benchmark and about 2.1x for the later ComfyUI benchmark—are
both reproducible under their native execution stacks.

## Result

Each backend, duration, and method received one complete excluded generation
before two measured generations. The table reports the two sampler/denoiser
times and the ratio of the Dense median to the Spark median.

| Backend | Duration | Dense runs (s) | Spark-H3 10% runs (s) | Dense median (s) | Spark median (s) | Speedup |
|---|---:|---|---|---:|---:|---:|
| Diffusers | 5s | 199.06, 199.27 | 142.40, 140.57 | 199.17 | 141.48 | **1.408x** |
| Diffusers | 10s | 585.70, 584.68 | 340.10, 340.13 | 585.19 | 340.12 | **1.721x** |
| Diffusers | 14.4s | 1073.42, 1072.99 | 573.31, 572.58 | 1073.21 | 572.94 | **1.873x** |
| ComfyUI | 5s | 167.78, 167.69 | 98.04, 98.31 | 167.74 | 98.18 | **1.709x** |
| ComfyUI | 10s | 538.76, 537.68 | 248.38, 248.49 | 538.22 | 248.44 | **2.166x** |
| ComfyUI | 14.4s | 1014.65, 1013.85 | 432.82, 432.61 | 1014.25 | 432.71 | **2.344x** |

The two measurements in every cell are close: the largest within-cell range is
1.82 seconds, or about 1.3% of the corresponding median. The results are
therefore not driven by a one-off compilation event.

## Warmup and evaluation accounting

Two different kinds of warmup were kept separate:

1. **Spark schedule warmup:** both backends use a 20% Dense prefix and keep
   transformer layer 0 Dense for every later sparse evaluation.
2. **Runtime warmup:** one full generation for every backend, method, and video
   length was excluded before measurement. This covers model initialization,
   shape-specific compilation, and the first sparse evaluation.

Diffusers requests 20 scheduler points but performs 19 transformer evaluations.
Its Spark schedule is therefore 4 Dense + 15 sparse evaluations. ComfyUI's
20-step sampler performs 20 model evaluations, giving 4 Dense + 16 sparse
evaluations. The results deliberately preserve these real native semantics
instead of relabeling either run as the other backend's NFE convention.

The frame accounting was aligned at the model input:

| Nominal duration | Diffusers request | Diffusers aligned/model frames | ComfyUI model frames |
|---|---:|---:|---:|
| 5s | 120 | 124 | 124 |
| 10s | 240 | 243 | 243 |
| 14.4s | 345 | 345 | 345 |

Both backends used the same cached 179-token conditioning, 1344x768 resolution,
and measured seeds 42 and 43.

## Interpretation

The apparently conflicting historical 10-second values are both reproducible:

- The current Diffusers result is 1.721x, within about 1.0% of the blog's
  1.703x 35-prompt mean.
- The current ComfyUI result is 2.166x, within about 0.9% of the earlier
  single-case ComfyUI result of 2.148x.

Consequently, `1.7x` and `2.1x` are not two estimates of one identical
experiment. They describe different native stacks. The discrepancy is real and
does not come from accidentally including first-run compilation or from
recomputing the same timing with a different denominator.

Cross-backend raw times and speedups are not strictly apples-to-apples. The
Diffusers run uses the full BF16 checkpoint with `torch.compile`; the ComfyUI
run uses `minimax_h3_fl2va_pruned_int8_convrot.safetensors` and its native H3
producer. The current backend defaults also differ in route execution details,
global-anchor precision, midpoint construction, and total evaluation count.
The table should therefore be cited as two internally controlled native-stack
measurements, not as evidence that changing only the frontend causes the entire
speedup difference.

The increase with video length is consistent in both stacks. With a fixed
four-evaluation Dense prefix and one always-Dense transformer layer, the
quadratic attention work saved by Spark occupies a larger fraction of the total
runtime as the video token count grows.

## Reproducibility record

- Hardware: two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (SM120)
- Experiment directory:
  `/autodl-fs/data/h3_experiments/current_warmup_diffusers_comfyui_2gpu_20261001_v2`
- Aggregate data: `results.json`
- Human-readable raw report: `REPORT.md`
- Full protocol and source hashes: `protocol.json`
- Per-run records: `diffusers/records/` and `comfyui/records/`
- Runner snapshot: `runner_source.py`
- Repository runner:
  `scripts/benchmark_current_warmup_diffusers_comfyui_2gpu_20261001.py`

The protocol records the Git revision, dirty-diff hash, hashes of the relevant
Diffusers, Spark, ComfyUI, and benchmark sources, the exact conditioning hash,
model paths, GPU assignment, and all Spark parameters.
