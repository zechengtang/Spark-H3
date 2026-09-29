# Diffusers Spark global-reweight precision: 25 prompts

> **Retired result (2026-09-27):** This numeric run inherited an explicit
> no-compile route helper. The tables below are historical and should not be
> treated as current precision evidence. Raw generated artifacts were removed;
> see `docs/uncompiled_delete_preview_20260927.tsv`.

Completed 2026-09-26. Runner:
`scripts/diffusers_spark_reweight_precision_25prompt_20260926.py`.
Retired protocol and record paths are itemized in the cleanup preview.

All three arms use Diffusers Spark TopK10, fanout 16, query-granularity
approximate tail, seed 42, 240 frames at 1344×768, and the 20-step scheduler
(19 transformer evaluations). Each prompt's three arms run on the same GPU in
rotating order, after one excluded full-denoise warmup per arm/GPU. Timing is
synchronized denoising only. The existing 25 matching dense videos and text
conditioning tensors are reused; no VBench scoring was run for this numeric
precision ablation.

| Arm | Anchor | Summary math | Log-mass key | Denoise mean (s) | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `spark_bf16` | BF16 | Tensor Core | BF16 stored | 353.651 | 20.8206 | 0.73142 | 0.19588 |
| `spark_anchor_fp32` | FP32 | Tensor Core | BF16 stored | 353.706 | 20.8868 | 0.73243 | 0.19495 |
| `spark_comfy_fp32` | FP32 | FP32 lane-wise FMA/reduction | FP32 pre-round | 353.438 | 20.8648 | 0.73157 | 0.19466 |

All quality metrics compare decoded videos with the reused dense reference.
The FP32-anchor arm changes denoise time by +0.055 s relative to BF16; the
Comfy-style FP32 arm changes it by −0.268 s relative to FP32-anchor. Both are
tiny relative to the approximately 354 s run and should not be called reliable
speed improvements.

Paired prompt-level quality differences:

| Comparison (candidate − reference) | Mean ΔPSNR | PSNR wins/losses | Mean ΔSSIM | Mean ΔLPIPS |
| --- | ---: | ---: | ---: | ---: |
| FP32 anchor − BF16 | +0.0661 dB | 17/8 | +0.00101 | −0.00093 |
| Comfy-style FP32 − FP32 anchor | −0.0219 dB | 14/11 | −0.00086 | −0.00030 |
| Comfy-style FP32 − BF16 | +0.0442 dB | 14/11 | +0.00016 | −0.00123 |

The approximate paired-prompt 95% t interval for FP32-anchor ΔPSNR is
[-0.129, +0.262] dB. Thus this 25-prompt set does **not** establish a stable
population-level PSNR benefit or show a clear winner between the two FP32
summary paths. The earlier 10-prompt FP32-anchor gain was +0.239 dB, but that
sample was **not nested** inside this canonical VBench core-five 10% 25-prompt
subset: only six prompt IDs overlap. On those six shared prompts, each arm's
PSNR is exactly reproduced across the two experiments, and their FP32-anchor
gain averages +0.1678 dB. The difference between the 10- and 25-prompt means
therefore reflects prompt composition, not a detected code/seed regression.

`comfy_fp32` reproduces the ComfyUI reweight **precision stages**, not its
bitwise CUDA reduction order or INT8 attention consumer. It does not change
route or approximate-tail granularity. See
`docs/diffusers_reweight_numeric_modes_20260926.md` for implementation scope.
