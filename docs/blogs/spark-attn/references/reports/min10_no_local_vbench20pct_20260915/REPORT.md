# Minimum-10 with local retention disabled: VBench 20%

50 matched prompts, 10s,1344x768,20 steps,seed42. New arms use32midpoint landmarks,Cholesky,FP16 cosine,BF16 tables,(16,16),minimum10 (actual12 frames/group),no reweighting. Top-K uses gemm_radix with all video keys as candidates. Context/sinks remain exact. Dense and original Sol outputs are reused. Timings include first-use compilation and exclude model loading.

| Method | Denoise s | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| dense | 578.216 | inf | 1.000000 | 0.000000 |
| sol | 373.626 | 19.6353 | 0.689055 | 0.229460 |
| tau1 | 392.001 | 23.1901 | 0.787919 | 0.145838 |
| topk10 | 345.602 | 21.1202 | 0.729122 | 0.195131 |

VBench uses the original dimension-specific memberships (14 subject,17 background,14 motion,19 imaging,19 aesthetic prompts per arm), normalized scores x100. No overall score.

| Dimension | Dense | Original Sol | min10 tau1 | min10 topk10 |
|---|---:|---:|---:|---:|
| subject_consistency | 90.5170 | 90.4674 | 90.8370 | 90.6217 |
| background_consistency | 94.0076 | 94.1772 | 94.0178 | 94.0139 |
| motion_smoothness | 99.0153 | 98.9646 | 99.0085 | 98.9844 |
| imaging_quality | 72.1242 | 72.1476 | 71.9900 | 71.7848 |
| aesthetic_quality | 67.8895 | 67.5251 | 67.7571 | 67.9993 |
