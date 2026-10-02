# Kernel performance matrix

This table tracks comparable Dense-versus-Spark kernel measurements across GPU
architectures and inference profiles. Each duration cell reports **speedup**
followed by `(mean Dense DiT time -> mean Spark DiT time)`. Speedups are rounded
to two decimal places and times to one decimal place after aggregation.

| GPU | SM | Model / profile | DiT evals | Warmup steps | Dense layers | Kernel / route | Commit | 5s / 120f | 10s / 240f | 14.4s / 345f |
|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|
| A800-SXM4-80GB | 80 | MiniMax-H3 native | 19 | 4 | 1 | fused virtual-query / legacy+threshold / Top-K 10% | `1185195` | **1.41x** (326.3s -> 231.4s) | **1.70x** (988.0s -> 581.5s) | **1.84x** (1841.9s -> 1003.0s) |

## Benchmark profile

- Resolution: 1344x768; durations: 120, 240, and 345 frames.
- Precision and seed: BF16, seed 42.
- MiniMax-H3 native requests 20 inference steps and executes 19 transformer/DiT
  evaluations. The table records actual DiT evaluations so that profiles whose
  requested and executed step counts match can be added without ambiguity.
- Spark configuration: 10% Top-K, legacy midpoint, threshold routing, four
  dense warmup evaluations, and one forced dense layer per sparse evaluation.
- Before each Spark measurement, a discarded pass executes the four configured
  dense evaluations plus the first sparse evaluation. The measured pass reuses
  the same plugin and static plan, excluding first-sparse compilation time.
- Timing covers only CUDA-synchronized DiT/denoising. Pipeline loading, VAE
  decode, and other end-to-end work are excluded.
- Each displayed value aggregates four independent Dense measurements and four
  independent Spark measurements. Speedup is calculated from the unrounded
  means: `mean(Dense) / mean(Spark)`.
- All measured Spark runs completed 19/19 evaluations with finite output using
  `sm80_fused_virtual_query`; no fallback backend was used.

## Adding results

Add one row for each GPU, model/inference profile, kernel backend, and tested
commit. Keep actual DiT evaluations explicit. Results from different profiles
may share this matrix, but absolute times should only be compared when their
profile and evaluation count match.
