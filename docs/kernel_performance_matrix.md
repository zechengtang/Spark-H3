# Kernel performance matrix

This table tracks comparable Dense-versus-Spark kernel measurements across GPU
architectures and inference profiles. Each duration cell reports **speedup**
followed by `(mean Dense DiT time -> mean Spark DiT time)`. Speedups are rounded
to two decimal places and times to one decimal place after aggregation.

| GPU | SM | Model / profile | DiT evals | Warmup steps | Dense layers | Kernel / route | Commit | 5s / 120f | 10s / 240f | 14.4s / 345f |
|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|
| A800-SXM4-80GB | 80 | MiniMax-H3 native | 19 | 4 | 1 | fused virtual-query / legacy+threshold / Top-K 10% | `1185195` | **1.41x** (326.3s -> 231.4s) | **1.70x** (988.0s -> 581.5s) | **1.84x** (1841.9s -> 1003.0s) |
| RTX PRO 6000 Blackwell Server Edition | 120 | MiniMax-H3 native | 19 | 4 | 1 | fused virtual-query / legacy+threshold / Top-K 10% | `57d401b` + dynamic-cache working tree | **1.42x** (197.8s -> 139.5s) | **1.72x** (583.9s -> 338.7s) | **1.87x** (1070.1s -> 571.8s) |

## Benchmark profile

- Resolution: 1344x768; durations: 120, 240, and 345 frames.
- Precision and seed: BF16, seed 42.
- MiniMax-H3 native requests 20 inference steps and executes 19 transformer/DiT
  evaluations. The table records actual DiT evaluations so that profiles whose
  requested and executed step counts match can be added without ambiguity.
- Spark configuration: 10% Top-K, legacy midpoint, threshold routing, four
  dense warmup evaluations, and one forced dense layer per sparse evaluation.
- Runtime warmup is architecture-row specific. The A800 row uses a discarded
  pass through the four configured dense evaluations plus the first sparse
  evaluation. The PRO 6000 row uses the validated short pass with three
  requested steps: exactly one dense and one sparse evaluation. In both cases,
  the measured pass reuses the compiled callable and excludes first-sparse
  compilation time.
- Timing covers only CUDA-synchronized DiT/denoising. Pipeline loading, VAE
  decode, and other end-to-end work are excluded.
- Each displayed value aggregates four independent Dense measurements and four
  independent Spark measurements. Speedup is calculated from the unrounded
  means: `mean(Dense) / mean(Spark)`.
- All measured Spark runs completed 19/19 evaluations with finite output. The
  A800 row used `sm80_fused_virtual_query`; no fallback backend was used.
- The PRO 6000 measurements use four prompts (cases 2, 13, 31, and 33). Raw
  records and the aggregate are under
  `${H3_EXPERIMENTS_ROOT}/pro6000_task1_matrix_20261003`; its four
  formal Spark runs at every duration reported zero new fused-kernel compile
  calls after runtime warmup.

## Adding results

Add one row for each GPU, model/inference profile, kernel backend, and tested
commit. Keep actual DiT evaluations explicit. Results from different profiles
may share this matrix, but absolute times should only be compared when their
profile and evaluation count match.
