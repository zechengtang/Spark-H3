# Attribution of the September 27 Spark PSNR discrepancy

The 50-prompt global-reweight ablation is internally matched but **must not**
be compared directly with the September 20 compiled Spark-H3-10pct baseline.
Its shared route helper passed `--no-torch-compile`. The old Spark run and
the dense reference used per-transformer-block `torch.compile`; the attention
forward remains outside the compiled graph. This was an experiment-protocol
error, not evidence of a 3 dB Spark-kernel regression.

The audit used old 20%-subset cases 2, 4, 6, and 8: 10 s, 1344×768, seed 42,
20 requested grid steps / 19 transformer evaluations. All 50 dense-reference
video hashes and the conditioning manifest match between the old and new
experiments. The four-prompt PSNR means against those same dense videos are:

| Variant | `torch.compile` | PSNR (dB) |
| --- | --- | ---: |
| September 20 Spark TopK10, BF16 threshold | on | 22.776041 |
| Current full reweight, packed route | off | 19.486342 |
| Current full reweight, packed route | on | 22.265985 |
| Current BF16 threshold | off | 19.679423 |
| Current BF16 threshold | on | 22.426622 |
| Current BF16 packed route | off | 19.621335 |

The current full compile-on test kept its attention config exactly equal to
the completed 50-prompt `full` arm. Enabling compilation recovered 2.780 dB
of its 3.290 dB four-prompt gap with the old run. The remaining 0.510 dB is
**not yet attributed**; numeric mode, reblock implementation, and route
selection differ from the old snapshot. Switching only packed to threshold
in the current BF16 eager path changed the mean by just +0.058 dB, so route
execution did not explain the large discrepancy.

The decode path was checked independently: decoding the old Spark latents
with the current FP32 VAE gave 22.776092 dB, essentially the same as their
original videos. The large gap is in the denoised latents, not video encoding
or VAE dtype.

Most importantly, a pure-dense control (no sparse plugin) used the same four
prompts. The current compile-on video latents matched the historical dense
latents **elementwise** on all four. Turning compilation off gave MSE values
of 0.254461, 0.050927, 0.197918, and 0.032300 against those historical dense
latents. Thus the large compile/eager divergence exists without Spark or Sol.
These tests prove the protocol mismatch and exclude a Spark-specific 3 dB
regression; they do **not** yet identify which operation inside the compiled
transformer block creates the divergence. It should not be called ordinary
rounding without a per-block numerical audit.

Artifacts:

- Old Spark and dense: `/autodl-fs/data/h3_experiments/topk_reblock_reweight_50prompt_20260920/`
  and `/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913/dense/`.
- Current eager full: retired; exact source paths are preserved in `docs/uncompiled_delete_preview_20260927.tsv`.
- Four-case BF16 route control: `/autodl-fs/data/h3_experiments/diagnose_spark_psnr_regression_4prompt_20260927/`.
- Four-case compile-on controls: `/autodl-fs/data/h3_experiments/diagnose_spark_compile_4prompt_20260927/`
  and `/autodl-fs/data/h3_experiments/diagnose_spark_full_compile_4prompt_20260927/`.
- Old-latent FP32 re-decode: `/autodl-fs/data/h3_experiments/diagnose_old_spark_fp32_decode_4prompt_20260927/`.
- Pure-dense compile control: `/autodl-fs/data/h3_experiments/diagnose_dense_compile_4prompt_20260927/`.

The public route helper now defaults to real compilation, and compile-off
source artifacts are rejected for reuse by the 50-prompt component runner.
The 50-prompt approximate-tail experiment is paused.

## Ten-prompt compile-on validation

Using the first ten cases of the same 20% subset, the historical compiled
Spark TopK10 scored 23.477736 dB, the eager current `full` scored
19.722497 dB, and the compile-on current `full` scored **23.375732 dB**.
Compilation recovered 3.653 dB of the 3.755 dB old-versus-eager gap; the
remaining mean difference is −0.102 dB. Against old Spark, compile-on `full`
won 4 cases and lost 6, with individual differences ranging from −0.818 to
+1.115 dB. This is close in mean, not bitwise or per-case equivalence.

The 10-prompt run reused the four matching compile-on artifacts above and
generated six new prompt outputs. All ten records have the same attention
configuration and `torch_compile=true` provenance. Raw results are under
`/autodl-fs/data/h3_experiments/diagnose_spark_full_compile_10prompt_20260927/`.

## Twenty-five-prompt compile-on validation

> **Removed on 2026-10-04.** This run used cases 1--25 of the `20pct`
> manifest rather than the canonical `10pct` subset and had no full-50
> successor. Its aggregate table and standalone experiment/output directories
> were removed from the active result namespace. The payload is recoverable
> under `/autodl-fs/data/.trash_h3_noncanonical_25prompt_20261004/`.
>
> The compile/eager diagnosis above remains supported by the independent
> Dense, four-prompt, and ten-prompt controls. No 25-prompt mean or ranking is
> retained as benchmark evidence.
