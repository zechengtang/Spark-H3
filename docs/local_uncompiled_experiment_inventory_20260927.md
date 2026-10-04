# Local uncompiled-experiment inventory (2026-09-27)

> **Post-cleanup note:** This document records the pre-deletion audit. On
> 2026-09-27, 12,619 exact files from its confirmed compile-off groups were
> removed under `uncompiled_delete_preview_20260927.tsv`; the execution log is
> `uncompiled_delete_executed_20260927.json`. The blog-facing four-prompt
> ablation and all four compiled 50-prompt main arms were deliberately retained.
> Paths described below may therefore be historical rather than present.

This was originally a **read-only deletion-review inventory**. The primary
scan covers top-level directories under
`/autodl-fs/data/h3_experiments/` and their linked H3 output directories.
There are 351 experiment directories and 192 top-level `protocol.json` files:
12 explicitly say `torch_compile=false`, four say `true`, and 176 omit the
field. A missing field alone is **not** evidence that compilation was off.
Counts below are JSON generation records, not necessarily distinct new model
runs; several experiments reused earlier records.

## Explicit `torch_compile=false` in the saved protocol

All paths in this section are relative to `/autodl-fs/data/h3_experiments/`.
These ten completed experiments have `status=complete`:

| Experiment directory | Generation records |
| --- | ---: |
| `blog_ablation_4prompt_nowarm_layer0_5s480p_seed42_20260923` | 12 |
| `vbench_core5_25prompt_nowarm_layer0_5s480p_seed42_20260923` | 75 |
| `step4_sparsity_10prompt_5s480p_seed42_20260923` | 60 |
| `step4_sparsity_25prompt_5s480p_seed42_20260924` | 150 |
| `step5_sparsity_25prompt_5s480p_seed42_20260924` | 150 |
| `step5_sparsity_blog4_5s768p_seed42_20260924` | 24 |
| `early_steps_sparsity_10prompt_5s480p_seed42_20260924` | 160 |
| `early_steps_sparsity_25prompt_5s480p_seed42_20260924` | 400 |
| `cumulative_step4_sparsity_25prompt_5s480p_seed42_20260924` | 150 |
| `cumulative_step4_sparsity_blog4_5s768p_seed42_20260924` | 24 |

`cumulative_step4_sparsity_25prompt_5s480p_seed42_20260924.aborted_retain20pct_20260924`
also says false, but has no generation records. The four-record
`diagnose_spark_psnr_regression_4prompt_20260927` is an intentional eager
diagnostic, not a production benchmark.

## Explicit compile-off in the saved/associated runner

The seven completed budget experiments below use source scripts that pass
`--no-torch-compile`; their protocols do not reliably record the flag:

| Experiment directory | Generation records |
| --- | ---: |
| `dense_step_ablation_5s_480p_20260920` | 84 |
| `dense_layer_ablation_5s_480p_20260920` | 208 |
| `slope_budget_5s_480p_20260920` | 32 |
| `exp_budget_5s_480p_20260920` | 28 |
| `sigma_budget_5s_480p_20260920` | 20 |
| `shallow_slope_5s_480p_20260920` | 24 |
| `reverse_layer_probe_5s_480p_20260921` | 52 |

The completed `giraffe_768p10s_seed42_three_variants_20260923` script also
passes `--no-torch-compile` for its three arms. The
`diffusers_sol_spark_4prompt_5s768p_20260926` runner explicitly passes the
flag and has 24 generation records.

The completed `fasth3_vbench50_four_variants_20260921` protocol has
`variants.*.contract.torch_compile=false` for all four arms (`v1_dense`,
`v1_dense_topk10_reblock_reweight`, `v1_vsa`, `v2_vsa`). Its output tree has
50 MKV videos per arm. This nested flag would be missed by a scan of only
top-level `protocol.torch_compile`.

`fasth3_0753_three_weights_reblock_20260918/results.json` also contains six
rows whose nested `settings.contract.torch_compile` is false. Its
`pre_layerwise_offload/` and `superseded_sol_attempt/` subdirectories have
related compile-off settings; review this experiment as a whole rather than
deleting individual JSON files.

Four `cosine_temporal_hypotheses_v1/stage3_case_*.json` diagnostic records
explicitly say `torch_compile=false`. Nine older isolated-run *smoke*
generation manifests contain the same false flag, under
`dense_sol_25_20260910`, `lmv2_f4_25_20260910`,
`lmv2_f8_25_20260910`, `lmv2_f4_orders_25_20260910`,
`lmv2_fullcov_25_20260910`, `lmv2_initial_order_25_20260910`,
`lmv2_query_mk_center_25_20260912`, and `diag_fullcov_budget_20260912`
(including its `_v2` copy). These are nested smoke artifacts, **not** proof
that the parent experiments' main results were uncompiled. The two
`why_mean_works/lmv2_f4_g1_jensen_phases_20260910/capture_*.json` records
are also explicitly compile-off captures.

## Silent no-op in the old open-source Benchmark path

`blog_ablation_4prompts_rerun_20260922` has 20 generation records (four
prompts across `dense_capture`, `dense`, `bsa`, `reblock`, and
`global_reweight`). Its frozen runner points `H3_IMPL_REPO` at the open-source
MiniMax-H3 tree and does not pass a compile-off flag, but the saved
`snapshot/h3_sparse_attention/__init__.py` does not export an acceleration
hook. At that time Benchmark's `_impl_bootstrap.acceleration_symbols()`
silently substituted a no-op. Thus the **transformer blocks were not
compiled**, despite the nominal default. This is the source of the blog's
four-prompt Reblock and Reweight ablation tables. Both comparisons are
internally matched; neither should be compared to compiled 50-prompt
absolute PSNR. `blog_ablation_4prompt_redense_20260922` has the same old
implementation snapshot but is incomplete and has no generation records.

The six legacy Benchmark sparsity runner sources were subsequently updated
to require compilation and no longer pass a compile-off flag. Their archived
`runner_source.py`, protocols, and results remain unchanged as historical
eager evidence. The updated runners reject an existing uncompiled protocol,
and the Benchmark output-provenance guard rejects old latents. A compiled
rerun must use a fresh experiment name/output directory and record the actual
wrapped-block count.

## Inherited compile-off shared route helper (strong reconstruction)

These experiments do not consistently store an actual compile flag per
record. Their execution chain used the old route helper, which explicitly
passed `--no-torch-compile` before its 2026-09-27 correction:

| Experiment directory | State / generation records |
| --- | --- |
| `diffusers_spark_reweight_precision_25prompt_10s768p_20260926` | Complete; 75 records, BF16 / anchor-FP32 / Comfy-FP32 arms |
| `diffusers_spark_reweight_components_10prompt_10s768p_20260926` | Complete; 30 new records and ten `full` cases reused from the 25-prompt precision run |
| `diffusers_spark_reweight_components_50prompt_10s768p_20260926` | Complete; 200 records, including 55 reused arm–prompt outputs |
| `diffusers_spark_anchor_tail_10prompt_10s768p_aligned_20260926` | Complete; 30 records, query / block / FP32 arms |
| `diffusers_spark_tail_granularity_50prompt_10s768p_20260927` | Paused; 20 prepared records are reused from the aligned ten-prompt run; no new inference results |

The 50-prompt component `full` result of 20.225 dB is **not** a controlled
comparison against the historical compiled Spark TopK10 result of 23.301 dB.
The former compiled 25-prompt attribution run used cases 1--25 of the
`20pct` manifest rather than the canonical `10pct` subset. Its active result
directories and aggregate values were removed on 2026-10-04; the independent
Dense and smaller compile controls remain the supported attribution evidence.

## Intentional diagnostic / not a deletion recommendation

`diagnose_dense_compile_4prompt_20260927` contains four eager and four
compiled dense controls. The eager half is intentionally uncompiled.
`diagnose_spark_psnr_regression_4prompt_20260927` is the four-case eager
BF16 threshold control noted above. Keep both if the compile-mismatch
attribution needs to remain reproducible.

The historical 2026-09-20 50-prompt Spark TopK10/TopK20 benchmark and its
dense/Sol references used real per-block compilation. The noncanonical
`diagnose_spark_full_compile_25prompt_20260927` result was removed from the
active namespace on 2026-10-04 and is no longer benchmark evidence.

The 176 protocols without a top-level compile flag and other experiment
directories lacking a protocol remain **unclassified**, not approved for
deletion. A directory should only be removed after checking downstream reuse
and source references (especially the blog and conditioning/dense manifests).
