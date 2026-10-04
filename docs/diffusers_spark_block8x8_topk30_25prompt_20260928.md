# Retired first-25 Spark tail and TopK ablations

> **Result removed on 2026-10-04.** The historical block8x8 and TopK30 tables
> in this document used cases 1--25 of the `20pct` manifest, not the canonical
> `10pct` subset. The two sets overlap on only 13 prompts, and the first half
> omits the Imaging and Aesthetic source suites. Those tables are therefore no
> longer retained as benchmark evidence.

The standalone block8x8 experiment had no full-50 successor. Its active
experiment and output directories were removed; the payload is recoverable
under `/autodl-fs/data/.trash_h3_noncanonical_25prompt_20261004/`. A block8x8
quality claim requires a new complete 50-prompt experiment.

TopK30 does have a completed 50-prompt successor. Use
`docs/diffusers_spark_topk30_50prompt_20260928.md` and its canonical experiment
directory, `diffusers_spark_topk30_50prompt_10s768p_20260928`, for all TopK30
quality and runtime reporting. The superseded first-25 source is retained only
as provenance inside that full-50 experiment and is not an independent result.
