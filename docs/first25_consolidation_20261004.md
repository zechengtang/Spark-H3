# First-25 result consolidation (2026-10-04)

The standalone cases-1--25 results that already had completed 50-prompt
successors were removed from the top-level experiment/output namespaces and
consolidated under their canonical full-50 directories. This prevents the
biased first half of the `20pct` manifest from being mistaken for the
canonical `10pct` subset or an independent benchmark result.

## Canonical results

- `diffusers_spark_topk30_50prompt_10s768p_20260928`
- `route_2x2_full50_4gpu_20261004_5s768p`
- `route_2x2_full50_4gpu_20261004_10s768p`

Each canonical experiment contains `first25_consolidation.json`. Historical
first-stage experiment metadata is retained under `provenance/`; historical
payload that was not part of the formal full-50 arms is retained under
`provenance_outputs/`. The route full-50 method directories now contain the
available per-case videos, latents, and denoise records directly, using hard
links where the source filesystem allowed it. Their full-50 manifests resolve
all 50 video records inside the canonical output directory.

This was a namespace and provenance consolidation, not a regeneration or
metric recomputation. No unique artifact was irreversibly deleted. Absolute
source references in the repository and known full-50 downstream experiment
metadata were rewritten to the consolidated locations.

## Removed standalone 25-prompt experiments

The following noncanonical cases-1--25 results had no full-50 successors and
were removed from the active experiment/output namespaces on 2026-10-04:

- `diagnose_spark_full_compile_25prompt_20260927`
- `diffusers_spark_block8x8_25prompt_10s768p_20260928`

Their payload is recoverable under
`/autodl-fs/data/.trash_h3_noncanonical_25prompt_20261004/`. Their 25-prompt
aggregate values are not retained as benchmark evidence. The smaller
compile/eager controls remain available, while the block8x8 hypothesis would
require a new complete 50-prompt experiment before any quality conclusion.
