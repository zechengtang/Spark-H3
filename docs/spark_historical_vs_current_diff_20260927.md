# 2026-09-27: historical Spark snapshot vs current Diffusers Spark

## Scope and observation

Compared the frozen 2026-09-20 implementation at
`/autodl-fs/data/h3_experiments/topk_reblock_reweight_50prompt_20260920/snapshot/h3_sparse_attention`
with the current `h3_sparse_attention/` tree. The investigation includes
production-shape, same-input single-layer replay. It identifies the first
actual divergence in that layer and isolates its responsible source branch;
it is **not** yet a full-run 19-evaluation causal bisect.

The five-case reproduction records at
`/autodl-fs/data/h3_experiments/blog50_bitwise_repro_5prompt_20260927`
show current Dense and Sol 5/5 video/audio latent bitwise matches, but current
Spark BF16 threshold and packed-external both 0/5. The frozen-code replay at
`/autodl-fs/data/h3_experiments/blog50_frozen_spark_repro_5prompt_20260927`
matches the original Spark video/audio latents 5/5. Thus the old Spark output is
reproducible, and the discrepancy is specific to the current Spark path (or its
interaction with the current runner), not evidence that historical compile was
disabled. The current records attest `compile_wrapped_blocks=50`; the frozen
replay records do likewise.

The current reproduction explicitly uses `sol_global_anchor_dtype="bfloat16"`,
`sol_tail_granularity="query"`, `sol_reweight_summary_math="tensorcore"`,
`sol_reweight_logmass_key="stored"`, `sol_reweight_components="full"`,
`sol_virtual_query_levels_up=99`, and `sol_route_topk_execution="threshold"`
for the primary Spark comparison. This is documented in its `protocol.json`.
The frozen replay loads the original config from the historical `protocol.json`.
Both report one active global virtual block, fanout 16, 72,576 video tokens,
73,584 total tokens, 19 evaluations, four warmup evaluations, and 735 sparse
attention calls per case. Since 72,576 is a multiple of 64, the video-tail
branch is not exercised in this reproduction.

## Production-shape, same-input result: first real divergence

The capture script `scripts/diagnose_blog50_spark_first_sparse_capture_20260927.py`
ran the compiled pipeline with cached conditioning for historical prompt index
05 (seed 42, 20-step grid), then stopped at the **first sparse evaluation**
(evaluation index 4), layer 1, *before* Spark attention. It saved the complete
post-RoPE Q/K/V for all 56 heads, one real 72,576-video-token / 73,584-total-
token sequence, at
`/autodl-fs/data/h3_experiments/blog50_spark_intermediate_allheads_20260927/`.
Thus both implementations replayed **identical input tensors**, not merely the
same prompt. The replay script is
`scripts/diagnose_blog50_spark_single_head_replay_20260927.py`; frozen and
current runs used GPU0/1 respectively. Neither touched historical artifacts.

The first observed divergence is the landmark-tree **reblock permutation**:

| Stage | Different elements, frozen vs current | Affected heads |
| --- | ---: | --- |
| Query permutation | 1,711 token IDs | head 13: 551; head 51: 1,160 |
| Key permutation | 1,122 token IDs | head 54: 1,122 |
| BF16 global anchor | 0 | none |
| Scalar Top-K cutoff | 172 FP32 values | head 13: 7; head 51: 19; head 54: 146 |
| Weighted K / V summaries | 43,301 / 45,237 elements | head 54 |
| Log mass | 357 elements | head 54 |
| Final BF16 attention output | 589,642 elements | head 13: 47,008; head 51: 25,070; head 54: 517,564 |

The 8-head sample (0, 1, 7, 14, 21, 28, 35, 42) was completely bitwise
equal through final attention. This is why the narrower replay initially
appeared to exclude reblock. Repeating the head-0 call three times also
preserved identical final outputs across implementations, including after the
plan's CUDA-graph replay path activated. All-head coverage was necessary.

## Confirmed source-level cause of that divergence

The current tree takes a new `fused_midpoint_directions` branch when
group size=1, midpoint landmarks=32, cosine distance, and linear aggregation
(`h3_sparse_attention/landmark_tree_v2.py:507-529`). The frozen tree instead
uses `indexed_interval_means` then `build_cosine_directions`
(frozen `landmark_tree_v2.py:466-477,:499-502`). The fused kernel directly
computes normalized landmark vectors, their FP32 dot-product distance, and
proxy-tree directions (`h3_sparse_attention/landmark_v2_fused_node.py:35-107`).

To isolate the branch, the replay temporarily replaced **only** the current
`fused_midpoint_directions` function with a wrapper calling the frozen-style
`indexed_interval_means + build_cosine_directions`, leaving the rest of the
current implementation unchanged. On real heads 13, 51, and 54, this restored
**bitwise equality at every recorded stage**, including query/key permutation,
cutoff, summaries, log mass, sparse output, and final attention output. This
is a causal A/B isolation for the first sparse layer. The wrapper also
computed both direction tensors on the same real inputs: across three calls,
112,517 FP32 elements differed, with maximum absolute difference
1.1920928955078125e-7. Small direction-rounding differences are sufficient
to change near-tie landmark routing and then attention. The precise internal
arithmetic operation responsible for each direction ulp difference has not
been separately bisected.

This proves the new fused midpoint-direction construction causes the **first
observed single-layer bitwise divergence**. It does not yet prove it is the
only contributor to the 5/5 full-run latent mismatch. A complete 19-evaluation
replay with this one branch restored would establish that stronger claim.

The current direct-root scoring path and SM120 score tile 128 (versus 64) in
`landmark_tree_v2.py:58-87,:559-568` and
`landmark_v2_cosine_fast.py:20-28,:107-123` did **not** prevent full stage
parity after restoring the old direction builder on the affected real heads.
Therefore they are not needed to explain this first difference. The current
CuTe threshold mainloop additions in `spark_reweight_sm120.py:51-85,:382-478,
:519-640` likewise produced bitwise-equal final attention once permutations
were aligned. The BF16/tensorcore/stored/full reweight-summary path is not an
independent first cause in this captured layer.

The new packed-external route remains a distinct algorithm: query-centroid
GEMM, exact `topk`, and an int32 packed bitmask in `sol_topk_cutoff.py:375-473`,
selected by `spark_integration.py:956-973`. It may have its own bitwise
differences from the frozen scalar gemm-radix cutoff, but the current
**threshold** arm already fails 5/5; external routing is not required for the
historical mismatch.

## Other source differences and applicability

- **Preset defaults changed, but were overridden in this comparison.** The
  frozen `H3SparseAttentionConfig.spark()` preset uses target-189 reweighting
  (`processor.py:188-209`); the current preset uses a global/root anchor and
  packed-external routing (`processor.py:259-287`). The historical frozen
  experiment explicitly overrode the target selection with
  `sol_virtual_query_levels_up=99, sol_virtual_query_target_blocks=null`;
  the current threshold arm explicitly restores threshold execution and BF16
  anchor. Both records report one active virtual block. It would be incorrect
  to attribute this particular 0/5 result merely to a changed default.
- **Sparse-query/dense-suffix handling changed, but should be inactive here.**
  The current `spark_integration.py:572-599,:755-872` limits sparse query rows
  to complete video blocks and overwrites the remaining rows densely; the old
  `spark_integration.py:560-597` also passed `_query_tokens=video_tokens` when
  the video prefix was block-aligned and overwrote context query rows densely.
  The reproduction has exactly 1,134 complete video blocks, so both evaluate
  the same query-row range. This remains a real semantic difference for other
  non-aligned shapes, not a demonstrated cause here.
- **K/V permutation fusion** replaces two head-wise gathers with one Triton
  gather (`spark_integration.py:686-713`,
  `landmark_tree_v2_triton.py:32-123`). With the affected real-head
  permutations restored, the final attention output became bitwise equal;
  this fusion is therefore not needed to explain the first difference.
- **FP32 anchors, block-granularity approximate tail, alternate summary math,
  optional dense evaluations, SM90/100 dispatch, and ComfyUI compact int32
  reblock routing** are new capabilities but are not selected by the recorded
  BF16/query-granularity/SM120 Diffusers threshold configuration. In
  particular, `landmark_tree_v2.py:340-349,:588-635` only takes the compact
  approximate-int32 route when `compact_direct_route=True`; the Diffusers
  plan construction at `spark_integration.py:375-397` does not set it.

## Remaining verification

If historical bitwise compatibility is a requirement, introduce an explicit
legacy-direction mode or revert this fused midpoint branch, then replay one
complete prompt using the historical BF16/threshold/global configuration.
Only if the full latent becomes bitwise identical should the branch be called
the sole cause of the 5/5 failure. A later case/step may expose another
independent divergence. Keep the historical result and source snapshot intact.
