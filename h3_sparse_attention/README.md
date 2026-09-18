# Sol-Attn and Spark for MiniMax-H3

This package ports the Sol-Attn integration and the `spark_reblock` and
`spark_reweight` primitives used by MiniMax-H3-Sparse. It includes local Sol-Attn
kernels, so inference does not depend on a sibling checkout. No SVOO, SVG,
SVG2, SVG-EAR, Radial, or experimental compensation methods are included.

## Install

From the MiniMax-H3 repository, in the same environment as your H3 pipeline:

```bash
python -m pip install -e '.[cuda]'
```

The pipeline still needs the H3-capable diffusers build and model dependencies
listed in the repository's main README and `requirements.txt`. CUDA execution
requires a matching PyTorch/CUDA installation. CuTe kernels are selected when
available for the GPU; the bundled Triton backend supports other NVIDIA GPUs
with compute capability at least 8.0. Spark requires CUDA and Triton for GPU
execution. Sol-Attn requires contiguous BF16 `[batch, tokens, heads, 128]`
tensors. These are forward/inference kernels.

## H3 pipeline integration

```python
from h3_sparse_attention import install_h3_sol_attn

# pipe is an already loaded MiniMaxH3Pipeline or MiniMaxH3ModularPipeline.
with install_h3_sol_attn(
    pipe.transformer,
    num_inference_steps=50,  # must match the pipeline invocation
    warmup_percent=20,
    sol_tau=1.0,
    sol_thresh_type="diag",
    sol_kv_splits=1,
    sol_dense_layers=1,
    sol_force_local_blocks=True,
) as attention:
    result = pipe(..., num_inference_steps=50)
    stats = attention.summary()
```

The context manager installs attention processors and a layout pre-hook,
then restores them even if inference raises. Call `attention.reset()` before
a second pipeline invocation inside the same context. H3 performs
`num_inference_steps - 1` transformer evaluations; warmup rounds up from the
nominal step count, matching MiniMax-H3-Sparse (10 dense evaluations for 50
steps). The first transformer layer remains dense by default.

Only the generated target-video grid is sparse. Conditioning video, text, and
audio are packed after the target as exact K/V sinks; their query rows are
recomputed with dense attention. The adapter supports H3's packed batch size
one and rejects external attention masks. It preserves Q/K normalization,
RoPE, fused or separate QKV projections, and the output projection.

## Spark inference integration

`install_h3_spark_attn` enables reblocking and reweighting together in the H3
attention path. It follows the updated MiniMax-H3-Sparse installer defaults:

```python
from h3_sparse_attention import install_h3_spark_attn

with install_h3_spark_attn(pipe.transformer, num_inference_steps=20) as attention:
    result = pipe(..., num_inference_steps=20)
    stats = attention.summary()
```

The installer is also exported from `h3_sparse_attention.spark`. It uses the
same reversible processor installation, dense warmup, dense first layer, and
exact context handling as the Sol installer. Call `attention.reset()` before
another pipeline invocation in the same context. Reset/removal releases the
cached reblocking plans and query topology.

Default Spark settings match the source:

- Native mean Top-K routing with ratio `0.1` and `gemm_radix` cutoffs.
- Landmark-v2 Q/K reblocking using opposite second moments, cosine distance,
  fanout 16 with `power_of_two_fanout` scheduling, 32 midpoint landmarks, and minimum 10-frame temporal groups.
- Query representatives selected from the reblocking hierarchy with target
  189 physical blocks and bounds 94–284. These are blocks per representative,
  not a fixed number of representatives; the hierarchy determines actual sizes.
- Query-conditioned K/V and log-mass summaries merged with exact attention.
- The positional local band is disabled after reblocking.

The installer explicitly disables fixed temporal chunks (`landmark_tree_v2_chunk_frames=0`).
To use fixed chunks, set `landmark_tree_v2_minimum_frames=0` and
`landmark_tree_v2_chunk_frames` to the desired frame count and disable
reweighting with `sol_virtual_query_target_blocks=None` and
`sol_virtual_query_levels_up=None`. As in the source, fixed-chunk plans do not
publish the hierarchy required by reweighting. Select
`landmark_tree_v2_fanout_mode="arbitrary_fanout"` for balanced arbitrary child
counts. `landmark_tree_v2_fanout` is an alias for `landmark_tree_v2_children`;
`landmark_tree_v2_root_fanout` and `landmark_tree_v2_final_fanout` independently
control the first nonfinal and final rounds. Both default to inheriting the
ordinary fanout. Reweighting follows the actual reblocking hierarchy. Cosine scoring
defaults to normalized FP16; `H3_LMV2_COS_PRECISION` selects other precision modes.

Sol and Spark now avoid redundant Q/K/V layout copies. Set `H3_SOL_LAYOUT_FAST=0`
before importing the package to use the comparison path. Optional
`sol_route_global_weighted_mean=True` enables global weighted Top-K routing;
`sol_route_global_weighted_side` selects `both`, `query`, or `key`.

Set `sol_route_topk_ratio=None` to use native Sol tau routing with Spark
reweighting (`sol_virtual_query_route_score="native_mean"`). SM120 reweighting
now skips unused ordinary value sums and avoids computing context query rows
that the integration replaces with dense attention.

Configuration overrides are passed as keyword arguments, for example
`sol_route_topk_ratio=0.2` or `sol_virtual_query_levels_up=2`. Setting levels-up
automatically clears the default target-block setting; specifying both is an
error. Small synthetic grids or grids that cannot form whole-frame,
64-token-aligned temporal groups can explicitly use
`landmark_tree_v2_minimum_frames=0`. The production default requires a compatible
temporal grid, as in the source implementation.

SM120 BF16 uses fused reweighting automatically above 8192 packed tokens;
smaller inputs use the streamed exact/skipped implementation.
`H3_SPARK_REWEIGHT_FUSED=0` or `1` selects the source's fallback/fused path on
SM120 for testing. This code does not change environment variables.

Plain `install_h3_sol_attn` retains its base Sol defaults. The standalone Spark
primitives below remain available. Unrelated attention methods and the source's
other compensation pipelines are not exposed by this port.

## Direct Sol-Attn

```python
from sol_attn import sol_attn

out = sol_attn(q, k, v, tau=1.0, thresh_type="diag",
               sink_start=video_tokens, sink_tokens=q.shape[1] - video_tokens)
```

The kernel's sinks make K/V exact. Use the H3 adapter above when context query
rows must also be computed densely.

## Spark primitives

```python
from h3_sparse_attention.spark import spark_reblock, spark_reweight

# samples: [..., T, D], in raster order for grid_shape=(frames, height, width).
blocks = spark_reblock(samples, grid_shape=(frames, height, width))
# blocks.permutation / blocks.inverse_permutation: [..., T]
# blocks.block_indices: [..., T // 64, 64]
# blocks.excluded_indices holds the T % 64 remainder.

# a: [B, P, H, 128] query representatives; k, v: [B, T, H, 128].
ak, av, lm = spark_reweight(a, k, v)
# ak, av: [B, P, H, ceil(T / 64), 128]
# lm:     [B, P, H, ceil(T / 64)] (FP32 log-mass correction)
```

`spark_reblock` is a direct alias for `recursive_landmark_tree_v2_blocks`.
Defaults match the source: 64-token leaves, cosine distance, eight children,
32 midpoint landmarks, flat initial order, single-token groups, and stable
parent order. CPU supports a PyTorch reference path; CUDA takes FP16/BF16.
The output includes remainder rows in the permutation and its inverse.
The `PreparedLandmarkTreeV2Permutation` class remains available from
`h3_sparse_attention.landmark_tree_v2` for repeated CUDA graph execution.

`spark_reweight` is a direct alias for `virtual_summaries`. It requires
matching CUDA FP16/BF16 inputs, contiguous K/V, and contiguous anchor
head/feature dimensions. For each anchor and physical 64-token K/V block,
it computes softmax-weighted keys and values and
`lm = logsumexp(a @ k / sqrt(128)) - a @ ak / sqrt(128)`.
Partial final blocks exclude padding. It does not choose query anchors,
select exact routes, or merge attention outputs. Its output allocation scales
with `B * P * H * ceil(T / 64) * 128`; choose the anchor count accordingly.
`spark_block` remains a compatibility alias for `spark_reblock`.

## Validation and provenance

```bash
OMP_NUM_THREADS=1 python -m pytest tests -q
```

Tests cover CPU reblocking/reference parity, CUDA reblocking, FP16/BF16
reweighting against explicit softmax (including partial blocks and strided
anchors), exact Sol sinks, packed target selection, and H3 processor cleanup.
CUDA graph replay and the Triton fallback are also covered.
GPU tests require a CUDA device; H3 integration tests require H3 diffusers.
Validation results for the current sync are recorded in `PORT_MANIFEST.json`.
Spark tests additionally cover installer defaults, temporal topology, fused and
streamed all-exact parity, and skipped-block reweighting against an explicit
softmax oracle. A synthetic default-Spark case matched the updated source
bitwise; its settings are recorded in `PORT_MANIFEST.json`.
Full model video generation was not run.

`PORT_MANIFEST.json` records source commits and source-file hashes. The Sol
code is copied from the Sana-sol-engine working tree used by MiniMax-H3-Sparse,
including its local kernel patches. Required landmark helpers were extracted
without the other attention methods. The file named `mahalanobis_kmeans.py`
contains shared Hilbert-order and metric-factor utilities, not the clustering
method. Exact-branch kernels are extracted from the source compensation modules
without importing their compensation methods.
See `sol_attn/THIRD_PARTY_NOTICES.md` and its included license for kernel notices.
