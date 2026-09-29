# ComfyUI 10s sparse-step TopK audit (2026-09-28)

One 10s/768p prompt, seed 42, 20 actual ComfyUI evaluations (four dense,
16 sparse), same conditioning and 5-step excluded warmup per method/GPU.
GPU0/1 each ran the three methods; official Sol and official TopK were run
in opposite orders on the two GPUs. Times below are mean of both GPUs.
All three use block-granularity tail and extra_tokens=0. The official entries
use ComfyUI's `BlockSparseAttention` node (`sol-attn` tau=1 or `sla`
keep_percent=10). The native Spark TopK-only entry uses the plugin's
experimental `ablation_mode=topk_only`, with reblock and global reweight both
disabled. This is a speed ablation, not a claim of identical output: the
native plugin also uses its video-first producer and treats all non-video
query rows as exact.

| Path | Whole denoise | Sparse evaluation, average of steps 4–19 | Sparse attention core per evaluation |
| --- | ---: | ---: | ---: |
| Official Sol tau=1 | 253.46 s | 9.102 s | 3.451 s |
| Official TopK10 | 230.29 s | 7.645 s | 2.024 s |
| Native Spark TopK10-only | 233.54 s | 7.872 s | 2.331 s* |

Official TopK versus Sol saves 23.17 s per generation (9.14%, 1.101x) and
1.457 s per sparse evaluation (16.0%, 1.191x). Its attention core saves
1.427 s per sparse evaluation (41.4%). Native TopK-only keeps most of the
benefit: it saves 19.92 s versus Sol (7.86%, 1.085x), but takes 3.25 s
longer than official TopK over the generation, or 0.227 s per sparse step.

*The Spark `spark_preprocess_route_tail_exact` range includes preprocessing.
The official `sol_route_tail_exact` does not include its
`sol_fused_rope_quant_chunk` range. Comparing both matched ranges, the
official TopK attention path is (32.38 + 4.44)/16 = 2.301 s per sparse
evaluation, versus 2.331 s for native Spark: only about 0.03 s per step.
The remaining full-step gap comes mainly from differing QKV/RoPE/copy and
output-layout paths, not a 0.3 s per-step route-kernel regression.

On the 16 sparse evaluations, the native path's Q/K RMSNorm+RoPE range is
2.72 s larger, its separate V buffer copy costs 1.23 s, and output layout
restoration costs 1.11 s. Its QKV projection is 2.11 s faster, while the
matched attention preprocessing/route/exact ranges differ by about 0.47 s.
These ranges plus smaller terms account for the observed 3.25 s whole-run
gap. The comparison is between *exclusive, like-for-like groups*: native
Spark's preprocess range includes work that official Sol records under
`sol_fused_rope_quant_chunk`, so raw range names must not be compared alone.

The official chunked TopK producer writes quantized attention carriers
directly and never materializes full BF16 Q/K/V. Reblock needs post-RMSNorm,
post-RoPE Q/K for its data-dependent plan, and global reweight needs K/V.
Consequently a high-performance graft must preserve a single QKV projection
and a single routing pass while making those intermediates available to the
plan and weighted summaries; invoking both full producer paths would be an
invalid speed comparison. The official pooled-tail fixed-TopK core also makes
one route pass to choose a cutoff and another to consume the pooled tail;
the native Spark path uses one route pass. This is an implementation detail,
not evidence that the native route computes TopK twice.

The 3.25 s full-denoise difference is therefore a **producer/layout** gap,
not a TopK-selector gap. A production-quality official-BSA integration must
rework the chunked producer and the reblock-plan handoff together. Merely
calling official `sol_attn_chunked` after building Spark Q/K/V would project
or preprocess Q/K/V twice, and applying reblock/reweight after the official
core would rerun route/exact. Neither is a valid optimized implementation.
The measured 12.19 s reblock-plan cost also means an ideal swap to the 230.29 s
official TopK baseline already reaches roughly 242.48 s *before* reweight and
other Spark work; optimizing the selector in isolation cannot establish a
1.05x production speedup over the roughly 253.5 s Sol baseline.

A screened Sol-producer-style BF16 materializer fused Q/K RMSNorm+RoPE with
the V copy in one tiled kernel. On a real-size 16K-token projected-QKV chunk,
the isolated operation fell from 1.16–1.18 ms to 1.00 ms. With about five
chunks per sparse attention call this is only around 0.7 s over a generation
before the packed-RoPE preparation cost. With valid rotation matrices its
Q/K outputs matched the existing path at 99.999% of BF16 elements; the
remaining differences reached 0.016/0.031 absolute. This small benefit does
not close the 3.25 s producer/layout gap, so the candidate was **not** enabled
or retained in production source. Raw A/B artifacts are `fused_materializer_gpu{0,1}.json`
in the experiment directory. The optimized official-producer graft remains
an unimplemented kernel-level change, not a completed speed result.

An independent kernel-only A/B replays the **same captured real Q/K/V** with
identical sinks, block tail, forced-local policy, no reblock and no reweight.
The official route took 45.88/45.95 ms on GPU0/1; native Spark took
46.23/46.12 ms, only 0.36/0.18 ms slower (0.8%/0.4%). This is the direct
test of whether the implanted TopK kernel is inefficient; it is not.

The production Spark path still includes reblock and reweight, so the
TopK-only result must not be quoted as production Spark speed. In the
separate four-prompt current-code benchmark, full Spark block took 246.05 s
versus Sol 253.55 s (1.0305x). The full-path profile measured about
12.19 s spent in reblock-plan construction over one generation; this is
the largest measured counterweight to the pure TopK gain. Because the
topology and local-exact policy also change with reblock, these measurements
do not constitute a strictly additive decomposition of the final 12.5 s gap.

Artifacts: `/autodl-fs/data/h3_experiments/comfyui_sol_topk_sparse_step_20260928/`
(`summary.json`, per-GPU `ranges.json`, `kernel_ab_gpu{0,1}.json`).
Runner: `scripts/comfyui_sol_topk_sparse_step_20260928.py` and
`scripts/probe_comfy_official_vs_spark_topk_20260928.py`.

## Production output-scatter optimization

The official exact kernel now optionally scatters video-first Spark rows
directly into the model's natural output layout, eliminating the Python
`output.index_select` pass. It is enabled by default for Spark; setting
`H3_SPARK_DIRECT_OUTPUT=0` preserves the old path for diagnostics. The full
10s/768p, 20-evaluation, excluded-5-step-warmup A/B used prompt 1, seed 42,
full block-granularity Spark TopK10/reblock/reweight, and reversed method
order across GPU0/1:

| GPU | Old output restoration | Direct scatter | Saving | Latent SHA256 |
| --- | ---: | ---: | ---: | --- |
| 0 | 245.843 s | 244.777 s | 1.066 s | identical within GPU |
| 1 | 248.639 s | 247.148 s | 1.491 s | identical within GPU |

The paired mean saving is **1.279 s** per generation, with bitwise-identical
saved latents (also the same SHA256 across both GPUs). The isolated real-QKV
kernel A/B saved 0.91–1.24 ms per attention call. This is a validated
production optimization, but it does not by itself achieve the 1.05x full
Sol/Spark speed target or complete the chunked-producer/reblock integration.
Full A/B artifacts are under
`/autodl-fs/data/h3_experiments/comfyui_direct_output_full_ab_20260928/`.
