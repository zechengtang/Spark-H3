# Current ComfyUI full-path speed audit (2026-09-28)

This audit uses the current ComfyUI-native Spark backend, the same archived
prompt 1 and conditioning, seed 42, 5s/10s 768p, and 20 ComfyUI model
evaluations. Each method has an excluded 5-step warmup, a non-profiled 20-step
control, and a separately profiled 20-step run. The paired control timings,
not isolated kernel microbenchmarks, determine the overall speed ratio.

| Duration | Sol tau=1, extra=0 | Spark block | Sol/Spark block | Spark query | Sol/Spark query |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5s | 95.279s | 98.568s | 0.967× | 99.669s | 0.956× |
| 10s | 252.419s | 244.880s | 1.031× | 248.698s | 1.015× |

The 10s Spark block target of 1.05× requires at most 240.399s on this exact
Sol control, an additional 4.481s reduction from the measured 244.880s.

The 10s profiled run attributes 54.821s to Sol's sparse route/tail/exact
stage. Spark block's plan-and-attention parent range is 49.590s: the reblock
plan parent range is 12.192s, followed by 37.396s for Spark
preprocess/route/tail/exact. Thus the Spark attention side saves about 5.23s
over Sol, while other Spark-specific overhead consumes most of that gain.
The two reblock-plan CUDA graph replays total 19.235s of **exclusive ranges**
but overlap on separate streams; adding them to the 12.192s plan parent would
double-count work. Likewise the QKV producer and projection ranges are nested.

The corresponding 5s exclusive stage totals are Sol route/tail/exact 15.94s;
Spark block preprocess/route/tail/exact 12.43s and plan parent about 8.27s.
At that length, plan overhead exceeds the 3.51s sparse-core saving.

A warmed single-layer 10s plan kernel trace on genuine captured Q/K found
the largest *kernel active-time* categories to be fused node splitting
(4.01ms), full-token projection (3.20ms), `gatherKthValue` (3.41ms across
the two int32 variants), score calculation (1.84ms), key compaction
(1.82ms), and cosine proxy fitting (1.38ms). These categories can overlap
and are not a decomposition of the 12.192s end-to-end plan range.

Initial one-layer tuning on the same capture rejected the following options:

- Fused-node 4/8 warps: plan median 16.96/19.44ms versus 15.93ms with 2.
- Fused-node tile 32/128: 16.01/17.99ms versus 15.98ms with tile 64;
  processing two/four rows per step regressed further to 21.40/51.39ms.
- Projection tile 64/32: 16.45/16.45ms versus 15.90ms with tile 128.
- Direct projection-kernel launch tuning gave at most a 0.013ms local gain
  (tile64/4 warps versus tile128/8 warps); other launch shapes regressed.
- A parity-preserving Triton batched int32 k-th selector: 17.80ms versus
  15.92/15.94ms in baseline A/B/A. The candidate was removed from source.
- A parity-preserving radix-histogram int32 k-th selector: 19.48ms versus
  16.09/15.99ms in baseline A/B/A. The candidate was removed from source.
- A full-node Triton sorting selector was abandoned before timing: the root
  segment is about 72k tokens, making on-chip sorting impractical. The
  experimental branch was removed.
- Compact int32 keys in fused-node routing changed roughly 0.46–0.48 million
  Q/K permutation entries while improving the plan median by only 0.07ms
  (15.85ms versus 15.92/15.92ms A/B/A); rejected and removed.
- Fused-node size cap 512 tokens made no measurable difference; reducing it
  to 256 tokens worsened the plan median to 17.66ms versus 15.92/15.95ms.
  Neither changes the production setting.
- Sorting only small int32 cutoff segments preserved the permutation but
  changed the plan median by just 0.04ms (15.50ms versus 15.57/15.52ms
  A/B/A). This is below the end-to-end optimization requirement, so the
  experimental branch was removed.
- Replacing the parallel split fused-root plans with one combined Q/K plan
  regressed to 17.62ms versus 15.93/15.97ms A/B/A and changed 40,538 Q plus
  71,532 K permutation positions; this alternative was removed.

A separate four-prompt, one-repetition, current-code speed check used the
same 20-step ComfyUI sampler with an excluded 5-step warmup on GPU0–1. Its
records are under `comfyui_current_spark_4prompt_speed_20260928`:

| Duration | Sol tau=1 mean | Spark block mean | Mean paired Sol/Spark block | Spark query mean | Mean paired Sol/Spark query |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5s | 94.568s | 97.832s | 0.9666× | 99.039s | 0.9549× |
| 10s | 253.553s | 246.047s | 1.0305× | 249.764s | 1.0152× |

On the four-prompt 10s mean, reaching 1.05× would require Spark block at
241.48s or less, about 4.57s below the present mean. No candidate screened
above delivers a material fraction of this gap; the 1.05× target is not met.

No production ComfyUI optimization is claimed by these screens. Further
candidates must pass isolated speed and permutation checks before full-video
validation. The full-path audit artifacts are under
`/autodl-fs/data/h3_experiments/comfyui_current_fullpath_profile_20260928/`;
the plan trace and screens are in adjacent `comfyui_current_plan_trace_20260928`
and `comfyui_plan_*_20260928` directories.

## Follow-up data-movement screens

The fused-root plan already computes the full-token projection and root
scores in the same CUDA kernel. The Q and K plans run on overlapping streams.
The plan trace attributes only about 0.20ms to global-index additions and
0.28ms to plain copies in one warmed plan call. Removing those alone cannot
close the roughly 6ms-per-sparse-call gap implied by the 1.05× full-denoise
target; the larger categories are node splitting, projection, selection,
scoring and key compaction.

Two adjacent data-movement hypotheses were screened, with no production
change:

- Doubling Spark QKV producer chunks from 16,384 to 32,768 reduced projection
  calls from 4,136 to 2,568, but on a simultaneous GPU0/1 one-prompt 10s
  A/B the profiled sampler took 244.40s versus 246.82s. The producer itself
  rose from 21.59s to 21.98s. The production default remains 16,384;
  an optional environment override is retained only for reproducibility.
- Rounding the plan's internal BF16 feature table to FP8 halved its bytes per
  element, but on the same captured Q/K the warmed plan median rose from
  15.74ms to 22.91ms. Q/K permutations changed at 97.58%/97.30% of
  positions; this is neither a faster nor a numerically faithful candidate.

Simple output rearrangement also failed a one-layer bandwidth screen:
`index_select` took 1.444ms and a contiguous-segment `cat` 1.465ms. Moving
the output projection before restoration took 14.202ms versus 13.984ms for
the current order in the same synthetic-shape GEMM screen. These are local
screens, not full-video measurements. Their artifacts are under
`comfyui_spark_producer_chunk_ab_20260928` and
`comfyui_plan_fp8_feature_table_20260928`.

The current exact plan does not appear to contain a removable duplicate
full-token projection. A material further gain would require a tested fused
route/selection or output-staging kernel that removes a large intermediate
read/write; changing block selection, plan reuse, or precision would be a
separate numerical ablation, not an implementation-only optimization.

Code audit after these screens sharpened that limit: with the production
`midpoint` landmark mode, fused-node construction reads only the chosen
landmark rows before its scoring pass, not the entire node twice. Subsequent
hierarchy levels must score all tokens again because their exact-capacity
parent assignment is known only after the preceding level's global cutoff.
The large repeated reads are therefore mainly *between necessary tree
levels*, not an accidental duplicate loop that can simply be deleted.
Avoiding them without changing the route would require cross-level fusion
with globally synchronized cutoffs or a different on-chip staging strategy;
neither has yet been implemented or benchmarked.
