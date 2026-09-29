# ComfyUI global reweight optimization experiments — 2026-09-25

2026-09-26 update: [matched-granularity incremental measurements](comfyui_reweight_granularity_increment_20260926.md)
separate KV reweight overhead from query-tail evaluation. Historical exact-only
increments below must not be interpreted as that matched marginal overhead.

Baseline interpretation correction: see [Sol tail granularity](comfyui_sol_tail_granularity.md).
The local ComfyUI and frozen Diffusers Sol implementations use different
tail granularity. Their historical reweight increments do not measure the
same added work; the tables below remain experiment records, not proof of
an equivalent-work efficiency gap or a scheduling-only bottleneck.

## Diffusers-inspired follow-up (v17–v19): no accepted speedup

All three candidate changes below were rejected and the pre-round CUDA
sources and binary restored byte-for-byte. The earlier uniform-prefill
optimization remains active. Invalid reweight increments and target-failure
claims based on the exact-only baseline were deleted on 2026-09-26.
The raw full-call timings below remain valid historical measurements.

Two independent CUDA implementations were tested, not imported from Diffusers:

- v17 uses BF16 WMMA for weighted K/V, stages source K/V once, and rounds the
  unnormalized softmax weights to BF16 as in the Diffusers dot formulation.
  FP32 anchor scores and the existing unrounded-key log-mass definition remain.
- v18 distributes the 64 source rows over four warps, caches K in registers,
  and reduces four partial weighted K/V vectors. It retains FP32 weights and
  changes only floating-point reduction order, not the reweight formula.

Same real 10s/768p capture, TopK10, fanout16 and FP32 anchor. Each GPU ran two
baseline and two candidate passes, in opposite ABBA/BAAB orders on GPU0/1;
each pass has four warmup calls and twelve CUDA-event samples. Values below
average the four per-pass medians for each mode. Plan time is excluded.

| Experiment | Native exact-only TopK10 (ms) | Spark, no reblock (ms) | Spark, fused reblock (ms) |
| --- | ---: | ---: | ---: |
| v17 control | 44.2672 | 50.8136 | 51.0756 |
| v17 BF16 WMMA summary | 44.2442 | 51.1045 | 51.6645 |
| v18 control | 44.2201 | 50.8339 | 51.0861 |
| v18 register-cached summary | 44.2259 | 50.9354 | 51.3248 |

Neither summary candidate is a latency win. Profiling one warmed invocation
on each GPU gives summary-kernel means of 1.598 -> 1.993 ms for v17 and
1.602 -> 1.814 ms for v18. Summary and routing execute on different streams;
these component durations cannot simply be added to attention wall time.

Compiler resources: original summary 36 registers/1804 shared bytes; WMMA
78 registers/36640 shared bytes; register-cached 108 registers/5900 shared
bytes. All have zero stack. These establish increased resource requirements,
but are not runtime occupancy/stall-counter measurements. The WMMA prototype
uses a 16-row tile for a single useful anchor row; it is not proof that every
possible Tensor Core implementation must be slower.

### Accuracy checks

- v17: 112/113 tests pass. The existing BF16-anchor versus FP32-anchor test
  fails its 0.9999 cosine threshold (0.9998896). Only the FP32-anchor branch
  uses the experimental builder, so that comparison changes both anchor and
  projection arithmetic. The threshold was NOT relaxed, and the candidate
  is not acceptable for deployment.
- v18: all 113 tests pass. Against the control, the full real-input attention
  output has relative L2 error 2.854e-5 without reblock and 2.934e-5 with
  reblock. The corresponding signal-to-error ratios are 90.89/90.65 dB.
- v17's real-input relative L2 errors are 2.307e-4/2.691e-4, with
  signal-to-error ratios 72.74/71.40 dB.

These are attention-tensor comparisons to the previous implementation, NOT
video PSNR against dense, and NOT VBench. No full-video tests are justified
by these failed speed gates. v17 and GPU1 v18 used FP32 metric reductions;
GPU0 v18 and subsequent probes use FP64 products/reductions, avoiding cosine
roundoff marginally above one for nearly identical tensors.

Artifacts: `summary_gemm_v17_gpu{0,1}.json`,
`summary_register_v18_gpu{0,1}.json`, and probe
`scripts/probe_comfy_summary_gemm_20260925.py`. The probe verifies the profiled
kernel specialization to reject an unsupported experimental mode rather than
silently benchmark the ordinary implementation.

A fresh, separate GPU1 replay of the unmodified frozen Diffusers snapshot
measured 94.726 ms for TopK10+reblock and 99.146 ms for global Spark: a 4.420 ms
increment. This single refresh (`diffusers_reweight_refresh_v19_gpu1.json`)
is a historical Diffusers measurement, not a matched comparison with the
ComfyUI exact-only diagnostic baseline.

### v19: summary-first attention with first-exact prefetch

The third candidate keeps the original summary constructor, processes virtual
summary tiles before exact tiles, and prefetches the first exact block into
the alternate shared buffer while computing the final summary tile. This
borrows Diffusers' idea of overlapping first-exact transport with approximate
attention, without copying its CuTe kernel or changing the TopK budget.
The accumulation order changes; the mathematical global-reweight algorithm
does not. Stock Sol dispatch remains on the control specialization.

Same two-GPU ABBA/BAAB protocol:

| v19 mode | Native exact-only TopK10 (ms) | Spark, no reblock (ms) | Spark, fused reblock (ms) |
| --- | ---: | ---: | ---: |
| Control | 44.1758 | 50.8687 | 51.0928 |
| Summary first + first-exact prefetch | 44.2397 | 52.4472 | 52.7218 |

All 113 tests pass. Real-input relative L2 errors against the control are
4.202e-5 / 4.110e-5 without/with fused reblock; signal-to-error ratios are
87.53 / 87.72 dB. Nonetheless, the latency regression rejects this candidate.
Compiler output keeps 168 registers and 33792 shared bytes, but stack usage
increases from 0 to 8 bytes. That is a code-generation observation, not proof
that stack traffic alone accounts for the regression.

The v19 profiler averages eight calls per mode, independently on each GPU.
Across the GPUs, the exact kernel is approximately 36.55 ms for native TopK,
41.98 ms for control Spark and 43.60 ms for summary-first Spark; summary
construction stays near 1.60 ms and selection changes from about 1.80 ms
(TopK) to 2.04 ms (Spark bitmap production). Profiling changes absolute
timings versus uninstrumented CUDA events, so the event table above is the
latency authority. These profiles locate most additional work in the fused
summary-attention consumer, not merely the summary builder. They do not
identify a hardware stall cause, and overlapping stages must not be added
as though serial.

Artifacts: `reweight_tail_first_v19_gpu{0,1}.json` and
`diffusers_inspired_v17_v19_rejected.patch`. The patch is relative to the
pre-round ComfyUI sources and retains all candidate modes for reproduction;
none of its experimental environment switches remain in the active backend.
The probe intentionally raises if these modes are requested without that
patch, rather than produce mislabeled baseline results.

The result does not establish that further optimization is impossible.
It does establish that a direct weighted-GEMM substitution, register caching,
and this particular Diffusers-inspired prefetch schedule did not improve
total Spark latency. No new end-to-end video or quality-metric run was
started, since the single-layer performance gate failed. Diffusers pipeline
and kernel sources were not modified.

After restoring the original two CUDA sources and extension binary, all
113 comfy-kitchen tests and 9 ComfyUI-plugin tests pass; the 48 saved edge
cases match bitwise. The real-capture Sol, Spark and fused-reblock Spark
output SHA256 hashes match the previously retained prefill binary exactly.
Final restored GPU1 replay (`reweight_v19_restored_gpu1.json`) measures
44.034 ms native TopK, 50.804 ms Spark and 51.089 ms fused-reblock Spark,
excluding the plan. This is restoration verification, not a newly achieved improvement.

---

Retained baseline: the earlier warp/pipeline follow-up below found and retained a
minimal prefill change. Binary A/B/A/B tests show about 3.7% faster Spark
attention and Sol, with matching real-input output hashes. These measurements
did not isolate the reweight increment. Earlier rejected rounds
remain documented below as historical results.

No significant Spark speedup was obtained in the initial v8–v10 round. The three candidate
kernel changes were reverted and the pre-round v6 source was rebuilt and
retested. No Diffusers kernel or pipeline was changed.

## Scope and measurement

- GPU0 and GPU1, separate processes, the same captured real 10s/768p QKV:
  `[1, 73565, 56, 128]`, 72576 video tokens.
- TopK10, fanout16, FP32 global anchor; CUDA-event timing after four warmup
  calls, twelve measured calls per variant. These are attention-call timings,
  not denoising-step or complete-video timings.
- Table entries average the two GPU-specific medians. Reblock plan is excluded
  from both Spark columns. Reblock data movement is fused in the second column.
- No new 5s test or end-to-end PSNR/VBench run was performed this round.

| Candidate | Spark without reblock (ms) | Spark with fused reblock (ms) | Decision |
| --- | ---: | ---: | --- |
| Previous v6 measurement | 52.924 | 53.250 | Starting implementation |
| v8: remove redundant per-row liveness reductions | 52.903 | 53.172 | No significant gain; reverted |
| v9: v8 plus L1-cached summary staging | 53.490 | 53.580 | Regression; reverted |
| v10: v8 plus compile-time Spark specialization and empty-state initialization | 53.416 | 53.562 | Regression; reverted |
| Restored v6, final verification | 53.002 | 53.313 | Active implementation |

The ordinary Sol branch became faster in v10, but Spark did not. That is not a
reweight optimization. The measurements do not establish the microarchitectural
cause of any regression.

## Historical exact-only diagnostic (not a reweight baseline)

The benchmark now includes `native_topk10_exact_only`. It calls the native Sol
entry with the same fixed-budget TopK selector, no pooled tail and no global
reweight. Calling the public Sol wrapper with `tail=False` would instead select
an external routing path, so it is not used for this comparison.

Final restored-v6 results:

| Operation | GPU0 (ms) | GPU1 (ms) |
| --- | ---: | ---: |
| Native exact-only TopK10 | 46.159 | 46.256 |
| Sol tau=1 | 75.429 | 75.698 |
| Sol TopK10 with pooled tail | 46.867 | 47.008 |
| Spark global, no reblock | 52.963 | 53.041 |
| Spark global, fused reblock | 53.172 | 53.455 |
| Reblock plan alone | 14.815 | 14.808 |

Do not subtract the exact-only baseline to report reweight overhead. It lacks
the ordinary approximate tail. Use the matched-granularity experiment linked above.

## GEMM and implementation comparison

The current ComfyUI exact kernel already contains Tensor Core INT8 QK and PV
for virtual summaries, sharing the query registers and online-softmax output
state with exact attention. A claim that ComfyUI lacks fused GEMM is inaccurate.
The summary-construction kernel still uses scalar FP32 FMA.

The frozen Diffusers baseline in
`/autodl-fs/data/h3_experiments/attn_kernel_speedup_20260921/snapshot/sol_attn/sm120/mainloop.py`
already performs per-query pooled QK/PV.
Its Spark mainloop replaces the pooled PV with weighted-summary PV and adds
the weighted-key QK. ComfyUI Sol's pooled approximation is shared across the
query block; Spark adds per-query summary QK/PV. Consequently, the measured
reweight increments include different work on the two sides. Fusion quality
alone cannot explain the difference, and matching the Diffusers incremental
percentage is not guaranteed by using GEMM.

The earlier group-interleaving prototype also failed to improve the measured
increment. Its performance does not by itself prove that pipeline restart or
synchronization is the cause; that requires more detailed profiling.

## Validation and limitations

All 113 tests in comfy-kitchen `tests/test_sol_attn.py` pass on the restored
binary. Added coverage includes 257/4097-token all-summary cases (partial
blocks and multiple summary groups), and fused versus explicit reblocking at
TopK10. The explicit global-summary reference was vectorized for these tests.

Nsight Compute was attempted but returned `ERR_NVGPUCTRPERM`. No hardware stall,
cache-hit or Tensor Core utilization conclusion can be drawn from that attempt.
`ncu_instrumented_not_latency.json` is instrumented output from this failed
attempt and MUST NOT be used as a latency baseline.

Artifacts are under
`/autodl-fs/data/h3_experiments/spark_cross_pipeline_profile_20260925/`:

- `comfy_reweight_guard_v8_gpu{0,1}.json`
- `comfy_reweight_l1_v9_gpu{0,1}.json`
- `comfy_reweight_specialized_v10_gpu{0,1}.json`
- `comfy_reweight_restored_v6_gpu{0,1}.json`
- `reweight_v6_ncu_spark.txt`

Further gains remain possible, but are unproven. The next substantial kernel
experiment should target the virtual QK/softmax/PV instruction schedule and
register lifetimes, or summary construction, rather than assume that moving
existing GEMMs between loops will remove their cost. Hardware counter access
would help discriminate these directions.

## Follow-up: unused-work removal, K reuse and QK scheduling

The requested three directions were implemented and tested on GPU0–1. None
gave a reproducible attention speedup. All production CUDA changes from this
follow-up were reverted to the saved pre-round sources and binary. The
experimental switches below therefore are NOT part of the active backend.

### v11: unused work

An internal process-scoped mask `COMFY_SPARK_PRUNE` independently enabled:

- 1: skip summary staging/scanning when the query block selects every KV block;
- 2: replace permanent sink-KV summary construction with initialized masked slots;
- 4: omit pooled-V sums and, for fixed TopK, variance/tau-threshold statistics.
  K centering, V maxima/scales, and tau-mode thresholds were retained.

Both GPUs ran the modes in opposite orders. Averages of per-GPU medians:

| Mask | No-reblock Spark (ms) | Fused-reblock Spark, excluding plan (ms) |
| --- | ---: | ---: |
| 0 | 53.325 | 53.634 |
| 1 | 53.347 | 53.642 |
| 2 | 53.403 | 53.685 |
| 4 | 53.321 | 53.556 |
| 7 | 53.324 | 53.619 |

No consistent material latency benefit was established. These measurements
include generated-code effects and concurrent summary/route execution; they
do not imply that the removed instructions or reads cost literally zero.

### v12: reuse K in shared memory

A templated summary variant staged each block's BF16/FP16 K once in shared
memory and reused it for anchor scores and weighted K. Global K reads were
reduced without changing the arithmetic order. It passed all 113 tests and
all 48 bitwise reference cases. However, profiled summary time increased from
about 1.572 ms to 1.759–1.765 ms. Mean attention medians across the GPUs were:

| Shared K cache | No reblock (ms) | Fused reblock (ms) |
| --- | ---: | ---: |
| Off | 53.126 | 53.440 |
| On | 53.204 | 53.590 |

This variant was rejected. Hardware counters remain unavailable, so increased
shared-memory occupancy cost is only a possible explanation, not a measured
cause.

### v13: consume QK fragments immediately

The virtual attention loop computed one INT32 QK fragment, immediately applied
its scales/bias/mask, and then computed the next fragment, instead of first
materializing every QK fragment. The arithmetic and output were unchanged.
The version included v11 mask 7, but excluded the failed shared-K cache.
An initial small apparent gain did not survive binary A/B/A verification:

| Binary/run | No-reblock Spark (ms) | Fused-reblock Spark (ms) |
| --- | ---: | ---: |
| Original A, before | 52.770 | 53.026 |
| Candidate B | 52.961 | 53.190 |
| Original A, after | 52.906 | 53.237 |

The candidate is 0.124 ms slower than the mean of the two original runs for
no-reblock, and 0.058 ms slower with fused reblock. These small differences do
not establish a speedup. Restore, rather than deploy, was the decision.

### Reproducibility and validation

`scripts/check_comfy_prune_20260925.py` saves original-binary outputs and compares
candidate outputs bitwise. It covers 48 combinations of BF16/FP16, 257/4097
tokens, partial video blocks, tail/interior/all sink ranges, TopK/tau/all-exact/
all-summary routing, with and without fused reblocking. Each of v11 masks
0/1/2/4/7 and v12/v13 matched the original outputs in these cases. These are
kernel tests, not end-to-end PSNR/VBench measurements.

Additional artifacts in the experiment directory:

- `prune_v11_flags{0,1,2,4,7}_gpu{0,1}.json`
- `cache_v12_mode{0,1}_gpu{0,1}.json`
- `fragment_v13_gpu{0,1}.json`
- `ab_baseline_a_gpu{0,1}.json`, `ab_candidate_b_gpu{0,1}.json`,
  `ab_baseline_c_gpu{0,1}.json`
- `candidate_v13_reverted.patch`: the rejected changes relative to the
  pre-round source, retained for inspection, not enabled.

No 5s/full-video experiment was launched because the single-layer speed gate
was not met. Diffusers pipeline and kernels were not modified.

## Warp/pipeline follow-up: retained uniform-prefill change

Hardware: GPU0–1, NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120.
Compiler: CUDA 13.0, nvcc V13.0.88. Same real 10s/768p capture and timing method
as above. No model, route, TopK ratio, fanout, or attention formula was changed.

### Execution-configuration experiments

The experimental kernel split each 64-query route into either one 4-warp CTA
or two 2-warp CTAs, reusing the same route list. Compile-time launch bounds
varied register budgets. This affects BOTH exact and virtual attention, so
these are fused-kernel experiments, not isolated reweight-GEMM measurements.

First build (v14), average GPU-specific median attention time without reblock:

| Mode | Warps / CTA | Requested minimum CTAs / SM | Registers | Time (ms) |
| --- | ---: | ---: | ---: | ---: |
| 0, original configuration | 4 | 3 | 168 | 52.933 |
| 1 | 4 | 4 | 128 | 69.570 |
| 2 | 4 | 2 | 219 | 54.365 |
| 3 | 2 | 3 | 229 | 67.409 |
| 4 | 2 | 6 | 168 | 78.339 |

Launch bounds guide compilation; requested minimum CTAs are NOT a measured
resident-CTA count. Shared memory can impose a lower residency limit.

Second build (v15) also parameterized pipeline depth and generalized virtual
prefill into an unrolled loop, with an unconditional commit for each prefill
slot and a conditional data load. Its default configuration unexpectedly
compiled without the original kernel's local-memory stack accesses:

| Mode | Warps | Buffers | Requested min CTAs | Time (ms) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4 | 2 | 3 | 50.957 |
| 5 | 4 | 1 | 3 | 53.739 |
| 6 | 4 | 1 | 4 | 63.801 |
| 7 | 4 | 3 | 2 | 54.419 |
| 8 | 2 | 1 | 3 | 70.520 |

Changing warp count, register budget or buffer depth did not beat the default
4-warp/two-buffer configuration. All tested configurations matched the 48
reference outputs bitwise. The experimental modes and environment dispatch
were removed from production.

### Minimal retained change and machine-code evidence

Only virtual prefill initialization in comfy-kitchen
`comfy_kitchen/backends/cuda/sage_attention/sol_attn_exact.cu` changed:
replace the guarded first-tile load/commit with an unrolled `NSTAGE - 1` loop
that conditionally loads each tile and uniformly commits each prefill slot.
At the production depth of two, this maintains the same data-loading order
for valid nonempty attention inputs.

For both BF16 and FP16 kernels, static resources remain 168 registers and
33792 bytes shared memory, while stack size drops from 16 bytes to zero.
Original SASS has three STL-family and five LDL-family instruction sites;
the minimal new kernel has none. This is concrete code-generation evidence,
not a hardware-counter measurement or proof that every millisecond saved is
caused exclusively by those instructions. Speed may depend on compiler/GPU.

### Counterbalanced binary A/B/A/B verification

Original and minimal-new binaries were alternated on both GPUs, with no
overlapping processes during binary replacement. Each table entry averages
four GPU/run medians (two runs per GPU), each from 12 warmed measurements.

| Operation | Original (ms) | Retained change (ms) | Reduction |
| --- | ---: | ---: | ---: |
| Native TopK10, exact only | 46.079 | 44.068 | 4.36% |
| Sol tau=1 | 75.295 | 72.510 | 3.70% |
| Sol TopK10 with pooled tail | 46.786 | 44.957 | 3.91% |
| Spark global, no reblock | 52.768 | 50.839 | 3.66% |
| Spark global, fused reblock, excluding plan | 53.033 | 51.085 | 3.67% |
| Reblock plan | 14.802 | 14.820 | Essentially unchanged |

This is a shared attention-kernel improvement, not a measurement of the
isolated reweight increment. No full-video speedup is claimed.

### Validation and artifacts

- 113 comfy-kitchen Sol/Spark tests pass on the minimal retained binary.
- 9 MiniMax-H3 ComfyUI plugin tests pass.
- All 48 reference cases match the original binary bitwise.
- Full output SHA256 hashes for real 10s/768p Sol, no-reblock Spark and
  fused-reblock Spark match the original on BOTH GPUs.
- No new full-video PSNR/VBench or 5s test was run; unchanged hashes apply to
  this captured attention input, not a claimed full-video evaluation.
- Diffusers code is unchanged. The active CUDA binary contains the minimal
  prefill improvement, not the experimental tuning dispatch.

Artifacts under the same experiment directory:

- `warp_v14_mode{0,1,2,3,4}_gpu{0,1}.json`
- `pipe_v15_mode{0,5,6,7,8}_gpu{0,1}.json`
- `prefill_ab_original_{a,c}_gpu{0,1}.json`
- `prefill_ab_new_{b,d}_gpu{0,1}.json`
- `warp_pipeline_v15_resources.txt`
- `warp_pipeline_experiment_only.patch` (rejected tuning framework, archived
  relative to the pre-round source; not part of the deployed implementation)

The profiler script now supports `--fingerprint`, which hashes selected Comfy
outputs AFTER timing to validate binary comparisons without timing CPU copies.
