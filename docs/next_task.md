# Next local inference tasks

Status: completed on 2026-10-03 after explicit approval to run.

Hardware used: four NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (SM120,
approximately 96 GiB each).

## High-priority follow-up: normalize warmup parameter semantics

Priority: high. This is an API and documentation correction, not a known
runtime-schedule correctness bug. The schedules exercised by the completed
experiments have the intended Dense/Sparse evaluation counts.

The current interfaces use three different counting conventions for the same
concept:

- Diffusers `H3SparseAttentionConfig` accepts `warmup_percent`, but computes it
  from requested sigma-grid points (`total_evaluations + 1`). For example,
  `num_inference_steps=9, warmup_percent=20` means two Dense evaluations out of
  eight actual transformer evaluations; `num_inference_steps=3,
  warmup_percent=33` means one Dense evaluation out of two.
- ComfyUI computes `warmup_ratio` from actual model evaluations and also
  exposes an explicit `warmup_steps` mode.
- FastVideo/FastH3 uses the explicit integer `dense_first_n_steps`, where
  "steps" are actual transformer evaluations.

Normalize the public terminology around **evaluations**, with an explicit
integer such as `dense_first_n_evaluations` (or `warmup_evaluations`) as the
canonical representation. Percentage/ratio inputs should use actual
transformer evaluations as their denominator. Preserve legacy
`warmup_percent` behavior through a documented compatibility path or migration
shim so existing configs and published workflows do not silently change their
effective schedules.

Acceptance requirements:

- Preserve the currently intended schedules during migration: the 3-point
  short runtime-warmup request remains one Dense plus one Sparse evaluation;
  the 9-point 8-evaluation LoRA schedule remains two Dense plus six Sparse;
  and the 20-point 19-evaluation baseline remains four Dense plus fifteen
  Sparse.
- Add boundary tests for zero, one, all, and non-integral ratio conversions,
  covering the 3→2, 5→4, 9→8, and 20→19 grid-point/evaluation cases.
- Update configuration serialization, CLI/UI labels, logs, examples, and
  ComfyUI workflow migration together. Do not use the ambiguous word `steps`
  where sigma-grid points and transformer evaluations can differ.
- Emit a clear warning or versioned migration for legacy percentage configs;
  never reinterpret an existing saved value silently.

## Completion artifacts

- Dynamic cache and warmup validation:
  `/autodl-fs/data/h3_experiments/spark_dynamic_text_warmup_paired_20261003`.
- Four-prompt speed matrix, kernel ablation, and cache A/B:
  `/autodl-fs/data/h3_experiments/pro6000_task1_matrix_20261003`.
- Ref2VA eight-output matrix:
  `/autodl-fs/data/h3_experiments/ref2va_768_480_dense_spark_20261003` and
  `/autodl-fs/data/h3_outputs/ref2va_768_480_dense_spark_20261003`.
- VDN 9-requested-step / 8-evaluation correction:
  `/autodl-fs/data/h3_experiments/vdn8_warmup_correction_20261003`.
- Published table updates: `docs/kernel_performance_matrix.md` and the two
  `Spark-H3-10pct (w/ warmup)` rows in `docs/blogs/spark-attn/README.md`.

## Execution order

1. Audit and tune the Spark compiled-kernel cache for variable-length text.
2. Run the four-way runtime-warmup study and select the formal warmup protocol.
3. Run the 4-prompt Dense-versus-Spark timing matrix and the independent
   10-second Spark kernel ablation.
4. Run the Ref2VA generation matrix.
5. Remeasure the 8-evaluation VDN comparison and correct the blog table.

Freeze and record the Git revision, dirty-diff hash, runner source, relevant
source hashes, model hashes, conditioning hashes, and hardware information
before each formal experiment. Do not mix results from different source hashes.

## 0. Variable-length text cache audit and tuning

### Motivation

The four T2VA prompts have different text-conditioning lengths. The packed DiT
sequence length therefore varies by prompt. Stock `sol_attn` conservatively
includes the token count in its compiled-kernel cache key. The Spark fused path
has a separate `_FUSED_COMPILED` cache whose current key is even stricter: it
contains every input tensor's exact shape, stride, and dtype. Although the CuTe
tensors use dynamic layouts, a different text length can miss this outer cache
and enter `cute.compile` again.

### Goal

For a fixed GPU architecture, video resolution/duration, dtype, and Spark
kernel configuration, reuse the compiled SM120 Spark kernel across different
text lengths. Preserve separate cache entries whenever a genuinely static ABI
or kernel choice differs.

### Required audit

- Instrument `cute.compile` calls, compilation wall time, and
  `_FUSED_COMPILED` size without including instrumentation overhead in formal
  timing.
- Check both the Spark CuTe cache and transformer `torch.compile` recompiles.
- Run prompt A followed by a different-length prompt B in one resident process.
- Determine whether the callable compiled from prompt A safely accepts prompt
  B's dynamic packed length.
- Keep token-dependent lightweight layout/topology caches separate unless their
  reuse is independently proved correct.

### Candidate change

Replace the exact tensor-size cache key with a canonical dynamic-signature key
that retains, as applicable: device, SM architecture, dtype, tensor rank,
layout/stride order and broadcast pattern, alignment constraints, route mode,
packed-route mode, local-block policy, Top-K policy, and all other compile-time
kernel choices. Do not merge legacy/fused or different route-execution kernels.

### Acceptance gates

- Prompt B does not trigger an expensive recompilation of the main Spark
  kernel after prompt A has compiled it.
- Old and new paths agree numerically within the existing kernel parity
  tolerance for at least two distinct text lengths.
- No out-of-bounds access, incorrect sink/context range, route corruption,
  non-finite output, or CUDA graph failure occurs.
- The compiled callable is reusable in both same-prompt and cross-prompt runs.
- Cross-prompt formal denoise time differs from the same-prompt control by no
  more than 2 seconds under the warmup study below.

If dynamic callable reuse is unsafe, evaluate a bounded/bucketed alternative
before accepting per-prompt recompilation as the fallback.

### Explicit cache-reuse speed A/B

After the four-way warmup study, quantify the benefit of the dynamic cache at
10 seconds/1344x768 in a single resident process.  Compile prompt A first, then
measure prompt B under both conditions, with two formal repeats and balanced
order:

1. reuse the prompt-A compiled Spark callable for prompt B;
2. clear only the Spark `_FUSED_COMPILED` callable cache immediately before
   prompt B, forcing the behavior equivalent to a variable-length cache miss.

Record CUDA-synchronized denoise wall time, `cute.compile` call count, measured
compile wall time, and denoise wall time after subtracting measured compilation.
Report both the user-visible compile-inclusive saving and the steady-state
execution difference.  Keep the model, prompt-B conditioning, seed, target
shape, Spark configuration, and `torch.compile` state identical; do not include
model loading or conditioning construction.

## 1. Runtime-warmup study

Use 10 seconds, 1344x768, seed 42, BF16, the first two prompts from the shared
4-prompt set, and the baseline `legacy + threshold` Spark-H3-10pct
configuration. Each formal timing is a complete 20-requested-step run with 19
actual DiT evaluations. Run two formal timings per condition and compare their
mean.

Test four conditions:

1. Run a discarded full 19-evaluation pass on prompt A, then time prompt A.
2. Run a discarded full 19-evaluation pass on prompt A, then time prompt B.
3. Run a discarded short pass on prompt A with `num_inference_steps=3` and
   `warmup_steps=1`, producing exactly one Dense evaluation followed by one
   Sparse evaluation, then time prompt A.
4. Run the same discarded short pass on prompt A, then time prompt B.

Record compilation-call counts and compilation time as well as CUDA-synchronized
DiT time. If both short-warmup conditions are within 2 seconds of their
corresponding full-warmup controls and show no first-use anomaly, use one Dense
+ one Sparse evaluation as the formal runtime warmup. Otherwise use the
`kernel_performance_matrix.md` protocol: four Dense evaluations plus the first
Sparse evaluation before each Spark measurement.

After cache tuning, the intended optimized policy is one selected warmup per
model, GPU, video shape/duration, and Spark kernel configuration in a resident
process, with reuse across prompt text lengths. The cross-prompt condition above
must prove that this is valid.

## 2. Four-prompt Dense-versus-Spark timing

Follow `docs/kernel_performance_matrix.md` unless the warmup study explicitly
selects the shorter validated warmup.

- Four shared prompts.
- Durations: 5 seconds, 10 seconds, and 14.4 seconds (345 frames for the long
  case).
- Resolution: 1344x768; BF16; seed 42.
- 20 requested inference steps and 19 actual DiT evaluations.
- Spark-H3-10pct with four schedule-Dense evaluations, one always-Dense layer,
  and the otherwise frozen T2VA configuration.
- Timing scope: CUDA-synchronized DiT/denoising only; exclude model loading,
  conditioning, VAE decode, saving, and runtime warmup.
- Report per-prompt raw times, mean Dense and Spark times, and speedup from
  unrounded means.

### Independent 10-second Spark kernel ablation

Use the same four prompts and formal timing configuration at 10 seconds and
compare the following four Spark arms against one matching Dense reference:

1. `legacy + threshold`
2. `fused + threshold`
3. `fused + packed_external`
4. `fused + packed_external_no_route_qk`

This is an independent ablation, not a 2x3 Cartesian product. The midpoint
ablation holds route execution at `threshold`; the route-execution ablation
holds midpoint construction at `fused`.

## 3. Ref2VA generation matrix

Keep the T2VA inference settings wherever applicable: target resolution 768p,
20 requested steps/19 DiT evaluations, seed 42, Dense versus Spark-H3-10pct,
and matching scheduler/precision settings. Reuse the same Ref2VA prompt,
reference video, and reference audio across durations. Reference audio
preprocessing remains unchanged.

Generate four Dense/Spark pairs, for eight outputs total:

1. Original 768p reference condition, 5-second target: Dense and Spark-H3-10pct.
2. Original 768p reference condition, 10-second target: Dense and Spark-H3-10pct.
3. Short-edge-480 reference condition, 5-second target: Dense and
   Spark-H3-10pct.
4. Short-edge-480 reference condition, 10-second target: Dense and
   Spark-H3-10pct.

For the 480p-reference arms, modify the actual Ref2VA preprocessing path rather
than merely pre-resizing the source while allowing it to be resized back. Add
shape assertions proving that both the visual LLM and reference-video VAE
receive the short-edge-480 representation. The generated target remains 768p.

Record denoise timing and produce the decoded audio-video outputs. Verify frame
count, resolution, duration, audio presence, finite latents, and the effective
reference tensors consumed by both conditioning paths.

## 4. Correct the VDN warmup row

Remeasure Dense versus Spark-H3-10pct with warmup at 14.4 seconds/345 frames,
1344x768, using `num_inference_steps=9`, which must execute exactly eight DiT
evaluations. The previous result requested eight steps and therefore executed
only seven evaluations.

- Cover the BF16 and FP8 rows used by the existing VDN table.
- Retain the intended two schedule-Dense warmup evaluations and one
  always-Dense transformer layer for the Spark-with-warmup row.
- Use the existing two evaluation prompts and one excluded runtime warmup per
  arm/precision unless a more specific frozen protocol is recorded before
  launch.
- Assert eight completed evaluations in every formal record.
- Report per-evaluation DiT and attention-module latency and speedup against the
  same-precision Dense arm.
- Update the Spark-H3-10pct `(w/ warmup)` BF16/FP8 values in
  `docs/blogs/spark-attn/README.md`; do not replace the no-warmup or VDN rows
  without a separate measurement.

## Completion checklist

- All runners are resume-safe and refuse source-hash drift.
- Warmups are explicitly marked discarded and excluded from aggregate timing.
- No first-use compilation appears in formal measurements.
- Actual evaluation counts are asserted, not inferred from requested steps.
- Results and human-readable reports include raw records and exact aggregation
  formulas.
- Only after all validations pass should documentation tables be updated.
