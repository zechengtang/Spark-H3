# Spark route-mode quality study: 25 prompts at 768p

## Resumed four-GPU follow-up queue (2026-10-04)

The user-required sequence is now a persistent background queue, not a manual
handoff after each experiment. The coordinator is
`scripts/run_three_task_queue_4gpu_20261004.py`. Live status and the effective
scoring requirements are recorded at
`/autodl-fs/data/h3_experiments/three_task_queue_4gpu_20261004/{status,protocol}.json`.
The original route runner's `vbench: not included/not_requested` fields predate
the user's additional scoring request; the queue protocol supersedes that scope.
Do not call the whole task complete until its required VBench stage finishes.

1. Finish the missing cases 26--50 for `fused_threshold` and
   `fused_packed_external_no_route_qk`, at 5s before 10s: 100 new videos total.
   Combine with the verified historical halves/legacy arms into four arms of
   50 prompts per duration. Complete PSNR/SSIM/LPIPS and source-assigned
   core-five VBench, including Dense references.
2. Run experiment-local 8fps Ref2VA Dense A/B/C at both durations, then C Spark
   TopK-10% with target-and-condition scope. B changes only reference-video
   bias by +5/3 RoPE units and reuses A conditioning/noise. C uses zero-origin
   source frames 0,3,6,... with rebuilt conditioning and truthful 8fps metadata.
   Regenerate A because historical inference has no complete core-source hash
   record; preserve old outputs. Score PSNR/SSIM/LPIPS against Dense (plus the
   diagnostic A/B and A/C pairs), **no VBench**. Saved-latent lossless decodes,
   coordinate/noise audits, stream-time checks and contact sheets are retained.
3. Run 50 prompts at 5s then 10s with `sol_global_anchor_dtype="float32"`;
   change no other sparse-attention setting from BF16-anchor legacy/threshold.
   Keep the model BF16, four dense warmup evaluations and the first layer dense.
   Complete PSNR/SSIM/LPIPS and source-assigned core-five VBench. Reuse task1
   Dense/BF16 VBench only after exact video and metric-source hash validation.

VBench assignments per arm are Subject 14, Background 17, Motion 14,
Imaging 19 and Aesthetic 19; these are not 50 scores per dimension and are not
an official overall VBench aggregate. Historical timing baselines are reused,
so timing comparisons remain descriptive across runs rather than a newly
rerun same-session timing matrix. At queue setup, six model-free tests passed,
including strict stage ordering, anchor-only config changes, frame-index
sampling and audio-/video-dominant bias layout checks. These are preflight
checks, not completed generation or quality results.

### Queue recovery after task1 scoring

Both durations of task1 now have complete four-arm full50 RGB fidelity and
assigned-core-five VBench results (`415` assigned jobs per duration, including
Dense). The first Ref2VA launch rebuilt the corrected C conditioning but
produced no new inference latents: its baseline decoder lacked `soundfile`,
the old official helper ignored the load-slot environment and serialized
weight loads, and the conditioning coordinator retained about 76 GiB on GPU0.
The failed attempt and its old source/protocol snapshots are retained.

The resumed runner uses existing SciPy for float-WAV IO, puts conditioning in
an isolated subprocess, uses four experiment-local weight-load lock slots,
and deletes the compile plugin owner before loading its decoder. Transformer
weights use the normal mmap-enabled loader, without changing their BF16 dtype.
The coordinator reuses completed task1 scores after provenance checks.
These infrastructure-only repairs are recorded in both queue and Ref2VA
`protocol.json` runtime-repair entries, with before/after source hashes and
previous protocols preserved. Core-source hashes, sampling, coordinates,
conditioning and all experimental configurations are unchanged. Ten
model-free tests passed, including exact float-WAV roundtrip and BF16-tensor
audio conversion without the missing dependency. An actual decoder smoke test
then completed both saved-latent 24fps Dense references: 120/240 lossless RGB
frames, float WAV audio, MP4 previews and temporal contact sheets; stream start
and audio/video length checks passed. The audio preview exporter specifically
requires a Torch tensor, whereas the lossless archive and SciPy WAV writer use
NumPy arrays; that adapter is experiment-local. The precision gate now audits
all parameters against the model's official `_keep_in_fp32_modules` policy:
the actual Ref2VA model has 626 BF16 parameter tensors and 12 deliberate FP32
input/time/output-head parameter tensors. The initial first-parameter check
was wrong because `proj_in.weight` is one of those intentional FP32 tensors.
No dtype is changed by the revised gate, and a regression test rejects an
incorrectly FP32 block stack. The noise-audit hook reads newly written outputs
from `PipelineState`, not the step's input-only `BlockState`; a real CPU run of
both noise-preparation steps verifies that capturing the condition and target
noise preserves their outputs and random-generator draw sequence exactly.
Ref2VA generation/scoring and FP32-anchor results are
still pending; completed task1 is not a claim that all three tasks are done.

## Decision

Keep `landmark_tree_v2_midpoint_direction_mode="legacy"` as the production
default. Evaluate the packed external route optimization independently with
`sol_route_topk_execution="packed_external_no_route_qk"`.

The first matrix coupled the packed external execution path to the fused
midpoint-direction builder, so `fused_packed_external` is not a clean test of
the external route optimization. The follow-up arm is therefore
`legacy_packed_external_no_route_qk`.

## Legacy versus fused midpoint directions

> **Sampling-scope warning (2026-10-04):** The 25-prompt sections in this
> document use cases 1--25 of the `20pct` manifest, not the canonical `10pct`
> subset. Only 13 prompts overlap, and this first half contains no Imaging or
> Aesthetic source-suite cases. These results are historical screening
> diagnostics only: they must not determine relative quality or a production
> choice without completing all 50 prompts. The full-50 sections below have
> now supplied that confirmation and supersede the 25-prompt quality means,
> signs, win counts, confidence intervals, and rankings.

All rows use cases 1--25 from the 50-prompt `20pct` manifest, seed 42, 20
requested steps (19 transformer evaluations), 10% route Top-K, one dense
transformer layer, and a matched Dense video as the quality reference. This is
a fixed 25-prompt fidelity subset, not the Benchmark's official `10pct`
25-prompt subset: it contains only Subject, Motion, and Background assignments.
Do not report its partial assigned-dimension scores as a five-dimension VBench
result.

| Duration | Legacy PSNR | Fused PSNR | Fused − legacy | 95% paired CI | Paired p-value |
|---|---:|---:|---:|---:|---:|
| 5s / 120 frames | 22.8410 | 22.7274 | -0.1136 dB | [-0.5679, +0.3407] | 0.610 |
| 10s / 240 frames | 23.2249 | 23.0336 | -0.1913 dB | [-0.5316, +0.1490] | 0.257 |

Both durations have a small negative PSNR mean, but neither difference is
statistically significant. SSIM and LPIPS also show no consistent significant
regression. This supports a conservative engineering decision to retain legacy
directions; it does not establish a universal fused-quality regression. The
10s legacy scores were reused from
`topk_reblock_reweight_50prompt_20260920`, so the 5s same-snapshot comparison is
the cleaner attribution.

## Orthogonal follow-up

Compare:

- `legacy_threshold`
- `legacy_packed_external_no_route_qk`

Hold `landmark_tree_v2_midpoint_direction_mode="legacy"` fixed and change only
`sol_route_topk_execution`. Run 5s768p to completion before starting 10s768p.
Dense references are reused; generated sparse latents, decoded FFV1 videos,
per-prompt PSNR/SSIM/LPIPS, and CUDA-synchronized denoising time are recorded.

### 5s768p result

The 25-prompt run completed with all latents, decoded lossless archives, and
quality rows present:

| Method | Denoise | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| `legacy_threshold` | 140.089s | 22.8410 | 0.771951 | 0.166729 |
| `legacy_packed_external_no_route_qk` | 139.404s | 23.0232 | 0.773431 | 0.163982 |

The candidate is 0.49% faster. Relative to `legacy_threshold`, its paired mean
changes are +0.1822 dB PSNR (95% CI [-0.2378, +0.6022], p=0.379), +0.00148
SSIM, and -0.00275 LPIPS. None is statistically significant; 17/25 prompts
improved in PSNR. This result provides no evidence of a quality regression from
the packed external no-route-QK execution path when legacy midpoint directions
are held fixed.

### 10s768p result

The 25-prompt run also completed with all latents, decoded lossless archives,
and quality rows present:

| Method | Denoise | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| `legacy_threshold` | 341.615s | 23.2249 | 0.777540 | 0.145403 |
| `legacy_packed_external_no_route_qk` | 335.720s | 22.9965 | 0.772262 | 0.146628 |

The candidate reduces denoising time by 1.73% (paired mean -5.895s, 95% CI
[-6.757, -5.034], p=4.03e-13; faster on 25/25 prompts). Relative to
`legacy_threshold`, its paired
mean changes are -0.2284 dB PSNR (95% CI [-0.5856, +0.1287], p=0.199),
-0.00528 SSIM (p=0.214), and +0.00123 LPIPS (p=0.707). None is statistically
significant; 10/25 prompts improved in PSNR. Together with the 5s result, this
does not establish a quality loss caused by the packed external no-route-QK
execution path, but motivates the larger 5s/50-prompt follow-up below.

## Full 5s/50-prompt follow-up

The complete `20pct` 50-prompt manifest is now finished. Cases 1--25 reuse the
completed runs above; cases 26--50 were generated for Dense,
`legacy_threshold`, and `legacy_packed_external_no_route_qk`. All three
combined manifests contain exactly 50 hash-verified videos.

| Method | Denoise | PSNR vs Dense | SSIM vs Dense | LPIPS vs Dense |
|---|---:|---:|---:|---:|
| Dense | 198.087s | -- | -- | -- |
| `legacy_threshold` | 139.972s | 22.9559 | 0.791489 | 0.160872 |
| `legacy_packed_external_no_route_qk` | 139.212s | 23.0502 | 0.790600 | 0.159992 |

The candidate is 0.54% faster than threshold (paired mean -0.760s, 95% CI
[-1.037, -0.484], p=1.26e-6). Candidate-minus-threshold quality changes are
+0.0943 dB PSNR (95% CI [-0.1733, +0.3619], p=0.482; 27/50 wins), -0.000889
SSIM (p=0.710), and -0.000879 LPIPS (p=0.688). Thus the larger paired sample
detects the small speed improvement but provides no evidence of a fidelity
difference between the two legacy-direction execution paths.

VBench was evaluated only on each prompt's declared dimensions. Scores below
are percentage-point means over the assigned prompts, not an official overall
VBench aggregate.

| Method | Subject (n=14) | Background (n=17) | Motion (n=14) | Imaging (n=19) | Aesthetic (n=19) |
|---|---:|---:|---:|---:|---:|
| Dense | 94.0476 | 95.0555 | 98.9033 | 72.1673 | 68.9392 |
| `legacy_threshold` | 94.0188 | 94.8104 | 98.9050 | 71.6157 | 68.9250 |
| `legacy_packed_external_no_route_qk` | 93.9297 | 94.8039 | 98.8874 | 71.5878 | 68.9503 |

Candidate-minus-threshold paired deltas are -0.0891, -0.0065, -0.0176,
-0.0280, and +0.0253 percentage points respectively. All five paired tests
are nonsignificant (p=0.598, 0.975, 0.284, 0.854, and 0.914). The complete
50-prompt evidence therefore supports retaining legacy midpoint directions and
using `packed_external_no_route_qk` as the slightly faster external execution
implementation without a detected quality penalty.

## Full 10s/50-prompt follow-up

The complete `20pct` 50-prompt manifest is now finished at 10s768p. Candidate
cases 1--25 reuse the completed legacy external-no-route-QK run above; cases
26--50 were newly generated with the same configuration. Dense references and
all 50 legacy-threshold quality/timing records are historical reused results.
The combined candidate manifest contains exactly cases 1--50, with prompt
identities, video hashes, and Dense reference hashes verified for every case.
This is not a newly rerun, same-snapshot two-arm timing comparison.

| Method | Denoise | PSNR vs Dense | SSIM vs Dense | LPIPS vs Dense |
|---|---:|---:|---:|---:|
| `legacy_threshold` | 342.076s | 23.3006 | 0.795121 | 0.139424 |
| `legacy_packed_external_no_route_qk` | 336.806s | 23.1831 | 0.793009 | 0.139406 |

The candidate reduces recorded denoising time by 1.54% (paired mean -5.271s,
95% CI [-6.116, -4.425], p=6.86e-17; faster on 47/50 prompts). Model loading,
discarded warmup, video decoding, and quality scoring are excluded from these
denoising measurements.

Candidate-minus-threshold paired quality statistics are:

| Metric | Mean delta | 95% paired CI | Paired p-value |
|---|---:|---:|---:|
| PSNR | -0.1175 dB | [-0.3284, +0.0934] | 0.268 |
| SSIM | -0.002112 | [-0.007524, +0.003300] | 0.437 |
| LPIPS | -0.000018 | [-0.003962, +0.003926] | 0.993 |

PSNR improved on 20/50 prompts. Its negative mean is smaller than on the first
25 prompts (-0.2284 dB); SSIM's negative mean also shrinks, and LPIPS is nearly
unchanged. None of the three paired differences is statistically significant.
There is no detected significant fidelity regression, but this does not prove
equivalence or establish that all observed differences are noise. The routing
paths are not guaranteed to select identical exact-block masks, so any causal
quality or speed claim still needs that semantic distinction and the historical
baseline caveat. VBench was not requested or run for this 10s extension.

Artifacts:

- Results: `/autodl-fs/data/h3_experiments/legacy_external_no_qk_full50_20261004_10s768p/results.json`, under `full50`.
- Combined manifest: `/autodl-fs/data/h3_outputs/legacy_external_no_qk_full50_20261004_10s768p/full50/legacy_packed_external_no_route_qk/generation_manifest.json`.
- Runner: `scripts/run_legacy_external_no_qk_full50_10s768p_20261004.py`.

The top-level `runtime` and `quality` entries in this extension's results file
describe newly generated cases 26--50 only. Use `full50` for the complete
50-prompt comparison; do not mix these two aggregation scopes.
