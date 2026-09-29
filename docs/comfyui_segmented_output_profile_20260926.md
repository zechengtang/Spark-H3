# ComfyUI Spark segmented output restore: isolated A/B

Completed 2026-09-26 on GPU4 (5s) and GPU5 (10s), while the separate
Diffusers component-ablation decoder was assigned only GPU0–3. Raw CUDA-event
samples: `/autodl-fs/data/h3_experiments/comfyui_segmented_output_profile_20260926/`.
Runner: `scripts/benchmark_comfy_segmented_output_20260926.py`.

The current inverse `output.index_select(1, inverse_permutation)` was compared
with `torch.cat((output[:, video_tokens:], output[:, :video_tokens]), dim=1)`
for the actual T2VA `[condition | video]` layout. Both reconstruct the same
natural-order BF16 tensor, verified bitwise on the same random output. The
test uses the real 5s/10s 768p token counts and 56×128 heads, 10 warmups,
8 excluded paired runs, and 64 alternating-order measured pairs per shape.

| Duration | Current inverse gather | Segmented `cat` | Candidate − current | Estimated 784-call change |
| --- | ---: | ---: | ---: | ---: |
| 5s 768p | 0.7493 ms | 0.7562 ms | +0.0070 ms | +0.0055 s |
| 10s 768p | 1.4465 ms | 1.4655 ms | +0.0190 ms | +0.0149 s |

These are isolated output-restore medians, not end-to-end denoise timings.
Their small differences do not justify replacing the current code with
segmented `cat`. Both approaches still copy the full attention output. The
larger potential optimization is writing the attention output or projection
result directly in the natural sequence order; that requires a kernel or
projection-epilogue change and was **not** tested here. No production layout
code was changed.
