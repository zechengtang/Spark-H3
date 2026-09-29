# Spark midpoint-direction speed and five-prompt replay (2026-09-27)

The September-20 Table 4 Spark code and current Spark were compared under the
same 10 s / 768p / seed 42 / 20-step grid (19 evaluations) protocol. The five
fixed prompt indices were 16, 32, 21, 8, and 5. The current Spark BF16,
threshold-route, query-tail, global-reweight configuration was changed in just
one place: in each worker process, `fused_midpoint_directions` was replaced by
the historical `indexed_interval_means` followed by
`build_cosine_directions`. The runtime source and historical results were not
modified. The four GPU workers exited successfully.

| Replay | Video latent bitwise equal | Audio latent bitwise equal |
| --- | ---: | ---: |
| Current fused midpoint directions | 0/5 | 0/5 |
| Current Spark with historical direction construction | 5/5 | 5/5 |

The historical-direction replay has zero maximum absolute error in both
latents for all five prompts. This confirms that the fused direction change
was sufficient to explain the five full-run mismatches in this sampled set;
it does not prove this for every prompt and setting.

The direct speed comparison used real captured case-05, evaluation-04,
layer-01 post-RoPE activations with all 56 heads and the production 72,576
video-token topology on an RTX PRO 6000 Blackwell Server Edition. Each of the
three direction-builder calls was warmed up five times and timed 20 times
with CUDA events. Summing per-call medians gives:

| Direction construction | Three-call total median time |
| --- | ---: |
| Current fused | 0.606 ms |
| Historical separate operations | 1.477 ms |

The fused function is **2.44× faster** for this narrowly measured operation,
saving about **0.871 ms per reblock plan**. This is not an end-to-end speedup
estimate: the complete denoising means for the same five prompts were
344.15 s with the current fused path and 344.03 s with the historical path.
Their 0.13 s difference is smaller than run-to-run/device variation, so no
measurable full-denoising improvement is established here. The gain is local
to a small part of the full attention pipeline, while the fused arithmetic
loses historical bitwise parity.

Following this result, the runtime setting
`landmark_tree_v2_midpoint_direction_mode` defaults to `"legacy"` in the
Diffusers Spark configuration and ComfyUI Spark configuration. Set it to
`"fused"` to opt into the newer direction kernel; the ComfyUI node exposes a
matching optional `midpoint_direction_mode` input. A production-shape
single-layer replay confirmed the default mode matches every recorded frozen
intermediate stage on affected heads 13, 51, and 54, while the explicit fused
mode takes the divergent route. This implementation check is separate from
the five-prompt full-run experiment above, which used an in-process function
replacement before the new configuration setting existed.

Artifacts:

- Current-path comparison: `/autodl-fs/data/h3_experiments/blog50_bitwise_repro_5prompt_20260927/results.json`
- Historical-direction replay: `/autodl-fs/data/h3_experiments/blog50_legacy_midpoint_repro_5prompt_20260927/results.json`
- All-head microbenchmark: `/autodl-fs/data/h3_experiments/blog50_midpoint_direction_speed_allheads_20260927/results.json`
- Runner and benchmark scripts: `../MiniMax-H3-Experiments/scripts/blog50_legacy_midpoint_repro_5prompt_20260927.py` and `../MiniMax-H3-Experiments/scripts/benchmark_midpoint_directions_real_20260927.py`
