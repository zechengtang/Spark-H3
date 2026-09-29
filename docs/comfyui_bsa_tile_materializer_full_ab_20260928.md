# BSA tile producer to Spark: full ComfyUI A/B (2026-09-28)

## What ran

The experimental `H3_SPARK_BSA_MATERIALIZER=1` path adds
`bsa_materialize_qkv_kernel` in comfy-kitchen's
`sage_attention/sol_attn_producer.cu`. It takes each chunk from **one** H3
QKV projection, uses the official BSA tile helpers `stage_tile64` and
`norm_rope_rows`, and writes full BF16 Q/K/V. The Spark path then runs its
existing complete reblock plan, current-step global reweight, TopK10 route,
block tail, exact kernel, and direct output scatter. `H3_SPARK_BSA_MATERIALIZER`
defaults to `0`; no production default changed.

This is a **BSA tile materializer adaptation**, not a run of the original
`sol_producer_kernel` INT8 carrier path or `sol_attn_chunked`/official BSA
route and exact kernels. Original BSA carriers depend on previous-step K/V
statistics, whereas Spark consumes current-step statistics and requires full
BF16 Q/K/V for reblock/reweight.

## Same-input numerical checks

GPU2 executed a real ComfyUI 10s/768p five-step run with the experimental
path. On the first actual sparse layer, both materializers consumed the
**same projected QKV chunk** of 16,384 tokens, 56 heads and rotation dim 96.
The candidate Q/K matched the baseline in 99.999666%/99.999690% of BF16
elements; maximum absolute differences were 0.0625/0.03125. V was bitwise
equal. The candidate materializer kernel took 1.002 ms versus 1.122 ms for
baseline RMSNorm/RoPE plus V copy on that chunk. The latter comparison does
not include candidate packed-RoPE preparation, projection, or Spark core.
Raw: `/autodl-fs/data/h3_experiments/comfyui_bsa_materializer_smoke_20260928/real_qkv_parity.json`.

An additional GPU2 replay tested all five chunk offsets using captured real
QKV values (last chunk 8,029 tokens). Q/K equality remained ~99.9988–99.9990%
with maximum absolute difference ≤0.03125; V remained bitwise. The offset
and ragged-tail checks passed. This replay applies valid generated RoPE and
norm weights to the captured post-RoPE QKV values, so it is an indexing and
kernel-parity check, not a substitute for the real ComfyUI check above.
Raw: `/autodl-fs/data/h3_experiments/comfyui_bsa_spark_reuse_ceiling_20260928/allchunks_gpu2.json`.

## Full 20-step paired A/B

Archived prompt 1, seed 42, 10s/1344×768, 20 actual ComfyUI model evaluations
with an excluded five-step warmup per arm. Both arms use TopK10, full reblock,
global reweight, block-granularity tail, FP32 anchor, and direct output scatter.
Each arm had its own fresh ComfyUI process. GPU2 ran candidate→baseline;
GPU3 ran baseline→candidate. The runner checked identical source/binary hashes
before and after each run and verified the candidate branch marker. All four
arms used the same hashes.

| GPU | Baseline Spark | BSA tile materializer + Spark | Candidate saving |
| --- | ---: | ---: | ---: |
| 2 | 245.499 s | 244.335 s | 1.165 s |
| 3 | 242.514 s | 242.134 s | 0.380 s |

Mean paired saving: **0.772 s** per generation, about **0.32%** of the
baseline mean. Both GPU comparisons point in the same direction, but their
different magnitudes and one run per arm do not establish a stable speedup.
This is far short of the roughly 4.5 s saving needed for the then-current
1.05× Sol/Spark target.

The baseline latent SHA256 was identical across GPU2/3, and the candidate
latent SHA256 was also identical across GPU2/3; the candidate and baseline
hashes differed. For the saved video latent, the two modes differed by mean
absolute 0.1412 and maximum absolute 4.1624; for audio, 0.0170 and 0.3039.
The run is deterministic per mode across GPUs, and chunk checks found no
large offset error. The output difference may reflect propagation of small
BF16 rounding differences through 20 denoising steps, but no video-quality
metric was measured; it must not be interpreted as a quality verdict.

Artifacts: `/autodl-fs/data/h3_experiments/comfyui_bsa_spark_full_ab_20260928/gpu{2,3}/{baseline,candidate}/`
contain protocols, logs, latent files and records; GPU2 summary is at
`gpu2/summary.json`. Runner: `scripts/comfyui_bsa_spark_full_ab_20260928.py`
(GPU2 used the same imported runner with GPU/port overridden in its own
process). CUDA extension SHA256:
`2738e05796a04f53514da8a86ecb97a0bdfbc0c81c59225a71599df568cb206b`.
Plugin `comfyui_nodes.py` SHA256:
`8c1cbd95ea1160fa2f97af3b2606f0d2f06397ccec28e348d7e4a37d4eef268d`.

**Decision:** retain the experimental switch as opt-in and leave the default
Spark producer unchanged. The measured whole-run gain is small and BF16/latent
outputs are not bitwise equal.
