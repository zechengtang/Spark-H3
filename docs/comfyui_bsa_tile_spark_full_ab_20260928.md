# ComfyUI BSA tile materializer + full Spark-H3 A/B (2026-09-28)

## What ran

This is a full ComfyUI sampler A/B, not a kernel-only projection estimate.
The experimental branch calls a new `bsa_materialize_qkv_kernel` in
`comfy-kitchen` after each existing QKV GEMM. It reuses the official BSA
`stage_tile64` and `norm_rope_rows` tile, RMSNorm, and RoPE helpers to write
full BF16 Q/K/V. The subsequent Spark path is unchanged: full reblock plan,
global reweight, TopK10 route, block-tail approximation, exact attention, and
direct output scatter. The original official `sol_producer_kernel` INT8
carriers and `sol_attn_chunked` route/exact core **were not called**. Thus the
candidate should be called the **BSA tile materializer adapted to Spark**, not
the full official BSA attention implementation.

The branch is gated by `H3_SPARK_BSA_MATERIALIZER=1`, default off. The two
modes ran in separate ComfyUI processes with the same rebuilt CUDA extension.
Each process ran an excluded 5-step warmup, then one 20-evaluation 10s/768p
sampler on archived prompt 1 (`vbench_all_0264`), seed 42. The graph used
`spark_block`, 1344×768, TopK 10%, `ablation_mode=full`, block tail,
`global_anchor_dtype=float32`, and `H3_SPARK_DIRECT_OUTPUT=1`. The time is
ComfyUI `SamplerCustomAdvanced` time; graph nodes for conditioning, VAE,
decode, and latent save are excluded, while any model initialization inside
the sampler is included. GPU2 used candidate→baseline order;
GPU3 used baseline→candidate order.

## Measured result

| GPU | Order | Baseline sampler | BSA tile candidate | Saved | Relative saving |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 | candidate→baseline | 245.499 s | 244.335 s | 1.165 s | 0.474% |
| 3 | baseline→candidate | 242.514 s | 242.134 s | 0.380 s | 0.157% |
| Mean of paired differences | | | | **0.772 s** | **0.316%** |

Both candidate server logs contain the
`H3_SPARK_BSA_MATERIALIZER active` marker exactly once; both baseline logs
contain it zero times. All four runs recorded the same plugin and CUDA binary
SHA256 before and after the sampler. The binary SHA256 is
`2738e05796a04f53514da8a86ecb97a0bdfbc0c81c59225a71599df568cb206b`;
`comfyui_nodes.py` is
`8c1cbd95ea1160fa2f97af3b2606f0d2f06397ccec28e348d7e4a37d4eef268d`.
The joint result checks that the full source hash maps, graph settings, and
archived case are identical across all four runs.

Baseline final latent SHA256 is identical across GPUs:
`f990568d94fb9d986170f753824d9aa3380f62559954ad45c8a372577563da10`.
Candidate final latent SHA256 is also identical across GPUs:
`f59b1a17fc3e5210d121d35061d99988dbdbe9f48d0ccffe609a6c11e2eabc52`.
The candidate differs from baseline. Video latent mean/max absolute error is
`0.14116`/`4.16244`; audio is `0.01701`/`0.30394`. A sampled real QKV
check found V bitwise equal and Q/K about 99.999% equal at the chunk output,
including the final ragged chunk, but tiny Q/K rounding differences propagated
through the 20 evaluations. No video quality comparison was performed.

This shows a small measured full-path saving in both orderings, but only one
prompt and one measured run per mode per GPU. The 0.38–1.16 s savings are close
to ordinary full-sampler variation, and the final output is not numerically
equivalent. **Keep the experimental branch off by default.** It does not
substantiate a larger Spark-specific speedup or justify replacing the current
producer.

An earlier GPU3 baseline was stopped while the extension was relinked in
place. It is preserved as `baseline_invalid_interrupted` and excluded from
every result above. The formal four runs used one frozen binary.

Runner: [`scripts/comfyui_bsa_spark_full_ab_20260928.py`](../scripts/comfyui_bsa_spark_full_ab_20260928.py).
Raw records and SHA audit:
`/autodl-fs/data/h3_experiments/comfyui_bsa_spark_full_ab_20260928/joint_summary.json`
and adjacent `gpu2`/`gpu3` directories. The real QKV smoke record is
`/autodl-fs/data/h3_experiments/comfyui_bsa_materializer_smoke_20260928/real_qkv_parity.json`.
