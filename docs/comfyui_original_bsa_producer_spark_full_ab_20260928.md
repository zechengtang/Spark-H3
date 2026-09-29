# Original BSA producer plus Spark: full ComfyUI A/B (2026-09-28)

This experiment tested the **original `sol_producer_kernel`** as a direct
producer for Spark-H3, under the opt-in `H3_SPARK_BSA_ORIGINAL_PRODUCER=1`.
The flag defaults to off. Each chunk undergoes one QKV projection, then the
original kernel performs its INT8 carrier/statistics writes **and** an added
BF16 Q/K/V write from the same 64-row tiles. Spark then runs its unchanged
full reblock plan, current-step global reweight, TopK10 route, block tail,
exact kernel, and direct output scatter. The original producer's INT8
carriers are not consumed: they used zero K mean/unit V scale because the
current-step statistics and dynamic reblock permutation are unavailable when
it runs. Spark therefore recomputes its own carriers. This measures the
cost of that direct graft, not a fused end-to-end BSA/Spark core.

## Execution and numerical checks

The experimental API is `sol_producer_chunk_materialize`. It calls
`sol_producer_begin`, then the original `launch_sol_producer`/CUDA
`sol_producer_kernel` with optional BF16 output pointers. A GPU2 CUDA
profiler recorded the kernel by that name at 1.392 ms for one 16K chunk,
confirming the original producer ran. Including workspace initialization,
the producer took 1.436 ms median versus 1.159 ms for the current
RMSNorm/RoPE plus V copy, a **0.277 ms chunk regression** before full-path
overhead. The original BSA workspace is 396.8 MB for 16K tokens and
1,897,297,248 bytes for the 73,573-token production attention input.
Raw profiler/numerical A/B:
`/autodl-fs/data/h3_experiments/comfyui_bsa_original_producer_20260928/chunk_gpu2.json`.

A real GPU2 ComfyUI five-step smoke exercised the opt-in branch. On the
first sparse layer, both materializers consumed the **same projected QKV**
chunk (16,384 tokens, 56 heads, rot_dim 96). Candidate Q/K matched current
BF16 output in 99.999666%/99.999690% of elements, with maximum absolute
differences 0.0625/0.03125; V was bitwise equal. Raw parity and workspace:
`/autodl-fs/data/h3_experiments/comfyui_bsa_original_producer_smoke_20260928/real_qkv_parity.json`.
The two targeted `sol_attn_chunked` tests passed after the optional-output
change, as did all 15 ComfyUI plugin tests.

## Full 20-step paired A/B

Archived prompt 1, seed 42, 10s/1344×768, 20 actual ComfyUI model
evaluations, with an excluded five-step warmup per arm. Both arms used the
same TopK10, full reblock, global reweight, block tail, FP32 anchor and
default output scatter. Each arm had a fresh ComfyUI server. GPU2 ran
candidate→baseline and GPU3 ran baseline→candidate. The runner verified
source/binary hashes before and after each arm; all four matched. Candidate
logs contained the opt-in marker.

| GPU | Baseline Spark | Original BSA producer + Spark | Candidate regression |
| --- | ---: | ---: | ---: |
| 2 | 245.509 s | 245.803 s | 0.294 s |
| 3 | 242.272 s | 243.385 s | 1.112 s |

The paired mean regression is **0.703 s** per generation (~0.29% of the
baseline mean). The direction is consistent on both GPUs, though its size
varies. The direct graft therefore does not accelerate Spark on this test.

Baseline latent SHA256 matched across GPUs, and candidate latent SHA256
matched across GPUs. Candidate output differs from baseline, while matching
the previous BSA-tile-only candidate's latent hash. On the saved video latent,
the baseline/candidate mean absolute difference was 0.1412, maximum 4.1624;
on audio, 0.0170 and 0.3039. No decoded-video quality metric was measured.
The small per-chunk BF16 rounding differences may propagate through denoising;
these latent differences are not a quality judgment.

Artifacts:
`/autodl-fs/data/h3_experiments/comfyui_bsa_official_producer_spark_full_ab_20260928/`
contains per-arm protocol, server log, latent, record and GPU2 summary.
Runner: `scripts/comfyui_bsa_official_producer_spark_full_ab_20260928.py`
and its shared 20-step runner. Frozen CUDA binary SHA256:
`89053fd68fcc69d634eb6e135b799ce5e2c4a716ad1bdc12067d569f4337e305`;
`comfyui_nodes.py` SHA256:
`d43a162c8d71733851c9e586631cb1b6ed87a3e3ae2ecd2e6369e1ec4feb4ca6`.

**Decision:** keep this path opt-in and keep the current Spark producer as
default. This direct original-producer integration performs redundant
carrier/statistics work and regressed end-to-end timing on both GPUs.
