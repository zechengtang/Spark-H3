# Original BSA producer + full Spark-H3 sampler A/B (2026-09-28)

## Candidate actually tested

This second experiment invokes the official `sol_producer_kernel` after each
existing QKV projection. Its experimental extension additionally writes full
BF16 Q/K/V for Spark. The original producer still generates its INT8
carriers/statistics, using initialized dummy K-mean/V-scale inputs, but Spark
does not consume those carriers. Spark subsequently runs its current full
reblock plan, global reweight, TopK10 route, block-tail approximation, exact
attention, and direct output scatter. Thus the candidate pays for official
carrier generation **and** Spark's later preprocessing. It does not invoke
the official `sol_attn_chunked` route/exact core.

The candidate switch is `H3_SPARK_BSA_ORIGINAL_PRODUCER=1`, default off.
The GPU2 kernel profile at
`/autodl-fs/data/h3_experiments/comfyui_bsa_original_producer_20260928/chunk_gpu2.json`
records the actual CUDA kernel name `sol_producer_kernel`; a real ComfyUI
5-step smoke also logged the branch and checked its first 16K projected-QKV
chunk. Both experimental switches were explicitly unset for each baseline.
On that isolated 16K QKV chunk, the producer plus BF16 materialization took
`1.436 ms` median versus `1.159 ms` for current RMSNorm/RoPE plus V copy.
That `0.277 ms` chunk difference is a diagnostic, not the complete sampler
result. The original producer also needs about 397 MB of temporary workspace
for this input.

## Complete sampler measurement

All modes used separate ComfyUI processes with an excluded 5-step warmup and
one measured 20-evaluation sampler. They used archived prompt 1
(`vbench_all_0264`), seed 42, 10s/1344×768, `spark_block`, TopK10, full
reblock/reweight, block tail, and direct output enabled. The reported time
is ComfyUI `SamplerCustomAdvanced` time; conditioning, VAE, decode, and latent
save graph nodes are outside it. Any initialization that occurs inside the
sampler is included. GPU0 and GPU2 ran candidate→baseline; GPU3 ran
baseline→candidate.

| GPU | Order | Baseline | Original BSA producer candidate | Candidate extra time |
| --- | --- | ---: | ---: | ---: |
| 0 | candidate→baseline | 247.374 s | 248.678 s | **1.305 s** (0.527%) |
| 2 | candidate→baseline | 245.509 s | 245.803 s | **0.294 s** (0.120%) |
| 3 | baseline→candidate | 242.272 s | 243.385 s | **1.112 s** (0.459%) |
| Mean paired difference | | | | **0.904 s slower** (0.369%) |

The candidate was slower in all three paired runs. These are one prompt and
one measured run per mode per GPU, so the table is directional evidence rather
than a precise estimate of a subsecond effect. The before/after hashes of
each process, all six source hash maps, archived case, and graph settings
match. Both modes used CUDA extension SHA256
`89053fd68fcc69d634eb6e135b799ce5e2c4a716ad1bdc12067d569f4337e305`
and plugin SHA256
`d43a162c8d71733851c9e586631cb1b6ed87a3e3ae2ecd2e6369e1ec4feb4ca6`.
Each candidate server log contains the branch marker exactly once; each
baseline log contains it zero times.

All three baseline final latent SHA256 values equal
`f990568d94fb9d986170f753824d9aa3380f62559954ad45c8a372577563da10`.
All three candidate values equal
`f59b1a17fc3e5210d121d35061d99988dbdbe9f48d0ccffe609a6c11e2eabc52`.
The candidate differs from baseline. Video latent mean/max absolute error is
`0.14116`/`4.16244`; audio is `0.01701`/`0.30394`. The candidate latent SHA
matches the earlier BSA tile materializer candidate, but the two experiments
used different rebuilt binaries and should not be subtracted as a controlled
cross-experiment speed comparison.

**Decision:** leave this branch off by default. It neither improves full
sampler speed nor preserves bitwise output. This concrete result does not
support replacing Spark's producer with the original BSA producer while
retaining the present reblock/reweight path.

Runner: [`scripts/comfyui_bsa_official_producer_spark_full_ab_20260928.py`](../scripts/comfyui_bsa_official_producer_spark_full_ab_20260928.py).
Raw joint audit:
`/autodl-fs/data/h3_experiments/comfyui_bsa_official_producer_spark_full_ab_20260928/joint_summary.json`.
The earlier, distinct tile-helper trial is documented in
[`comfyui_bsa_tile_spark_full_ab_20260928.md`](comfyui_bsa_tile_spark_full_ab_20260928.md).
