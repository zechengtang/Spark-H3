# Original BSA producer + Spark full sampler A/B protocol

This is a second experiment, separate from the BSA tile BF16 materializer
trial in `comfyui_bsa_tile_spark_full_ab_20260928.md`.

## Candidate

Enable `H3_SPARK_BSA_ORIGINAL_PRODUCER=1` only for the candidate. The adapted
producer must invoke the original official `sol_producer_kernel` after the
existing QKV projection. It writes official INT8 carriers/statistics and, via
the experimental extension, full BF16 Q/K/V for Spark. The retained Spark
path then runs its existing full reblock plan, global reweight, TopK10 route,
block tail, exact attention, and direct output scatter. Official INT8 carrier
generation can therefore be redundant with Spark's later preprocessing; it is
part of the measured candidate cost. The official `sol_attn_chunked` route and
exact core are outside this candidate.

## A/B conditions

- GPU3, baseline→candidate order, separate ComfyUI process per mode.
- Same frozen plugin source and CUDA extension binary for both modes, with
  SHA256 captured before and after each sampler.
- Archived prompt 1 (`vbench_all_0264`), seed 42, 10s/1344×768, 20 model
  evaluations, `spark_block`, TopK10, full reblock/reweight, block tail,
  direct output enabled, and otherwise unchanged graph settings.
- Excluded 5-step warmup in each process, followed by one complete 20-step
  sampler. Use ComfyUI's `SamplerCustomAdvanced` time and save full latents.
- Require `H3_SPARK_BSA_ORIGINAL_PRODUCER active` in the candidate server log
  and absent from baseline. The baseline process explicitly unsets both BSA
  experimental switches.
- Compare full latent SHA256 and per-tensor absolute error. Preserve the raw
  records, logs, protocols, source hashes, and final summary in a new
  experiment directory. A numerically different final latent must be reported
  and cannot be called bitwise-equivalent acceleration.

Runner: [`scripts/comfyui_bsa_official_producer_spark_full_ab_20260928.py`](../scripts/comfyui_bsa_official_producer_spark_full_ab_20260928.py).
Planned raw root:
`/autodl-fs/data/h3_experiments/comfyui_bsa_official_producer_spark_full_ab_20260928/`.

Status: complete. The GPU3 run followed this protocol; GPU0 and GPU2 added
reverse-order pairs under the same frozen build. Results are in
[`comfyui_bsa_original_producer_spark_full_ab_20260928.md`](comfyui_bsa_original_producer_spark_full_ab_20260928.md).
