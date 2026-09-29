# Diffusers global-reweight numeric ablation switches

This document records the September-26 numeric ablation. The Diffusers Spark
default has since been restored to the historical Table 4 BF16 global anchor,
with Tensor-Core summary and log-mass subtraction against the BF16-stored
summary key. Pass `sol_global_anchor_dtype="float32"` for the FP32 ablation.

```python
install_h3_spark_attn(
    transformer,
    num_inference_steps=20,
    sol_global_anchor_dtype="float32",
    sol_reweight_summary_math="comfy_fp32",
    sol_reweight_logmass_key="pre_round",
)
```

The three switches are independent:

| Switch | Choices | Changed stage |
| --- | --- | --- |
| `sol_global_anchor_dtype` | `bfloat16`, `float32` | Storage of the FP32-reduced query mean |
| `sol_reweight_summary_math` | `tensorcore`, `comfy_fp32` | Anchor–K scores and weighted K/V summary arithmetic |
| `sol_reweight_logmass_key` | `stored`, `pre_round` | BF16-rounded versus FP32 pre-round key in the log-mass shift |

`comfy_fp32` uses FP32 lane-wise FMA for anchor–K scores and FP32
softmax/weighted K/V reductions, then stores BF16 summary K/V and FP32
log-mass. This matches the ComfyUI reweight **precision stages**, not its
bitwise CUDA reduction order. ComfyUI's subsequent INT8 Sol attention
consumer is intentionally outside the reweight switches; neither route nor
query-tail granularity changes. Thus this preset is not an end-to-end
bitwise reproduction of the ComfyUI pipeline.

On SM120, the isolated summary benchmark with one global anchor, 73,573
tokens, and 32 heads gave 0.917 ms for `tensorcore/stored` and 0.893 ms for
`comfy_fp32/pre_round`. These are single-run kernel timings, **not** full
denoising speed results. Run
`scripts/benchmark_reweight_summary_numeric_modes.py` to repeat locally.

Unit tests compare per-block summaries against an FP32 softmax reference,
verify that switching only the log-mass convention leaves K/V unchanged,
and exercise the fused query-attention consumer. Full-video PSNR ablations
remain separate from this implementation check.
