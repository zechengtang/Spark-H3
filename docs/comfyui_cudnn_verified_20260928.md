# Verified cuDNN SDPA benchmark on ComfyUI H3 (2026-09-28)

## Scope and verification

This experiment compares Flash and cuDNN for the **dense SDPA calls** in the
existing 10 s / 768p, 20-step ComfyUI Spark-H3 sampler. It does not alter
Spark's sparse algorithm, TopK, reblock, reweight, or production defaults.
The prior external `sdpa_kernel(CUDNN_ATTENTION)` attempt was invalid because
`comfy.ops.scaled_dot_product_attention` resets backend priority to Flash
first. Its results were deleted. This run uses a separate ComfyUI source copy
with a backend override only for BF16 `[1, 56, 73573, 128]` H3 dense calls.
Each full-path arm has a fresh ComfyUI process and CUDA context. The two pairs
use opposite method order on GPU0, same prompt, seed 42, model, graph, and
5-step pre-run. The 20-step `SamplerCustomAdvanced` timing excludes first
model-load/warmup time.

The isolated probe used the same post-QKV projection view: stride
`[7168, 128, 21504, 1]`, no mask, no dropout, noncausal. CUDA profiler names
show `pytorch_flash::flash_fwd_kernel` for ComfyUI default and forced Flash,
and `cudnn_generated_fort_native_sdpa_sm120_flash_fprop...` for forced cuDNN.
The full-path cuDNN server logged the target branch with the same shape,
dtype, and stride in both pairs. Thus the backend switch really executed;
an outer context alone would not establish that.

## Results

| Pair order | Flash full sampler | cuDNN full sampler | Flash minus cuDNN |
| --- | ---: | ---: | ---: |
| Flash → cuDNN | 247.1008 s | 243.4290 s | 3.6718 s |
| cuDNN → Flash | 247.0941 s | 243.4025 s | 3.6916 s |
| Mean | **247.0975 s** | **243.4158 s** | **3.6817 s (1.51%)** |

Seven isolated SDPA calls on GPU0 gave median 430.36 ms for forced Flash
and 415.20 ms for forced cuDNN, about 3.5% lower latency for that call.
ComfyUI default launched Flash and had a 428.63 ms median. The two full-path
pairs support a consistent 3.68 s speed trend for this one prompt and GPU;
they do not establish a distribution across prompts, GPUs, or system load.

The first five-step prompt in each fresh server took 178.65/167.72 s for
Flash and 251.24/229.76 s for cuDNN. Its first model evaluation took
127.92/116.14 s for Flash and 188.67/171.67 s for cuDNN. This is a repeated
cold-start penalty of about 62–73 s at the prompt level, which includes
model loading and backend initialization; the present logs cannot isolate its
cause to a specific cuDNN operation. The steady-state sampler times above
exclude that cost.

The Flash latent SHA256 was identical in both pairs and to the earlier
2026-09-28 Spark control; the cuDNN latent SHA256 was also identical across
its two pairs, but differs from Flash. At the isolated attention output,
Flash versus cuDNN had mean absolute difference `1.12e-5` and maximum
absolute difference `2.44e-4` in BF16. At the final video latent, mean
absolute difference was `0.195` and RMSE `0.295`; audio latent mean absolute
difference was `0.0213`. Small attention differences can propagate through
the denoising trajectory. These measurements do **not** establish whether
visual or audio quality changes.

This is an independently measured dense-backend option, not a Spark-H3
implementation improvement. It is not enabled by default because the
full-generation numerical result changes and the fresh-process cost is
material. A quality check and cold-start investigation would be needed
before considering deployment.

## Artifacts

- Raw timings, backend CUDA kernel names, branch-hit status, output hashes,
  and latent errors: `/autodl-fs/data/h3_experiments/comfyui_cudnn_verified_20260928/summary.json`
- Individual server logs and measured latents:
  `/autodl-fs/data/h3_experiments/comfyui_cudnn_verified_20260928/fullpath_gpu0_10s*/`
- Isolated probe: `scripts/probe_comfy_cudnn_sdpa_20260928.py`
- Full-path A/B runner and experiment-only source copy creation:
  `scripts/comfyui_cudnn_fullpath_ab_20260928.py`

The production ComfyUI and Spark files were not edited for this experiment.
