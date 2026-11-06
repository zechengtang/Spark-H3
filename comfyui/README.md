# Spark-H3 ComfyUI Nodes

<p align="center">
  <a href="README.md"><strong>English</strong></a> |
  <a href="README.zh-CN.md">简体中文</a>
</p>

Spark-H3 provides **MiniMax H3 Spark Attention (SM86)**, **MiniMax H3 Spark
Attention (SM89)**, and **MiniMax H3 Spark Attention (SM120)** nodes for
ComfyUI's native MiniMax-H3 implementation.
Place the matching node between the model loader and `BasicGuider`, and apply
only one attention patch to a model. The current release targets Linux or Windows x86_64, NVIDIA
SM86 (GeForce RTX 30 series), SM89 (GeForce RTX 40 series), or SM120 (GeForce RTX 50 series and RTX PRO
5000/6000 Blackwell), and CUDA BF16 execution.

## Installation

See the **[English installation and release guide](standalone/README.md#installation)**
for release selection, ZIP installation, manual version selection, and
troubleshooting. A **[Chinese installation guide](INSTALL.zh-CN.md)** is also
available.

## Models and Workflows

Download the native model files from the
[Comfy-Org MiniMax-H3 repository](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main).
Place the diffusion model under `models/diffusion_models`, the text encoder
under `models/text_encoders`, and the video and audio VAEs under `models/vae`.
You can also reuse an existing model directory through `extra_model_paths.yaml`.
The examples default to the official `minimax_h3_video_vae_fp16.safetensors`;
the VAE choice does not change the Spark attention algorithm.

Drag a workflow JSON onto the ComfyUI canvas and verify the model filenames.
The DMAD and LBH examples generate approximately 5.2-second videos; the other
examples generate approximately 14.4-second videos:

| Example workflow | Configuration |
| --- | --- |
| [Native MiniMax-H3](../workflows/spark_h3_vdn8_14p4s_t2va.json) | Core example; 20 steps, no LoRA, no extra nodes |
| [LightX2V 8-step LoRA](../workflows/spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json) | Extended example; [download the LightX2V LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/main) separately |
| [Larryvrh 8-step LoRA](../workflows/spark_h3_larryvrh_8step_lora_14p4s_t2va.json) | Extended example; [download the LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) and convert it to ComfyUI's generic format |
| [DMAD 4-step LoRA](../workflows/spark_h3_dmad_4step_lora_5p2s_t2va.json) | Uses the DMAD re-noise sampler; convert the DMAD LoRA first |
| [Official LBH two-pass + LightX2V 4-step](../workflows/spark_h3_lbh_official_lightx2v_4step_5p2s_i2va.json) | Requires the official LBH ComfyUI node and latent upscaler |

The LightX2V and Larryvrh workflows remain included in the standalone ZIP, but
Spark-H3 never installs their additional LoRAs silently. Put those files under
`models/loras`. Every example explicitly fixes `topk_mode=topk_ratio` and
`reblock_layout=q_reuse_k`. The native, LightX2V, Larryvrh, and DMAD examples
use `topk_ratio=0.2`; the LBH example uses `topk_ratio=0.1` to match the
Diffusers `h3_lbh` path. See the
[Spark node parameter reference](../docs/comfyui_spark_node_parameters.md)
(currently in Chinese) for the full interface.

The first Spark option is ordered as `fidelity`, `speed`, `custom`, `dense`.
`fidelity` (default) uses 20% of model evaluations as dense warmup steps, `speed` uses none,
`custom` uses manually configured warmup values, and `dense` returns the
unmodified model. The former enable switch is gone, but `warmup_mode`,
`warmup_ratio`, and `warmup_steps` remain available; manual changes after
selecting a preset automatically change the profile to `custom` when an active
warmup value no longer matches. Changing an inactive value does not. Switching
to `dense` preserves those values so switching back does not discard an
advanced configuration. Selecting `fidelity` again reapplies its 20% warmup to
unlinked controls. In `fidelity`, a non-serialized reminder button offers one-click
switching to `speed`, as recommended for the high-resolution second pass of a
two-pass workflow.

`sparse_scope` defaults to `target-video`, which preserves the released target-only
layout and direct-output path. Select `joint-video` for Ref2VA to place both
multi-timestep reference-video visual rows and the generated target video in
one shared sparse domain with joint Reblock, Top-K, and Reweight; static
reference images, text, reference audio, and target audio remain exact. With no
reference-video visual rows, `joint-video` automatically uses the identical
target-only path. Prefer `topk_ratio` with `joint-video`, because a fixed
`topk_blocks` budget represents
a smaller fraction of the enlarged candidate set. While an unlinked
`sparse_scope` remains `target-video`, a non-serialized reminder offers one-click
switching to `joint-video` for Ref2VA tasks with video conditioning.

`producer_chunk` optionally controls how many tokens the Spark Q/K/V producer
handles per chunk. It defaults to `16384` and must be a positive multiple of
64. Smaller values may reduce temporary producer memory at the cost of more
chunk calls; keep the default unless profiling shows a benefit. This is a
Python-side runtime setting and does not require rebuilding the wheel.

`dense_backend` controls dense warmup steps, permanently dense layers, short-sequence
execution, and non-strict fallback calls. Keep the default `inherit` to follow
the model's or ComfyUI's configured attention backend, or select `kitchen`, `sage`, or `sdpa`
to force an available backend for those calls only. This is a
Python-level dispatch choice and does not require rebuilding the architecture-
specific Spark wheel. `sage` requires SageAttention to be installed, while
`kitchen` uses the isolated `spark-h3-kitchen` INT8 attention implementation
and does not replace ComfyUI's official `comfy-kitchen` package.
`inherit` injects no override: an attention backend already selected by the
model, an upstream node, or ComfyUI—including an official comfy-kitchen
Kitchen Attention override—continues unchanged, but Spark-H3 does not select it.
While the unlinked setting remains `inherit`, the node shows a non-serialized
button announcing that Kitchen INT8 is available for dense stages. Clicking it
switches `dense_backend` to `kitchen`; the default remains `inherit`.

### KJNodes low-VRAM patches

Spark composes with ComfyUI-KJNodes' `MiniMax H3 Low VRAM Attention` and
`MiniMax H3 Chunk FeedForward`. Use this supported MODEL-chain order:

```text
model loader -> MiniMax H3 Low VRAM Attention -> MiniMax H3 Chunk FeedForward
             -> MiniMax H3 Spark Attention -> BasicGuider
```

Low VRAM Attention must precede Spark. Spark detects its `sol_take_forward`
compose hook while installing the model patch and selects a separate head-chunk
path. Without that node, Spark installs the original block patch and adds no
per-layer compatibility branch. Chunk FeedForward only chunks the MLP and is
orthogonal to the attention patch. Both integrations reuse the existing Spark
CUDA entry point, so the SM86, SM89, and SM120 wheels do not need rebuilding.
Additional chunks normally reduce peak VRAM at the cost of more kernel launches.

Convert the original Larryvrh weights with
`tools/convert_larryvrh_lora_comfyui.py`. The conversion retains 208 backbone
LoRAs and projects 51 AdaLN updates onto the pruned model's eight-dimensional
timestep curve. The workflow can then use ComfyUI's native
`LoraLoaderModelOnly`, just like LightX2V, without the Larryvrh custom node or
per-step runtime-LoRA matrix multiplications.

The LBH example follows the official two-pass structure. Its 0.2 MP
low-resolution stage performs four model evaluations, separates the audio and
video latents from `denoised_output`, and upscales only the video latent. It
then re-noises at 0.98 MP and performs three high-resolution evaluations with
`0.9035, 0.6316, 0.3158, 0.0`. Spark applies only to the second stage, with
`mode=speed` and 10% Top-K. Install
`LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler` and place
`minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors` under
`ComfyUI/models/latent_upscale_models/`.

## Implementation Notes

This is the initial Spark-Attn ComfyUI implementation, and end-to-end efficiency
continues to be optimized. Sol-Engine's approximate Sol branch downsamples only
K/V and retains per-query Q, while ComfyUI's native Sol downsamples both Q and
K/V by block. Spark defaults to the per-query mode, but the two implementations
still differ in routing, INT8 numerical paths, model weights, and inference
steps, so their outputs and acceleration ratios are not yet directly aligned.
