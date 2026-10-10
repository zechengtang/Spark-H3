# Spark-H3 ComfyUI Nodes

<p align="center">
  <a href="README.md"><strong>English</strong></a> |
  <a href="README.zh-CN.md">简体中文</a>
</p>

Spark-H3 provides **MiniMax H3 Spark Attention (SM89)** and **MiniMax H3
Spark Attention (SM120)** nodes for ComfyUI's native MiniMax-H3 implementation.
Place the matching node between the model loader and `BasicGuider`, and apply
only one attention patch to a model. The current release targets Linux or Windows x86_64, NVIDIA
SM89 (GeForce RTX 40 series) or SM120 (GeForce RTX 50 series and RTX PRO
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
zero warmup and 10% Top-K. Install
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
