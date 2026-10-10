---
license: apache-2.0
pipeline_tag: text-to-video
tags:
  - text-to-video
  - image-to-video
  - audio-video-generation
  - minimax-h3
  - sparse-attention
  - block-sparse
  - inference-acceleration
  - triton
  - comfyui
inference: false
---

<div align="center">

<img src="assets/spark-h3-wordmark.png" alt="Spark-H3" width="300">

<h1 align="center">Spark-H3</h1>

<h3 align="center">Adaptive Block Sparse Attention for MiniMax-H3</h3>

**Reblock similar tokens · Reweight tail tokens**

[Technical Blog](https://zechengtang.github.io/Spark-H3/) ·
[GitHub](https://github.com/zechengtang/Spark-H3) ·
[ComfyUI](comfyui/README.md) ·
[Documentation](h3_sparse_attention/README.md) ·
[Base Model](https://huggingface.co/MiniMaxAI/MiniMax-H3)

</div>

> [!NOTE]
> This is a code-only release of the Spark-H3 attention backend. It does not
> redistribute MiniMax-H3 model weights. Download the base model from
> [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## Overview

Spark-H3 is the MiniMax-H3 implementation of **Spark-Attn**, a block sparse
attention method for accelerating video generation while preserving output
fidelity. It improves both branches of block-sparse attention:

- **Spark-Reblock** groups tokens with similar attention preferences so the
  exact branch spends its budget on more relevant interactions.
- **Spark-Reweight** builds weighted key/value summaries and corrects their
  attention mass, reducing the bias introduced by mean-pooled tail blocks.

Spark-H3 is an inference backend rather than a LoRA or a new checkpoint. It
does not modify the MiniMax-H3 backbone weights and can be combined with
community LoRAs.

### Highlights

- Up to **2.33× attention speedup** and **1.73× DiT speedup** in a 10-second,
  768p benchmark at 10% attention density. All sparse methods share a 20%
  dense warmup, with the first Transformer layer kept dense throughout.
- **23.30 dB PSNR** at 10% attention density, compared with **20.36 dB** for
  Sol-H3 under the same evaluation protocol.
- Spark-H3's Diffusers integration implements dedicated backends for NVIDIA
  **SM80**, **SM89**, and **SM120** GPUs.
- Native ComfyUI nodes are implemented for **SM89** and **SM120**, with
  prebuilt Linux x86_64 release packages available for both architectures.
- Thanks to ComfyUI's optimized MiniMax-H3 execution stack, Spark-H3 reaches
  up to **3.15× DiT speedup** for 14.4-second, 1344×768 generation using the
  20% attention-density Speed profile.

> Unless otherwise noted, all latency measurements on this page were collected
> on a single NVIDIA RTX PRO 6000 Blackwell GPU.

## Samples

### ComfyUI — Dense, Fidelity, and Speed

Each video below contains three synchronized outputs from complete ComfyUI
workflows: **Dense** with Spark-H3 disabled, **Fidelity** with 20% attention
density and a short dense warmup, and **Speed** with 20% attention density and
no dense warmup. Fidelity and Speed use `dense_layers=0`.

Across all cases, Native MiniMax-H3 uses 20 denoising steps and 4 Fidelity
warmup steps. The LightX2V variant uses 8 denoising steps and 2 Fidelity
warmup steps.

The prompts used in these comparisons were provided by
[Video DeltaNet (VDN)](https://openvdn.github.io/) and
[LBH](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler). We thank
both teams for their awesome prompts.

#### 1344×768 · 14.4 seconds · VDN Prompt 08

This case uses VDN Prompt 08, a cinematic Japanese city sequence.

| Variant | Dense | Fidelity | Speed |
| --- | ---: | ---: | ---: |
| Native MiniMax-H3 | 1018.5 s · 1.00× | 461.7 s · **2.21×** | 323.3 s · **3.15×** |
| LightX2V 8-step v1.0 768p | 408.0 s · 1.00× | 199.5 s · **2.05×** | 129.6 s · **3.15×** |

##### Native MiniMax-H3

<video controls width="100%" src="https://huggingface.co/Aazeus/Spark-H3/resolve/main/comfyui/demos/H3_native_dense_fidelity_speed_short_warmup_blog_raspberry.mp4"></video>

##### LightX2V 8-step v1.0 768p

<video controls width="100%" src="https://huggingface.co/Aazeus/Spark-H3/resolve/main/comfyui/demos/H3_lightx2v_dense_fidelity_speed_short_warmup_blog_raspberry.mp4"></video>

#### 1344×768 · 14.4 seconds · VDN Selected Prompt 06

This case shows a young woman opening an attic window into the rain.

| Variant | Dense | Fidelity | Speed |
| --- | ---: | ---: | ---: |
| Native MiniMax-H3 | 1020.1 s · 1.00× | 465.3 s · **2.19×** | 326.6 s · **3.12×** |
| LightX2V 8-step v1.0 768p | 408.8 s · 1.00× | 200.7 s · **2.04×** | 131.0 s · **3.12×** |

##### Native MiniMax-H3

<video controls width="100%" src="https://huggingface.co/Aazeus/Spark-H3/resolve/main/comfyui/demos/H3_native_vdn_selected_0006_dense_fidelity_speed_native_resolution.mp4"></video>

##### LightX2V 8-step v1.0 768p

<video controls width="100%" src="https://huggingface.co/Aazeus/Spark-H3/resolve/main/comfyui/demos/H3_lightx2v_vdn_selected_0006_dense_fidelity_speed_native_resolution.mp4"></video>

#### 1344×768 · 14.4 seconds · VDN Prompt 01

This case uses a cinematic Japanese city montage.

| Variant | Dense | Fidelity | Speed |
| --- | ---: | ---: | ---: |
| Native MiniMax-H3 | 1030.5 s · 1.00× | 472.0 s · **2.18×** | 331.9 s · **3.10×** |
| LightX2V 8-step v1.0 768p | 413.7 s · 1.00× | 203.8 s · **2.03×** | 133.3 s · **3.10×** |

##### Native MiniMax-H3

<video controls width="100%" src="https://huggingface.co/Aazeus/Spark-H3/resolve/main/comfyui/demos/H3_native_vdn_0001_dense_fidelity_speed_native_resolution.mp4"></video>

##### LightX2V 8-step v1.0 768p

<video controls width="100%" src="https://huggingface.co/Aazeus/Spark-H3/resolve/main/comfyui/demos/H3_lightx2v_vdn_0001_dense_fidelity_speed_native_resolution.mp4"></video>

#### 1920×1088 · 5.2 seconds · LBH prompt

This case uses an anime action sequence prompt from LBH.

| Variant | Dense | Fidelity | Speed |
| --- | ---: | ---: | ---: |
| Native MiniMax-H3 | 569.7 s · 1.00× | 277.3 s · **2.05×** | 203.8 s · **2.80×** |
| LightX2V 8-step v1.0 768p | 228.1 s · 1.00× | 118.5 s · **1.92×** | 81.5 s · **2.80×** |

##### Native MiniMax-H3

<video controls width="100%" src="https://huggingface.co/Aazeus/Spark-H3/resolve/main/comfyui/demos/H3_native_lbh_dense_fidelity_speed_blog_raspberry.mp4"></video>

##### LightX2V 8-step v1.0 768p

<video controls width="100%" src="https://huggingface.co/Aazeus/Spark-H3/resolve/main/comfyui/demos/H3_lightx2v_lbh_dense_fidelity_speed_blog_raspberry.mp4"></video>

### VDN + LightX2V

The three samples below use **VDN prompts 01, 06, and 10** from [VDN blog](https://openvdn.github.io/) with the
[LightX2V MiniMax-H3 Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
and Spark-H3 at 10% attention density, with a 20% dense warmup and the first
Transformer layer kept dense throughout.

#### VDN Prompt 01 — Japanese city montage

<video controls width="100%" src="https://zechengtang.github.io/Spark-H3/vdn10/media/vdn_0001_japanese_woman_city_closeup.mp4"></video>

#### VDN Prompt 06 — Rainy-night camcorder documentary

<video controls width="100%" src="https://zechengtang.github.io/Spark-H3/vdn10/media/vdn_0006_night_camcorder_street_documentary.mp4"></video>

#### VDN Prompt 10 — 2D anime fashion sequence

<video controls width="100%" src="https://zechengtang.github.io/Spark-H3/vdn10/media/vdn_0010_2d_anime_eye_pullout.mp4"></video>

### LBH two-stage sampling

The following samples combine Spark-H3 with SelfLift two-stage sampling based
on the LBH learned latent upsampler.

#### LBH Case 01 — Black cat by the pool

<table>
<tr>
<th width="50%" align="center" valign="top">Dense</th>
<th width="50%" align="center" valign="top">Spark-H3, 10%</th>
</tr>
<tr>
<td align="center"><sub>DiT latency: 108.5 s · 1.00×</sub></td>
<td align="center"><sub>DiT latency: 59.8 s · 1.81×</sub></td>
</tr>
<tr>
<td valign="top"><video controls width="100%" src="https://zechengtang.github.io/Spark-H3/selflift/media/0753_dense.mp4"></video></td>
<td valign="top"><video controls width="100%" src="https://zechengtang.github.io/Spark-H3/selflift/media/0753_spark.mp4"></video></td>
</tr>
</table>

#### LBH Case 02 — Stormtrooper on the beach

<table>
<tr>
<th width="50%" align="center" valign="top">Dense</th>
<th width="50%" align="center" valign="top">Spark-H3, 10%</th>
</tr>
<tr>
<td align="center"><sub>DiT latency: 108.9 s · 1.00×</sub></td>
<td align="center"><sub>DiT latency: 59.5 s · 1.83×</sub></td>
</tr>
<tr>
<td valign="top"><video controls width="100%" src="https://zechengtang.github.io/Spark-H3/selflift/media/0685_dense.mp4"></video></td>
<td valign="top"><video controls width="100%" src="https://zechengtang.github.io/Spark-H3/selflift/media/0685_spark.mp4"></video></td>
</tr>
</table>

### Ref2VA

Spark-H3 is also compatible with MiniMax-H3's Ref2VA pipeline, as
shown on the [official Ref2VA case](https://github.com/MiniMax-AI/MiniMax-H3#case-ref2va).

#### Official Ref2VA case

<table>
<tr>
<th width="33.33%" align="center" valign="top">Dense</th>
<th width="33.33%" align="center" valign="top">Spark-H3, 10%</th>
<th width="33.33%" align="center" valign="top">Spark-H3, 10% + Compact Cond</th>
</tr>
<tr>
<td align="center"><sub>DiT latency: 708.5 s · 1.00×</sub></td>
<td align="center"><sub>DiT latency: 455.0 s · 1.56×</sub></td>
<td align="center"><sub>DiT latency: 202.6 s · 3.50×</sub></td>
</tr>
<tr>
<td valign="top"><video controls width="100%" src="https://zechengtang.github.io/Spark-H3/ref2va/media/dense.mp4"></video></td>
<td valign="top"><video controls width="100%" src="https://zechengtang.github.io/Spark-H3/ref2va/media/spark_ref_target.mp4"></video></td>
<td valign="top"><video controls width="100%" src="https://zechengtang.github.io/Spark-H3/ref2va/media/spark_ref_target_condition_compression.mp4"></video></td>
</tr>
</table>

See the [technical blog](https://zechengtang.github.io/Spark-H3/) for more results.

## Benchmark Results

**Evaluation setup.** Dense attention, Sol-H3, Spark-H3, and Spark-H3-Lite are
evaluated on VBench prompts with 19 denoising steps, 1344×768 resolution, and 240 frames at 24 fps
(10 seconds). All sparse methods—Sol-H3, Spark-H3, and Spark-H3-Lite—use a 20%
dense warmup (the first 4 of 19 Transformer evaluations) and keep the first
Transformer layer dense throughout. Apart from these shared dense settings,
Sol-H3 follows its official configuration. Quality metrics compare each method
against dense outputs generated with the same prompts and seeds. Higher PSNR
and SSIM are better; lower LPIPS is better.

| Method | Density ↓ | DiT latency (s) ↓ | DiT speedup ↑ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | ATTN speedup ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 100% | 583.3 | 1.00× | ∞ | 1.00 | 0.00 | 1.00× |
| Sol-H3 | — | 355.4 | 1.64× | 20.36 | 0.71 | 0.20 | 2.19× |
| **Spark-H3, 10%** | **10%** | 337.4 | 1.73× | **23.30** | 0.80 | 0.14 | **2.33×** |
| Spark-H3, 15% | 15% | 355.4 | 1.64× | 24.45 | 0.83 | 0.11 | — |
| Spark-H3, 20% | 20% | 371.1 | 1.57× | 25.38 | 0.85 | 0.09 | 1.96× |
| Spark-H3, 30% | 30% | 404.1 | 1.44× | 27.07 | 0.88 | 0.07 | 1.76× |
| **Spark-H3-Lite, 10%** | **10%** | **330.2** | **1.77×** | **23.34** | 0.80 | 0.14 | — |
| Spark-H3-Lite, 15% | 15% | 346.1 | 1.69× | 24.47 | 0.83 | 0.11 | — |
| Spark-H3-Lite, 20% | 20% | 362.6 | 1.61× | 25.12 | 0.85 | 0.10 | — |
| Spark-H3-Lite, 30% | 30% | 395.4 | 1.48× | 26.26 | 0.87 | 0.08 | — |

At the 10% density operating point, Spark-H3 improves PSNR by **2.94 dB** over
Sol-H3 while also reducing DiT latency from 355.4 s to 337.4 s. Increasing the
attention density provides a monotonic fidelity trade-off for standard
Spark-H3, reaching 27.07 dB PSNR at 30% density.

## Quick Start

Set up the upstream MiniMax-H3 pipeline first, then install Spark-H3 in the
same environment with a compatible PyTorch/CUDA stack:

```bash
git clone https://github.com/zechengtang/Spark-H3.git
cd Spark-H3
python -m pip install -e '.[cuda]'
```

Wrap an already loaded MiniMax-H3 pipeline with the Spark installer:

```python
from h3_sparse_attention import install_h3_spark_attn

num_denoise_steps = 19
num_inference_steps = num_denoise_steps + 1

with install_h3_spark_attn(
    pipe.transformer,
    num_denoise_steps=num_denoise_steps,
    warmup_mode="warmup_steps",
    warmup_steps=4,
    dense_layers=0,
    min_tokens=8192,
):
    result = pipe(**inputs, num_inference_steps=num_inference_steps)
```

MiniMax-H3's scheduler includes the terminal sigma value in
`num_inference_steps`, but that point does not execute the Transformer. Pass
the actual Transformer evaluation count to Spark as `num_denoise_steps`.

For configuration options, supported GPU architectures, and implementation
details, see the [Spark-H3 documentation](h3_sparse_attention/README.md).

## ComfyUI

Spark-H3 provides released Linux ComfyUI nodes for NVIDIA SM89 and SM120 GPUs,
along with example workflows. Follow the
[ComfyUI installation guide](comfyui/README.md), then start with one of the
workflows under [`workflows/`](workflows/).

## License

Spark-H3's original code and documentation are released under the
[Apache License 2.0](LICENSE). Included third-party components retain their
respective licenses and attribution notices. MiniMax-H3 model weights are not
included and remain subject to the
[upstream MiniMax-H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).

## Acknowledgments

Spark-H3 builds on
[MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) and
[Sol-Attn / Sol-Engine](https://github.com/NVlabs/Sana/tree/sol-engine).
The VDN prompt cases are sourced from the
[Video DeltaNet project](https://openvdn.github.io/).
