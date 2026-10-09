# ComfyUI plugin (SM120)

This repository is also a ComfyUI custom node for the native MiniMax-H3 model.
Its current implementation and cross-pipeline differences are described in the
[ComfyUI guide](comfyui/README.md#实现说明).
The node uses the bundled Spark-H3 kernels on NVIDIA SM120 GPUs (for example,
GeForce RTX 50-series) and keeps all packed text, image/video reference, and
audio conditioning exact.

The integration is built on ComfyUI's official MiniMax-H3 sparse-attention
architecture and the comfy-kitchen v0.2.36 Sol kernel stack. It installs
`dit/double_block` replacements, follows the native sigma-based sparse window,
resets state through `ON_CLEANUP`, and projects QKV in 4096-token chunks. The
projected BTHD tensors then use comfy-kitchen kernels for Top-K routing and
global reweighting. The fanout-16, 32-landmark, group-1 reblock permutation is
consumed directly by Sol preprocessing and its output scatter, without four
materialized Q/K/V/output gathers. Global reweighting uses the mean
of all target-video queries as one anchor per head; conditioning queries remain
dense. It does not replace each attention module's `forward` method.

## Install

The recommended release artifact is the standalone
`ComfyUI-Spark-H3-<version>.zip`. It contains only the custom node, its reblock
runtime, workflows, and compatible backend wheels—not this repository's model
or research assets. Extract it below `custom_nodes` and run its installer with
the same Python interpreter that starts ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
unzip /path/to/ComfyUI-Spark-H3-<version>.zip
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py
```

After registry publication, ComfyUI Manager is the recommended discovery and
installation UI, but it remains optional: Spark-H3 neither bundles nor installs
Manager. The manual ZIP path is intentionally kept as an independently testable
baseline.

The cross-platform installer first accepts an existing compatible backend,
then selects a bundled or pinned-release wheel. It falls back to building the
pinned, patched `comfy-kitchen` source only when no wheel matches. Source builds
require Git, CMake, Ninja, a C++ compiler, and CUDA `nvcc`.

The current prebuilt wheels target Linux x86_64, Python 3.12+, CUDA 13.0, and
SM120. CUDA 12.8/12.9 users must run `install.py --source` with the matching
local CUDA toolkit. Python 3.10/3.11 also requires its own wheel or source
compilation. Windows packaging is prepared, but the Windows wheel and SM120
runtime have not yet been validated. This release does not support non-SM120
GPUs.

Restart ComfyUI afterward; the node appears as **MiniMax H3 Spark Attention
(SM120)** in `model_patches/attention`. See the dedicated
[ComfyUI installation and workflow guide](comfyui/README.md) for model files and
examples.

## Workflow

Insert the node after `UNETLoader` and before `BasicGuider`:

```text
UNETLoader -> MiniMax H3 Spark Attention (SM120) -> BasicGuider
```

The [14.4 s native model example](workflows/spark_h3_vdn8_14p4s_t2va.json)
uses the complete three-shot VDN prompt 8, 345 frames at 1344×768, and a
20-step sampler. It has one Spark patch in the model path, without a Turbo
LoRA or a second sparse-attention patch. Bundled workflows default to the
official `minimax_h3_video_vae_fp16.safetensors`.

Two matching 14.4-second, 8-step LoRA examples use the same full prompt,
seed, resolution, Spark 20% Top-K ratio, and two dense warmup evaluations. The
DMAD and Alibaba-PAI examples retain the sampling protocol required by their
respective students:

| Workflow | LoRA loader | Sampler |
| --- | --- | --- |
| [LightX2V 768p 8-step LoRA](workflows/spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json) | `LoraLoaderModelOnly` | `res_multistep` |
| [Larryvrh v4 8-step LoRA](workflows/spark_h3_larryvrh_8step_lora_14p4s_t2va.json) | `MiniMaxH3TurboLoRA` | `MiniMaxH3TurboSampler` |
| [DMAD 4-step LoRA](workflows/spark_h3_dmad_4step_lora_5p2s_t2va.json) | `LoraLoaderModelOnly` | `MiniMaxH3DMADSampler` |
| [Alibaba-PAI PDD Acc FL2VA 8-step LoRA](workflows/spark_h3_alibaba_pai_acc_fl2va_8step_lora_5p2s_t2va.json) | `MiniMaxH3PDDAccApply` | `euler` + node-provided sigmas |

Larryvrh's LoRA requires its custom loader; the stock LoRA loader cannot
apply that file to the local pruned base. The workflows use filenames from the
local ComfyUI LoRA model path. Both the LightX2V and Larryvrh examples remain
in the standalone ZIP as explicitly labelled optional integrations.

DMAD publishes a Diffusers-format rank-128 LoRA. Convert it before selecting it
in `LoraLoaderModelOnly`:

```bash
python scripts/convert_dmad_lora_to_comfyui.py \
  /path/to/dmad_minimax_h3_4step_lora_critic.safetensors \
  /path/to/ComfyUI/models/loras/dmad_minimax_h3_4step_lora_critic_comfyui_bf16.safetensors
```

The converter fuses Q/K/V adapters without changing their represented update
and swaps the Diffusers SwiGLU halves to ComfyUI order. The DMAD workflow then
applies `MiniMaxH3SigmaShift` with video shift 12 and audio shift 2 before
Spark. Use the bundled `MiniMaxH3DMADSampler`; a regular Euler or
`res_multistep` sampler does not implement the re-noise rule used to train the
released student.

Alibaba-PAI's MiniMax-H3-Acc checkpoints are also Diffusers-side weights, but
they are not ordinary PEFT LoRAs. Each file contains a rank-64 trunk LoRA and
32 PDD video/audio output heads. Diffusers can use the official file directly
with [`scripts/minimax_h3_pdd.py`](scripts/minimax_h3_pdd.py): call
`apply_pdd_lora` on `pipeline.transformer` for FL2VA or
`pipeline.transformer_ref` for Ref2VA, then pass
`num_inference_steps=nfe + 1` to the modular pipeline.
The pinned source revision, local file sizes, and SHA-256 checksums are recorded
in the [MiniMax-H3-Acc integration note](docs/alibaba_pai_minimax_h3_acc.md).

For ComfyUI, install the Apache-2.0
`BSAI-ComfyUI-MiniMax-H3-PDD-Acc` node pack and place the original weights in
`models/pdd_acc`. The node reads the official files directly, converts the
trunk keys in memory, handles the pruned checkpoint's AdaLN curve basis, and
installs the PDD head bank. Pair the FL2VA weight only with an FL2VA base and
the Ref2VA weight only with a Ref2VA base. Use CFG 1.0, video/audio shifts
12/3, the plain `euler` sampler, and wire the Apply node's `sigmas` output to
`SamplerCustomAdvanced`; do not substitute `BasicScheduler`, `res_multistep`,
or another distillation LoRA.

The defaults select a 20% Top-K ratio, 20% dense warmup, and the first
transformer block kept dense. Set `steps` to the
number of model evaluations made by the sampler (normally the sampler's step
count). The warmup ratio is converted to an exact number of model evaluations;
minimum-token and dense-layer gating use the same
policy object as ComfyUI's official sparse node.
The public Spark node uses the full reweight path and a dense video tail;
research ablation and tail-mode controls are not exposed in the node UI.
The default `reblock_layout=q_reuse_k` builds the K-side landmark layout once
and aliases that permutation for Q, matching Spark's `q_from_k` mode while
avoiding the query-side M2 transform, tree plan, and duplicate index storage.
Select `independent` to reproduce the former separately planned Q/K layouts.
The default `topk_mode=topk_ratio` selects a fraction of the target video's
64-token key blocks with `topk_ratio` (default 0.2). Set
`topk_mode=topk_blocks` to request a fixed number with `topk_blocks` (default
228); the ratio field is ignored in this mode. The block count is capped at the number
of candidate video blocks, while conditioning/sink blocks remain exact and
do not consume the Top-K budget. To use a 10% ratio, set `topk_mode=topk_ratio`
and `topk_ratio=0.1`.
`warmup_mode=warmup_ratio` (default) computes dense warmup as the ceiling of
`steps * warmup_ratio`, where `0.2` means 20%; `warmup_mode=warmup_steps`
uses the fixed
`warmup_steps` count (default 4). Both modes cap warmup at `steps`.
The frontend migrates legacy workflow values such as `warmup_percent=20` to
`warmup_ratio=0.2` when loading the graph.
`strict=true` is recommended: it reports an incompatible ComfyUI build,
dtype, GPU, or kernel error instead of silently switching to dense attention.

Requirements:

- ComfyUI 0.38.x or 0.39.x with native MiniMax-H3 support.
- CUDA BF16 execution on a compute-capability 12.0 GPU.
- The ComfyUI-supported PyTorch/CUDA stack and a comfy-kitchen build containing
  the Spark Top-K, reblock, and global-reweight extension.

The ComfyUI integration never calls Spark-H3's research Triton/CuTe attention
backends. The compatibility Sol node delegates to ComfyUI's official
`apply_block_sparse_attention`, while the Spark node fails explicitly unless
the comfy-kitchen global Spark backend is selected. The repository's original
PyTorch pipeline and experimental kernels remain available only through the
Python API.

The node is compatible with ComfyUI's T2VA, FL2VA, and Ref2VA packed layouts.
It uses `transformer_options["minimax_h3_layout"]` rather than guessing token
spans. Do not stack it with another node that replaces MiniMax-H3
`dit/double_block` entries; choose one sparse-attention implementation per
workflow.
