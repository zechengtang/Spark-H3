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

See the dedicated [ComfyUI installation and workflow guide](comfyui/README.md).
The installer builds the patched comfy-kitchen Spark CUDA backend alongside
the custom node. Restart ComfyUI afterward; the node appears as **MiniMax H3
Spark Attention (SM120)** in `model_patches/attention`.

## Workflow

Insert the node after `UNETLoader` and before `BasicGuider`:

```text
UNETLoader -> MiniMax H3 Spark Attention (SM120) -> BasicGuider
```

The [14.4 s native model example](workflows/spark_h3_vdn8_14p4s_t2va.json)
uses the complete three-shot VDN prompt 8, 345 frames at 1344×768, and a
20-step sampler. It has one Spark patch in the model path, without a Turbo
LoRA or a second sparse-attention patch.

Three matching 14.4-second, 8-step LoRA examples use the same full prompt,
seed, resolution, Spark 20% Top-K ratio, and two dense warmup evaluations:

| Workflow | LoRA loader | Sampler |
| --- | --- | --- |
| [MiniMax-H3 / ComfyUI 8-step LoRA](workflows/spark_h3_minimax_h3_comfyui_8step_lora_14p4s_t2va.json) | `LoraLoaderModelOnly` | `res_multistep` |
| [LightX2V 768p 8-step LoRA](workflows/spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json) | `LoraLoaderModelOnly` | `res_multistep` |
| [Larryvrh v4 8-step LoRA](workflows/spark_h3_larryvrh_8step_lora_14p4s_t2va.json) | `MiniMaxH3TurboLoRA` | `MiniMaxH3TurboSampler` |

Larryvrh's LoRA requires its custom loader; the stock LoRA loader cannot
apply that file to the local pruned base. The three examples use the filenames
already present under the local ComfyUI LoRA model path.

The defaults select a 20% Top-K ratio, 20% dense warmup, and the first
transformer block kept dense. Set `steps` to the
number of model evaluations made by the sampler (normally the sampler's step
count). The warmup ratio is converted to an exact number of model evaluations;
minimum-token and dense-layer gating use the same
policy object as ComfyUI's official sparse node.
The public Spark node uses the full reweight path and a dense video tail;
research ablation and tail-mode controls are not exposed in the node UI.
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

- ComfyUI 0.30.0 or newer with native MiniMax-H3 support.
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
