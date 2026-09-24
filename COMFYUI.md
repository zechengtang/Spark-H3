# ComfyUI plugin (SM120)

This repository is also a ComfyUI custom node for the native MiniMax-H3 model.
The node uses the bundled Spark-H3 kernels on NVIDIA SM120 GPUs (for example,
GeForce RTX 50-series) and keeps all packed text, image/video reference, and
audio conditioning exact.

The integration is built on ComfyUI's official MiniMax-H3 sparse-attention
architecture: it installs `dit/double_block` replacements, follows the native
sigma-based sparse window, resets state through `ON_CLEANUP`, and projects QKV
in 4096-token chunks. The projected BTHD tensors then enter Spark-Reblock,
Spark-Reweight, and Spark Top-K routing. It does not replace each attention
module's `forward` method.

## Install

Clone the repository directly under `ComfyUI/custom_nodes`, then install the
kernel package into the same Python environment that runs ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/zechengtang/Spark-H3.git
cd Spark-H3
/path/to/ComfyUI/venv/bin/python -m pip install -e '.[cuda]'
```

Restart ComfyUI. The node appears as **MiniMax H3 Spark Attention (SM120)** in
`model_patches/attention`.

## Workflow

Insert the node after `UNETLoader` and before `BasicGuider`:

```text
UNETLoader -> MiniMax H3 Spark Attention (SM120) -> BasicGuider
```

The defaults reproduce the repository's current Spark policy: Top-K 10%, 20%
dense warmup, and the first transformer block kept dense. Set `steps` to the
number of model evaluations made by the sampler (normally the sampler's step
count). Warmup remains an exact percentage of those model evaluations for
workflow compatibility; minimum-token and dense-layer gating use the same
policy object as ComfyUI's official sparse node.
`strict=true` is recommended: it reports an incompatible ComfyUI build,
dtype, GPU, or kernel error instead of silently switching to dense attention.

Requirements:

- ComfyUI 0.30.0 or newer with native MiniMax-H3 support.
- CUDA BF16 execution on a compute-capability 12.0 GPU.
- A PyTorch/CUDA/Triton/CuTe stack compatible with this repository.

The node is compatible with ComfyUI's T2VA, FL2VA, and Ref2VA packed layouts.
It uses `transformer_options["minimax_h3_layout"]` rather than guessing token
spans. Do not stack it with another node that replaces MiniMax-H3
`dit/double_block` entries; choose one sparse-attention implementation per
workflow.
