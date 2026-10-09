# ComfyUI Spark-H3

Independent ComfyUI custom nodes for running Spark block-sparse attention with
ComfyUI's native MiniMax-H3 model.

## Supported configuration

- ComfyUI 0.38.x or 0.39.x with native MiniMax-H3 support
- NVIDIA SM89 (RTX 4090) or SM120 GPU (RTX 50 series)
- PyTorch CUDA 13.0+ and CUDA BF16 execution
- Linux; Windows wheels can be produced from the same backend source but require
  a separately validated Windows release asset

Model weights are not bundled. Use ComfyUI's native MiniMax-H3 model files.

## Installation

Chinese instructions: [ComfyUI 中文安装指南](INSTALL.zh-CN.md).

This release provides the following architecture-specific archives:

| GPU | CUDA toolkit | Release archive |
| --- | --- | --- |
| SM120 (RTX 50 series) | 13.0 | `ComfyUI-Spark-H3-<version>-cu130.zip` |
| SM89 (RTX 4090) | 13.0 | `ComfyUI-Spark-H3-<version>-sm89-cu130.zip` |

Select by both GPU architecture and `torch.version.cuda`. CUDA 12.8 has been
removed from the release plan: ComfyUI disables its optimized comfy-kitchen
CUDA backend on that runtime, causing a severe end-to-end performance
regression. Existing local CU128 artifacts are diagnostic-only and must not be
published. Explicit local-wheel and source-build paths remain available under
`--experimental-cuda` for adaptation and correctness work. Each archive
supports both ComfyUI release lines, and
the installer selects the wheel matching the `comfy-kitchen` base already
present in ComfyUI:

| Installed ComfyUI | Spark backend |
| --- | --- |
| 0.38.x | `comfy-kitchen 0.2.36+spark.h3.<architecture>.cu130.1` |
| 0.39.x | `comfy-kitchen 0.2.37+spark.h3.<architecture>.cu130.1` |

SM89 wheels use the `+spark.h3.sm89.cu130.1` local version and contain code
compiled for CUDA architecture `89`. SM120 wheels use
`+spark.h3.sm120.cu130.1` and compile for `120f`. Experimental SM120 CU128
wheels use `+spark.h3.sm120.cu128.1`, so CUDA variants cannot match each other
even when placed in the same wheelhouse. The installer rejects a wheel for a
different architecture or CUDA toolchain.

Do not mix the architecture-specific backend versions or CUDA archives.
Source builds use CUDA 13.0+ by default. A CUDA 12.8 source build is available
only through the explicit experimental path and is not a release target.

For a release ZIP, stop ComfyUI, extract
the package under `ComfyUI/custom_nodes`, and use the same Python interpreter
that starts ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
unzip /path/to/ComfyUI-Spark-H3-<version>-cu130.zip  # SM120 example
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py
```

For a supported CUDA 13.0+ environment that needs a local source build:

```bash
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py --source
```

The installer checks the existing environment, the package's `wheelhouse`, and
the matching release wheel in that order. It compiles the pinned backend only
when no compatible wheel is available or `--source` is supplied. Source
compilation requires Git, CMake, Ninja, a C++ compiler, and a matching CUDA
13.0+ `nvcc`.

For local CU128 adaptation or correctness work, automatic wheel discovery is
disabled. Supply an explicit local wheel or request a source build:

```bash
python ComfyUI-Spark-H3/install.py \
  --experimental-cuda --wheel /path/to/cu128/comfy_kitchen-*.whl

python ComfyUI-Spark-H3/install.py --experimental-cuda --source
```

The SM120 source path uses CUDA 12.8's `120a` target. Experimental mode prints
a performance-regression warning and must not be represented as release
support or used for CU130-equivalent performance claims.

Automatic selection is recommended. For troubleshooting, select the expected
base explicitly:

```bash
# ComfyUI 0.38.x
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py --kitchen-base 0.2.36

# ComfyUI 0.39.x
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py --kitchen-base 0.2.37
```

Restart ComfyUI after installation. ComfyUI Manager can also install the node
after registry publication, but Manager is optional and is not bundled by this
package.

Both supported ComfyUI lines retain ComfyUI's default `cudaMallocAsync`
allocator; no `--disable-cuda-malloc` launch flag is required. The node keeps
only its allocation-heavy reblock planner out of a private PyTorch CUDA graph
to avoid captured-allocation lifetime failures. Spark's attention CUDA kernel
still runs normally.

## Repository independence and offline boundary

Each release ZIP contains the architecture/CUDA wheel combination named in the
table above, with wheels compatible with ComfyUI 0.38.x and 0.39.x. On Linux
x86_64 with CPython 3.12+ it does not clone or import code from the MiniMax-H3, Spark-H3,
Diffusers, `comfy-aimdo`, or `comfy-kitchen` repositories. It reuses the
PyTorch, Triton, CUDA integration, and native MiniMax-H3 implementation already
provided by ComfyUI.

ComfyUI and model weights are not bundled. Windows, Python 3.10/3.11, and
other wheel tags still require a separately built wheel or the source-build
fallback, which downloads the pinned `comfy-kitchen` source. Optional example
workflows can require separately downloaded LoRAs or third-party nodes; the
core Spark-H3 node does not.

To install a wheel supplied out of band:

```bash
python install.py --wheel /path/to/comfy_kitchen-0.2.37+spark.h3.sm120.cu130.1-*.whl
```

## Nodes

- **MiniMax H3 Spark Attention (SM89)**
- **MiniMax H3 Spark Attention (SM120)**
- **MiniMax H3 Sol Attention (Official Compatibility)**

Spark defaults to legacy midpoint directions, FP32 global anchors, exact
key-block budgets, query-granularity tails, and `q_reuse_k` reblocking.

## Example workflows

The package keeps five examples under `workflows/`:

- Native MiniMax-H3, 20 steps and no LoRA (minimal baseline).
- LightX2V 768p 8-step LoRA (the LoRA is downloaded separately).
- Larryvrh v4 8-step LoRA (requires the separately downloaded LoRA converted
  with `convert_larryvrh_lora_comfyui.py`; no Larryvrh custom node).
- DMAD 4-step LoRA with the bundled re-noise sampler (the upstream LoRA must be
  converted separately).
- LBH's official two-pass LightX2V 4-step workflow (requires the separately
  installed `LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler` node and LBH
  upscaler checkpoint).

All five explicitly store `reblock_layout=q_reuse_k` and default to the
official `minimax_h3_video_vae_fp16.safetensors`. They also use
`topk_mode=topk_ratio`. The native, LightX2V, Larryvrh, and DMAD examples use
20%; LBH uses the Diffusers-side `h3_lbh` default of 10% on its three-step
high-resolution pass. Optional LoRAs and third-party nodes are never installed
silently by Spark-H3.

The published Larryvrh weight must be converted once before loading that
workflow. The converter and its small projection asset are included in this
package; it does not require the Larryvrh custom node or the MiniMax-H3 source
repository:

```bash
python convert_larryvrh_lora_comfyui.py \
  --input /path/to/minimax_h3_turbo_v4_step600_ema.safetensors \
  --output /path/to/ComfyUI/models/loras/minimax_h3_turbo_v4_step600_ema_comfyui_curve_bf16.safetensors
```

## Backend wheel maintenance

From this custom-node directory:

```bash
python kernel_builder.py --kitchen-base 0.2.36 --output-dir wheelhouse
python kernel_builder.py --kitchen-base 0.2.37 --output-dir wheelhouse
```

The builder pins the selected upstream revision, applies the bundled Spark
patch, marks a default release wheel as either
`0.2.36+spark.h3.sm120.cu130.1` or
`0.2.37+spark.h3.sm120.cu130.1`, and compiles only SM120 by default. One custom-node ZIP
can contain both wheels and the installer chooses the matching one.
