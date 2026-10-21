# ComfyUI Spark-H3

Independent ComfyUI custom nodes for running Spark block-sparse attention with
ComfyUI's native MiniMax-H3 model.

Spark-H3 accelerates MiniMax-H3 video generation by grouping similar
tokens and correcting tail tokens through reweighting during sparse attention.
It provides `fidelity`, `speed`, `custom`, and `dense` modes, supports KJNodes
low-VRAM workflows, and lets dense warmup steps use the inherited, Kitchen,
Sage, or PyTorch SDPA attention backend.

The v0.1.3 Linux benchmark used ComfyUI 0.38.2 and the LightX2V 768p 8-step
LoRA workflow at 1344×768, 345 frames, and 24 FPS. It used
`topk_ratio=20%`, `warmup_steps=2`, and PyTorch SDPA for dense attention.

| GPU | Fidelity vs. Dense | Speed vs. Dense |
| --- | ---: | ---: |
| RTX 3090 (SM86) | **2.42×** | **4.62×** |
| RTX 4090 (SM89) | **2.37×** | **4.35×** |
| RTX PRO 6000 Blackwell (SM120) | **2.05×** | **3.15×** |

Additional Linux tests on RTX 3090 and RTX PRO 6000 Blackwell showed slightly
higher speedup ratios with ComfyUI 0.39.2. ComfyUI 0.40.x uses the same
comfy-kitchen 0.2.37 backend and is also supported.

## Supported configuration

- ComfyUI 0.38.x, 0.39.x, 0.40.x, or 0.42.x with native MiniMax-H3 support
- NVIDIA SM86 (GeForce RTX 30 series), SM89 (GeForce RTX 40 series), or SM120 GPU (GeForce RTX 50 series and
  RTX PRO 5000/6000 Blackwell)
- PyTorch CUDA 13.0 and CUDA BF16 execution
- Linux x86_64 or Windows x86_64; the Windows CUDA 13.0 builds have been
  validated on their target platforms

**Supported ComfyUI releases: 0.38.x, 0.39.x, 0.40.x, and 0.42.x.** The installer selects the
Spark backend matching the `comfy-kitchen` dependency in the ComfyUI environment.
Install or update dependencies with that ComfyUI release's `requirements.txt`.

Model weights are not bundled. Use ComfyUI's native MiniMax-H3 model files.

## Installation

Chinese instructions: [ComfyUI 中文安装指南](INSTALL.zh-CN.md).

The Comfy Registry package contains the node and workflows, without bundled
backend wheels. When installed through ComfyUI Manager, its `install.py` runs
in the ComfyUI Python environment and selects the Spark-enabled comfy-kitchen
wheel from the selected GitHub Release archive using the installed
comfy-kitchen version, operating system, GPU architecture, and CUDA version.
This requires access to GitHub Releases during installation. A matching local wheel can be
supplied with `install.py --wheel` for offline installation.

This release provides the following architecture-specific archives:

| GPU | CUDA toolkit | Release archive |
| --- | --- | --- |
| SM86 (GeForce RTX 30 series) | 13.0 | `ComfyUI-Spark-H3-<version>-linux-x86_64-sm86-cu130.zip` |
| SM120 (GeForce RTX 50 series; RTX PRO 5000/6000 Blackwell) | 13.0 | `ComfyUI-Spark-H3-<version>-linux-x86_64-sm120-cu130.zip` |
| SM89 (GeForce RTX 40 series) | 13.0 | `ComfyUI-Spark-H3-<version>-linux-x86_64-sm89-cu130.zip` |
| SM86 (GeForce RTX 30 series) | 13.0 | `ComfyUI-Spark-H3-<version>-windows-x86_64-sm86-cu130.zip` |
| SM120 (GeForce RTX 50 series; RTX PRO 5000/6000 Blackwell) | 13.0 | `ComfyUI-Spark-H3-<version>-windows-x86_64-sm120-cu130.zip` |
| SM89 (GeForce RTX 40 series) | 13.0 | `ComfyUI-Spark-H3-<version>-windows-x86_64-sm89-cu130.zip` |

Select by both GPU architecture and operating system. The installer selects
the wheel matching the `comfy-kitchen` base already
present in ComfyUI:

| Installed ComfyUI version | Corresponding comfy-kitchen backend version |
| --- | --- |
| ComfyUI 0.38.x | `comfy-kitchen 0.2.36+spark.h3.<architecture>.<CUDA>.1` |
| ComfyUI 0.39.x or 0.40.x | `comfy-kitchen 0.2.37+spark.h3.<architecture>.<CUDA>.1` |
| ComfyUI 0.42.x | `comfy-kitchen 0.2.39+spark.h3.<architecture>.<CUDA>.1` |

The v0.1.5 archives contain matching 0.2.36, 0.2.37, and 0.2.39 wheels for
their operating system and GPU architecture. The installer selects the wheel
required by the installed ComfyUI version.

SM86 wheels use the `+spark.h3.sm86.cu130.1` local version and compile for
CUDA architecture `86`. SM89 wheels use the `+spark.h3.sm89.cu130.1` local version and contain code
compiled for CUDA architecture `89`. SM120 wheels use
`+spark.h3.sm120.cu130.1` and compile for `120f`. The installer rejects a wheel
for a different architecture. On a newer PyTorch CUDA runtime, it tries the
packaged cu130 wheel and checks the native entry point after installation.

The archive and its `spark_h3_build.json` also encode the target platform.
Linux packages accept `linux_x86_64` or compatible manylinux wheel tags;
Windows packages accept only `win_amd64`. A package built for the other OS is
rejected before backend installation. The Windows CUDA 13.0 packages have been
validated independently on SM89 and SM120 hardware.
The Linux SM86 CUDA 13.0 package has been validated on an RTX 3090.

Do not mix the architecture-specific backend versions or CUDA archives.

For a release ZIP, stop ComfyUI, extract
the package under `ComfyUI/custom_nodes`, and use the same Python interpreter
that starts ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
unzip /path/to/ComfyUI-Spark-H3-<version>-linux-x86_64-sm120-cu130.zip  # SM120 example
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py
```

For a supported environment that needs a local source build:

```bash
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py --source
```

The installer checks the existing environment, the package's `wheelhouse`, and
the matching release wheel in that order. It compiles the pinned backend only
when no compatible wheel is available or `--source` is supplied. Source
compilation requires Git, CMake, Ninja, a C++ compiler, and a matching CUDA
`nvcc`.

CUDA 12.8 remains available only as an explicit local experimental path:

```bash
python ComfyUI-Spark-H3/install.py --experimental-cuda --wheel /path/to/cu128/comfy_kitchen-*.whl

python ComfyUI-Spark-H3/install.py --experimental-cuda --source
```

The SM120 source path uses CUDA 12.8's `120a` target. This path is not a release
asset and must not be represented as release-supported.

The installer does not check the PyTorch version number; it requires a CUDA-enabled PyTorch build with `torch.version.cuda >= 13.0`.
It automatically tries the packaged cu130 wheel on newer runtimes without an
extra flag. It checks the wheel's platform, Python tag, GPU architecture and
comfy-kitchen base, then probes the native entry point after installation.
A successful install does not establish full workflow or performance
compatibility on the newer runtime.

Automatic selection is recommended. For troubleshooting, specify the expected
`comfy-kitchen` base with `--kitchen-base`:

```bash
# ComfyUI 0.38.x
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py --kitchen-base 0.2.36

# ComfyUI 0.39.x or 0.40.x
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py --kitchen-base 0.2.37
```

Restart ComfyUI after installation. ComfyUI Manager can also install the node
after registry publication, but Manager is optional and is not bundled by this
package.

All supported ComfyUI lines retain ComfyUI's default `cudaMallocAsync`
allocator; no `--disable-cuda-malloc` launch flag is required. The node keeps
only its allocation-heavy reblock planner out of a private PyTorch CUDA graph
to avoid captured-allocation lifetime failures. Spark's attention CUDA kernel
still runs normally.

If only `fidelity` runs out of GPU memory while the same workflow succeeds in
`dense` and `speed`, try adding `--disable-cuda-malloc` to the ComfyUI launch
command as a temporary workaround. This is a ComfyUI startup flag, not a node
setting.

## Repository independence and offline boundary

Each release ZIP contains the architecture/CUDA wheel combination named in the
table above, with wheels compatible with ComfyUI 0.38.x, 0.39.x, 0.40.x, and 0.42.x. On Linux
or Windows x86_64 with CPython 3.12+ it does not clone or import code from the MiniMax-H3, Spark-H3,
Diffusers, `comfy-aimdo`, or `comfy-kitchen` repositories. It reuses the
PyTorch, Triton, CUDA integration, and native MiniMax-H3 implementation already
provided by ComfyUI.

ComfyUI and model weights are not bundled. Python 3.10/3.11 and other wheel
tags require a matching wheel or the source-build fallback, which
downloads the pinned `comfy-kitchen` source. Optional example
workflows can require separately downloaded LoRAs or third-party nodes; the
core Spark-H3 node does not.

To install a wheel supplied out of band:

```bash
python install.py --wheel /path/to/comfy_kitchen-0.2.37+spark.h3.sm120.cu130.1-*.whl
```

## Nodes

- **MiniMax H3 Spark Attention (SM86)**
- **MiniMax H3 Spark Attention (SM89)**
- **MiniMax H3 Spark Attention (SM120)**
- **MiniMax H3 Sol Attention (Official Compatibility)**

Spark defaults to legacy midpoint directions, FP32 global anchors, exact
key-block budgets, query-granularity tails, and `q_reuse_k` reblocking.

`sparse_scope=target-video` preserves the original target-only layout and direct-output
path. For Ref2VA, `sparse_scope=joint-video` places multi-timestep `ref_img`
rows and target-video rows in one shared sparse domain with joint Reblock,
Top-K, and Reweight, while text and audio remain exact. If no visual reference
is present, it automatically uses the identical target-only path. Prefer
`topk_ratio` for `joint-video`; a fixed `topk_blocks` value covers a smaller
fraction after reference blocks are added. An unlinked target scope displays a
non-serialized one-click reminder for video-conditioned Ref2VA workflows.

`producer_chunk` is an optional Spark-node setting for the number of tokens
handled by each Q/K/V producer chunk. It defaults to `16384` and must be a
positive multiple of 64. Smaller chunks may lower temporary producer memory
but add chunk-call overhead. It is a Python runtime setting and does not
require rebuilding the bundled wheel.

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
`0.2.37+spark.h3.sm120.cu130.1`, and compiles only the selected architecture. One custom-node ZIP
can contain both wheels and the installer chooses the matching one.
