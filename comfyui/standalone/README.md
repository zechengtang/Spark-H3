# ComfyUI Spark-H3

Independent ComfyUI custom nodes for running Spark block-sparse attention with
ComfyUI's native MiniMax-H3 model.

## Supported configuration

- ComfyUI 0.30.0 or newer with native MiniMax-H3 support
- NVIDIA SM120 GPU (RTX 50 series)
- CUDA BF16 execution
- Linux; Windows wheels can be produced from the same backend source but require
  a separately validated Windows release asset

Model weights are not bundled. Use ComfyUI's native MiniMax-H3 model files.

## Installation

Install through ComfyUI Manager after the package is published to the registry.
For a ZIP or Git checkout, use the same Python interpreter that starts ComfyUI:

```bash
python -m pip install -r requirements.txt
python install.py
```

`install.py` first accepts an already-installed Spark backend, then searches the
package's `wheelhouse` and the pinned GitHub release for a compatible wheel. It
only compiles the pinned `comfy-kitchen` source when no wheel is available.
Source compilation requires Git, CMake, Ninja, a C++ compiler, and CUDA `nvcc`.
The installer deliberately retains ComfyUI's own torch-compatible Triton build
instead of asking pip to upgrade torch or Triton independently.

To install a wheel supplied out of band:

```bash
python install.py --wheel /path/to/comfy_kitchen-0.2.36+spark.h3.1-*.whl
```

## Nodes

- **MiniMax H3 Spark Attention (SM120)**
- **MiniMax H3 Sol Attention (Official Compatibility)**

Spark defaults to legacy midpoint directions, FP32 global anchors, exact
key-block budgets, query-granularity tails, and `q_reuse_k` reblocking.

## Backend wheel maintenance

From this custom-node directory:

```bash
python kernel_builder.py --output-dir wheelhouse
```

The builder pins `comfy-kitchen` v0.2.36, applies the bundled Spark patch, marks
the wheel as `0.2.36+spark.h3.1`, and compiles only SM120 by default.
