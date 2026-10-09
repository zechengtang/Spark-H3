# Spark-H3 ComfyUI 安装指南

本指南适用于 Spark-H3 的 ComfyUI 独立安装包，目前发布版支持：

- ComfyUI 0.38.x 或 0.39.x，且已包含原生 MiniMax-H3 支持。
- Linux x86_64、Python 3.12+ 和 CUDA BF16。
- NVIDIA SM89（RTX 4090）或 SM120 GPU（RTX 50 系）。

安装包不包含 ComfyUI 本体和 MiniMax-H3 模型权重。

## 选择版本

本次发布提供三个按 GPU 架构和 CUDA 工具链区分的安装包：

| GPU | CUDA 工具链 | 安装包 |
| --- | --- | --- |
| SM120（RTX 50 系） | 12.8 | `ComfyUI-Spark-H3-<version>-cu128.zip` |
| SM120（RTX 50 系） | 13.0 | `ComfyUI-Spark-H3-<version>-cu130.zip` |
| SM89（RTX 4090） | 13.0 | `ComfyUI-Spark-H3-<version>-sm89-cu130.zip` |

请同时根据 GPU 架构和 `torch.version.cuda` 选择安装包。本次发布
不提供 SM89 CUDA 12.8 预编译包；该组合需要使用匹配的本机 CUDA
工具链运行 `install.py --source`。每个 ZIP 均同时支持 ComfyUI
0.38.x 和 0.39.x；安装器会根据当前 ComfyUI 环境自动选择
匹配的 backend wheel：

| ComfyUI 版本 | 后端基础版本 |
| --- | --- |
| 0.38.x | `comfy-kitchen 0.2.36+spark.h3.1` |
| 0.39.x | `comfy-kitchen 0.2.37+spark.h3.1` |

SM89 wheel 使用架构标识 `+spark.h3.sm89.1`，SM120 wheel 保留
`+spark.h3.1`。安装器会检测当前 GPU，只接受与架构匹配的 wheel；找不到时
从固定版本的 comfy-kitchen 源码编译，SM89 使用 `89`，SM120 使用 `120f`。

两种架构的后端版本不能混用。正常情况下不需要手动选择，直接运行
`install.py` 即可。

本次发布的 SM120 安装包提供 **CUDA 12.8** 和 **CUDA 13.0**
预编译 wheel；SM89 安装包提供 **CUDA 13.0** 预编译 wheel。
CUDA 12.9 以及 SM89 CUDA 12.8 需要使用本机 CUDA 工具链从源码编译：

| 本机 CUDA | 安装方式 |
| --- | --- |
| CUDA 13.0 | 按 GPU 架构使用 SM120 或 SM89 发布包内的预编译 wheel |
| CUDA 12.9 | 运行 `install.py --source`，使用本机 CUDA 编译 |
| CUDA 12.8 + SM120 | 使用 `-cu128.zip` 中的预编译 wheel |
| CUDA 12.8 + SM89 | 运行 `install.py --source`，使用本机 CUDA 编译 |

这里的 CUDA 版本指编译 `comfy-kitchen` 扩展时使用的 CUDA 工具链。

## 安装 ZIP

1. 停止正在运行的 ComfyUI。
2. 根据上表同时核对 GPU 架构和 `torch.version.cuda`，下载对应 ZIP。
3. 将 ZIP 直接解压到 `ComfyUI/custom_nodes`。
4. 使用**启动 ComfyUI 的同一个 Python**运行安装器。

```bash
cd /path/to/ComfyUI/custom_nodes
unzip /path/to/ComfyUI-Spark-H3-<version>-cu130.zip  # SM120 示例
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py
```

SM120 CUDA 12.8/13.0 和 SM89 CUDA 13.0 使用对应 ZIP 后直接运行
安装器。CUDA 12.9 或 SM89 CUDA 12.8 用户应显式执行源码编译：

```bash
/path/to/ComfyUI/.venv/bin/python \
  ComfyUI-Spark-H3/install.py --source
```

安装完成后重启 ComfyUI。RTX 4090 选择
**MiniMax H3 Spark Attention (SM89)**，RTX 50 系选择
**MiniMax H3 Spark Attention (SM120)**。

在 workflow 中将节点接在 MiniMax-H3 模型加载器和
`BasicGuider` 之间：

```text
UNETLoader -> MiniMax H3 Spark Attention (SM89 或 SM120) -> BasicGuider
```

## 手动指定 ComfyUI 版本

只有当自动检测失败，或者需要排查安装问题时，才需要手动指定：

```bash
# ComfyUI 0.38.x
/path/to/ComfyUI/.venv/bin/python \
  ComfyUI-Spark-H3/install.py --kitchen-base 0.2.36

# ComfyUI 0.39.x
/path/to/ComfyUI/.venv/bin/python \
  ComfyUI-Spark-H3/install.py --kitchen-base 0.2.37
```

如果已经单独下载了匹配的 wheel，可以通过 `--wheel` 指定：

```bash
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py \
  --kitchen-base 0.2.37 \
  --wheel /path/to/comfy_kitchen-0.2.37+spark.h3.sm89.1-<wheel-tag>.whl
```

## 常见问题

### 安装到了错误的 Python 环境

不要直接使用系统的 `python` 或 `pip`，除非 ComfyUI 本身就由该
Python 启动。可以在 ComfyUI 环境中检查当前版本：

```bash
/path/to/ComfyUI/.venv/bin/python -m pip show comfy-kitchen
```

### 找不到匹配的 wheel

安装器会先检查 ZIP 内的 wheel，再查找匹配的 Release wheel。SM120
CUDA 12.8/13.0 和 SM89 CUDA 13.0 应使用对应 ZIP；CUDA 12.9 或
SM89 CUDA 12.8 应直接使用 `--source`。源码编译需要：

- Git、CMake 和 Ninja。
- C++ 编译器。
- 与目标环境匹配的 CUDA 12.8、12.9 或 13.0 `nvcc`。

如果不希望自动进入源码编译，可在安装命令后加上
`--no-source-fallback`。

### 是否包含模型和 LoRA

不包含。MiniMax-H3 模型权重、LoRA 以及部分示例 workflow 需要的
第三方节点需要另行下载。模型放置方式和示例 workflow 说明见
[ComfyUI 节点文档](README.md)。
