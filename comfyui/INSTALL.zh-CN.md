# Spark-H3 ComfyUI 安装指南

本指南适用于 Spark-H3 的 ComfyUI 独立安装包，目前发布版支持：

- ComfyUI 0.38.x 或 0.39.x，且已包含原生 MiniMax-H3 支持。
- Linux x86_64、Python 3.12+、PyTorch CUDA 13.0+ 和 CUDA BF16。
- NVIDIA SM89（RTX 4090）或 SM120 GPU（RTX 50 系）。

安装包不包含 ComfyUI 本体和 MiniMax-H3 模型权重。

## 选择版本

本次发布提供两个按 GPU 架构区分的 Linux x86_64 CUDA 13.0 安装包：

| GPU | CUDA 工具链 | 安装包 |
| --- | --- | --- |
| SM120（RTX 50 系） | 13.0 | `ComfyUI-Spark-H3-<version>-linux-x86_64-sm120-cu130.zip` |
| SM89（RTX 4090） | 13.0 | `ComfyUI-Spark-H3-<version>-linux-x86_64-sm89-cu130.zip` |

请同时根据 GPU 架构和 `torch.version.cuda` 选择安装包。CUDA 12.8
已从发布计划中移除：虽然本地 Spark 内核能够编译和执行，但 ComfyUI
会全局禁用该环境下的 comfy-kitchen CUDA backend，造成严重的端到端
性能回退。因此 CU128 不得作为发布资产，也不得用于正式性能声明；这不
影响开发者显式使用本地 wheel 或源码构建继续进行适配和正确性验证。

每个正式 ZIP 均同时支持 ComfyUI
0.38.x 和 0.39.x；安装器会根据当前 ComfyUI 环境自动选择
匹配的 backend wheel：

| ComfyUI 版本 | 后端基础版本 |
| --- | --- |
| 0.38.x | `comfy-kitchen 0.2.36+spark.h3.<架构>.cu130.1` |
| 0.39.x | `comfy-kitchen 0.2.37+spark.h3.<架构>.cu130.1` |

正式 wheel 使用完整的架构和 CUDA 身份：SM89 为
`+spark.h3.sm89.cu130.1`，SM120 为 `+spark.h3.sm120.cu130.1`。
实验性 SM120 CU128 wheel 使用 `+spark.h3.sm120.cu128.1`，因此即使不同
CUDA 版本的 wheel 位于同一目录，安装器也不会交叉选择。找不到匹配 wheel
时，固定版本的 comfy-kitchen 源码编译在 SM89 使用 `89`，在 SM120 CU130
使用 `120f`。

ZIP 文件名和包内 `spark_h3_build.json` 同时记录目标平台。Linux 包只接受
`linux_x86_64` 或兼容的 manylinux wheel，Windows 包只接受 `win_amd64`
wheel；安装器也会在安装 backend 前检查当前操作系统。Windows 完成独立
编译和实机验证后使用 `windows-x86_64` 文件名，不能与 Linux ZIP 混用。

两种架构的后端版本不能混用。正常情况下不需要手动选择，直接运行
`install.py` 即可。

本次发布仅提供 **CUDA 13.0** 预编译 wheel：

| 本机 CUDA | 安装方式 |
| --- | --- |
| CUDA 13.0 | 按 GPU 架构使用 SM120 或 SM89 发布包内的预编译 wheel |
| CUDA 12.8 | 不正式发布；仅允许显式启用的本地实验构建/安装 |
| 其他低于 13.0 的版本 | 不属于当前发布支持范围 |

这里的 CUDA 版本指编译 `comfy-kitchen` 扩展时使用的 CUDA 工具链。

## 安装 ZIP

1. 停止正在运行的 ComfyUI。
2. 根据上表同时核对 GPU 架构和 `torch.version.cuda`，下载对应 ZIP。
3. 将 ZIP 直接解压到 `ComfyUI/custom_nodes`。
4. 使用**启动 ComfyUI 的同一个 Python**运行安装器。

```bash
cd /path/to/ComfyUI/custom_nodes
unzip /path/to/ComfyUI-Spark-H3-<version>-linux-x86_64-sm120-cu130.zip  # SM120 示例
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py
```

SM120 CUDA 13.0 和 SM89 CUDA 13.0 使用对应 ZIP 后直接运行安装器。
`--source` 只用于在受支持的 CUDA 13.0+ 环境中重建 backend，例如当前
Python wheel tag 没有预编译产物时：

```bash
/path/to/ComfyUI/.venv/bin/python \
  ComfyUI-Spark-H3/install.py --source
```

### CUDA 12.8 实验通道

CU128 不会被默认安装器自动发现或从 Release 下载。为修复和验证该路径，
开发者可以显式提供本地 CU128 wheel：

```bash
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py \
  --experimental-cuda \
  --wheel /path/to/cu128/comfy_kitchen-<version>+spark.h3.sm120.cu128.1-<wheel-tag>.whl
```

也可以显式从源码构建；SM120 在 CUDA 12.8 下会使用 `120a`：

```bash
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py \
  --experimental-cuda --source
```

该开关只保留研发能力，不代表正式支持。安装器会打印性能回退警告，且不会
自动选择正式 CU130 包中的 wheel。

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
  --wheel /path/to/comfy_kitchen-0.2.37+spark.h3.sm89.cu130.1-<wheel-tag>.whl
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
和 SM89 的正式安装均应使用 CUDA 13.0 对应 ZIP。CUDA 12.8 只能通过
`--experimental-cuda` 配合显式 `--wheel` 或 `--source` 进入研发通道，
不能由默认安装流程自动选择。源码编译需要：

- Git、CMake 和 Ninja。
- C++ 编译器。
- 与目标环境匹配的 `nvcc`；正式构建使用 CUDA 13.0+，实验 CU128
  SM120 构建使用 CUDA 12.8 的 `120a` 目标。

如果不希望自动进入源码编译，可在安装命令后加上
`--no-source-fallback`。

### 是否包含模型和 LoRA

不包含。MiniMax-H3 模型权重、LoRA 以及部分示例 workflow 需要的
第三方节点需要另行下载。模型放置方式和示例 workflow 说明见
[ComfyUI 节点文档](README.md)。
