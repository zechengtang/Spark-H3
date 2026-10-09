# Spark-H3 ComfyUI 安装指南

本指南适用于 Spark-H3 的 ComfyUI 独立安装包，目前发布版支持：

- ComfyUI 0.38.x 或 0.39.x，且已包含原生 MiniMax-H3 支持。
- Linux x86_64、Python 3.12+ 和 CUDA BF16。
- NVIDIA SM120 GPU（RTX 50 系）。

安装包不包含 ComfyUI 本体和 MiniMax-H3 模型权重。

## 选择版本

ComfyUI 0.38.x 和 0.39.x 使用同一个
`ComfyUI-Spark-H3-<version>.zip`。安装器会根据当前 ComfyUI 环境
自动选择匹配的安装包：

| ComfyUI 版本 | 对应安装包 |
| --- | --- |
| 0.38.x | `comfy-kitchen 0.2.36+spark.h3.1` |
| 0.39.x | `comfy-kitchen 0.2.37+spark.h3.1` |

两个版本不能混用。正常情况下不需要手动选择，直接运行
`install.py` 即可。

本次发布只提供基于 **CUDA 13.0** 编译的预编译 wheel。CUDA 12.8
和 CUDA 12.9 环境理论上受支持，但没有对应的预编译 wheel，必须使用
本机 CUDA 工具链从源码编译：

| 本机 CUDA | 安装方式 |
| --- | --- |
| CUDA 13.0 | 使用发布包内或 Release 中的预编译 wheel |
| CUDA 12.9 | 运行 `install.py --source`，使用本机 CUDA 编译 |
| CUDA 12.8 | 运行 `install.py --source`，使用本机 CUDA 编译 |

这里的 CUDA 版本指编译 `comfy-kitchen` 扩展时使用的 CUDA 工具链。

## 安装 ZIP

1. 停止正在运行的 ComfyUI。
2. 从 Release 下载 `ComfyUI-Spark-H3-<version>.zip`。
3. 将 ZIP 直接解压到 `ComfyUI/custom_nodes`。
4. 使用**启动 ComfyUI 的同一个 Python**运行安装器。

```bash
cd /path/to/ComfyUI/custom_nodes
unzip /path/to/ComfyUI-Spark-H3-<version>.zip
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py
```

上面的默认命令适用于 CUDA 13.0 预编译版本。CUDA 12.8 或 CUDA 12.9
用户应显式执行源码编译：

```bash
/path/to/ComfyUI/.venv/bin/python \
  ComfyUI-Spark-H3/install.py --source
```

安装完成后重启 ComfyUI。节点列表中应出现
**MiniMax H3 Spark Attention (SM120)**。

在 workflow 中将节点接在 MiniMax-H3 模型加载器和
`BasicGuider` 之间：

```text
UNETLoader -> MiniMax H3 Spark Attention (SM120) -> BasicGuider
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
  --wheel /path/to/comfy_kitchen-0.2.37+spark.h3.1-<wheel-tag>.whl
```

## 常见问题

### 安装到了错误的 Python 环境

不要直接使用系统的 `python` 或 `pip`，除非 ComfyUI 本身就由该
Python 启动。可以在 ComfyUI 环境中检查当前版本：

```bash
/path/to/ComfyUI/.venv/bin/python -m pip show comfy-kitchen
```

### 找不到匹配的 wheel

安装器会先检查 ZIP 内的 wheel，再查找匹配的 Release wheel。当前预编译
wheel 面向 CUDA 13.0；CUDA 12.8/12.9 不应等待自动匹配，而应直接使用
`--source`。源码编译需要：

- Git、CMake 和 Ninja。
- C++ 编译器。
- 与目标环境匹配的 CUDA 12.8、12.9 或 13.0 `nvcc`。

如果不希望自动进入源码编译，可在安装命令后加上
`--no-source-fallback`。

### 是否包含模型和 LoRA

不包含。MiniMax-H3 模型权重、LoRA 以及部分示例 workflow 需要的
第三方节点需要另行下载。模型放置方式和示例 workflow 说明见
[ComfyUI 节点文档](README.md)。
