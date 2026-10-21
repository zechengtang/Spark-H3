# Spark-H3 ComfyUI 中文安装指南

本指南适用于 Spark-H3 ComfyUI 独立安装包。普通用户按照“快速安装”操作即可；手动指定 wheel、源码构建和实验 CUDA 通道集中在文末的“高级用法”。

## 支持范围

- ComfyUI 0.38.x、0.39.x、0.40.x 或 0.42.x，且已包含原生 MiniMax-H3 支持。
- Linux 或 Windows x86_64。
- Python 3.12+、支持 CUDA 的 PyTorch（`torch.version.cuda >= 13.0`），以及支持 CUDA BF16 的 NVIDIA GPU。安装器不检查 PyTorch 版本号。
- SM86：GeForce RTX 30 系。
- SM89：GeForce RTX 40 系。
- SM120：GeForce RTX 50 系及 RTX PRO 5000/6000 Blackwell。

安装包**不包含** ComfyUI、MiniMax-H3 模型权重、LoRA 或第三方工作流节点。CUDA 12.8 不属于 v0.1.5 正式发布范围。

0.1.5 安装包内含对应平台和架构的 `comfy-kitchen 0.2.36`、`0.2.37`、`0.2.39`
wheel。安装器按当前 ComfyUI 版本选择对应 wheel。选择安装包时以 ZIP 内的
`spark_h3_build.json` 和 `wheelhouse/` 为准。

通过 Comfy Registry 安装时，Registry 包不附带 wheel。ComfyUI Manager 会执行
包内的 `install.py`，由安装器依据当前 comfy-kitchen 版本、系统、GPU 架构和
CUDA 版本，从 `spark_h3_build.json` 指定的 GitHub Release 平台 ZIP 中提取
匹配的 Kitchen wheel。此方式安装时需要访问 GitHub Releases；离线环境可手动
运行 `install.py --wheel /path/to/匹配的.whl`。

## 快速安装

### 1. 选择正确的 ZIP

请同时按照操作系统和 GPU 架构选择安装包：

| GPU | 架构 | 平台 | 0.1.5 安装包 |
| --- | --- | --- | --- |
| GeForce RTX 40 系 | SM89 | Windows | `ComfyUI-Spark-H3-0.1.5-windows-x86_64-sm89-cu130.zip` |
| GeForce RTX 50 系、RTX PRO 5000/6000 Blackwell | SM120 | Windows | `ComfyUI-Spark-H3-0.1.5-windows-x86_64-sm120-cu130.zip` |
| GeForce RTX 50 系、RTX PRO 5000/6000 Blackwell | SM120 | Linux | `ComfyUI-Spark-H3-0.1.5-linux-x86_64-sm120-cu130.zip` |

正式发布安装时，文件名中的以下部分必须与本机环境一致：

- `linux` 或 `windows`：操作系统。
- `sm86`、`sm89` 或 `sm120`：GPU 架构。
- `cu130`：wheel 以 CUDA 13.0 构建。

不同操作系统或 GPU 架构的 ZIP 和后端 wheel 不能混用。安装器只检查 PyTorch CUDA ≥ 13.0，不限制 PyTorch 版本号；较新 CUDA 版本可直接尝试包内 cu130 wheel，无需额外参数。能安装不代表该 PyTorch/CUDA 组合已完成完整工作流验证。

### 2. 关闭 ComfyUI 并解压

先完全关闭正在运行的 ComfyUI，再把 ZIP 解压到 `ComfyUI/custom_nodes/`。解压后应直接看到：

```text
ComfyUI/
├── main.py
└── custom_nodes/
    └── ComfyUI-Spark-H3/
        └── install.py
```

Windows 便携版的正确目录是：

```text
ComfyUI_windows_portable\
├── python_embeded\python.exe
└── ComfyUI\
    └── custom_nodes\
        └── ComfyUI-Spark-H3\
            └── install.py
```

不要形成两层同名目录，例如：

```text
custom_nodes/ComfyUI-Spark-H3/ComfyUI-Spark-H3/install.py
```

如果出现这种情况，请把内层 `ComfyUI-Spark-H3` 的内容上移一层。

### 3. 使用 ComfyUI 自己的 Python 安装

必须使用**启动当前 ComfyUI 的同一个 Python**执行 `install.py`。不要直接双击 `install.py`，也不要随意使用系统的 `python` 或 `pip`。

#### Windows 便携版

在 `ComfyUI_windows_portable` 目录打开 PowerShell：

```powershell
.\python_embeded\python.exe .\ComfyUI\custom_nodes\ComfyUI-Spark-H3\install.py
```

#### Windows `.venv` 虚拟环境

在包含 `.venv` 和 `custom_nodes` 的 ComfyUI 根目录打开 PowerShell：

```powershell
.\.venv\Scripts\python.exe .\custom_nodes\ComfyUI-Spark-H3\install.py
```

#### Linux `.venv` 虚拟环境

在 ComfyUI 根目录打开终端：

```bash
./.venv/bin/python ./custom_nodes/ComfyUI-Spark-H3/install.py
```

使用启动器或自定义环境时，请从启动器设置或 ComfyUI 启动日志中找到实际使用的 Python 路径，再用它执行 `install.py`。

### 4. 重启并打开工作流

安装命令完成且没有报错后：

1. 完全关闭并重新启动 ComfyUI，仅刷新浏览器页面不够。
2. 将安装包内 `workflows` 目录中的 JSON 拖入 ComfyUI 画布。
3. 按工作流说明另行准备模型、LoRA 和可选第三方节点。

安装器会根据 ComfyUI 版本自动选择匹配的后端：

| ComfyUI 版本 | comfy-kitchen 后端 |
| --- | --- |
| ComfyUI 0.38.x | `0.2.36+spark.h3.<架构>.cu130.1` |
| ComfyUI 0.39.x 或 0.40.x | `0.2.37+spark.h3.<架构>.cu130.1` |
| ComfyUI 0.42.x | `0.2.39+spark.h3.<架构>.cu130.1` |

每份安装包均内置上表三个后端版本的对应 wheel。

使用 W6A8 权重时，建议使用 ComfyUI 0.39.2 或更新的 0.40.x。两者配套的 comfy-kitchen 0.2.37 已适配 W6A8。

## 第一次使用

### 选择对应节点

- RTX 30 系：**MiniMax H3 Spark Attention (SM86)**。
- RTX 40 系：**MiniMax H3 Spark Attention (SM89)**。
- RTX 50 系及 RTX PRO Blackwell：**MiniMax H3 Spark Attention (SM120)**。

将对应节点接在 MiniMax-H3 模型加载器与 `BasicGuider` 或采样器之间：

```text
模型加载器 → MiniMax H3 Spark Attention → BasicGuider / 采样器
```

### 推荐初始设置

第一次使用时建议保持：

- `mode=fidelity`：前 20% 的采样调用使用 dense attention，之后使用 Spark，在保真度和速度之间取得平衡。
- `topk_mode=topk_ratio`、`topk_ratio=0.2`：精确计算约 20% 的候选视频块。
- `dense_backend=inherit`：沿用模型或 ComfyUI 当前的注意力设置。
- `producer_chunk=16384`：Spark producer 每批处理的 token 数。必须是 64 的正整数倍；除非显存压力测试表明更小值有帮助，否则保持默认值。

确认能够正常生成后，可以复制一份工作流再尝试：

- `mode=speed`：关闭 dense warmup，从第一次模型调用开始使用 Spark，速度更快。
- `mode=dense`：完全绕过 Spark，用于排错或建立性能对照。
- `mode=custom`：保留手动设置的 warmup 参数。

`kitchen` 和 `sage` dense 后端仍属于实验选项，尚未完成完整的端到端质量回归测试。`sage` 还需要单独安装 `sageattention`。

双采或两阶段工作流的高分辨率第二阶段建议使用 `mode=speed`，关闭 dense warmup。节点中的一键切换提示只用于界面交互，不会写入工作流 JSON。

`producer_chunk` 只改变 Python 运行时的 Q/K/V producer 分块，不改变 wheel ABI，也不需要重新编译 comfy-kitchen。减小它可能降低临时显存，但会增加分块调用开销。

## 配合 KJNodes 降低显存占用

Spark Attention 可以配合 ComfyUI-KJNodes 提供的以下节点：

- `MiniMax H3 Low VRAM Attention`
- `MiniMax H3 Chunk FeedForward`

推荐的 MODEL 连接顺序为：

```text
模型加载器 → Low VRAM Attention → Chunk FeedForward → Spark Attention → 采样器
```

`Low VRAM Attention` 必须放在 Spark 前面。分块设置越小，通常越节省显存，但也可能增加执行开销。KJNodes 需要单独安装，不包含在 Spark-H3 安装包中。

## 常见问题

### 重启后没有看到 Spark 节点

依次检查：

1. 是否完全重启了 ComfyUI，而不只是刷新网页。
2. `install.py` 是否直接位于 `ComfyUI/custom_nodes/ComfyUI-Spark-H3/`。
3. 安装时是否使用了启动当前 ComfyUI 的 Python。
4. ComfyUI 启动终端中是否出现红色导入错误。

可以用同一个 Python 检查后端是否安装成功：

```bash
/path/to/ComfyUI/.venv/bin/python -m pip show comfy-kitchen
```

Windows 便携版可以运行：

```powershell
.\python_embeded\python.exe -m pip show comfy-kitchen
```

### 提示找不到匹配的 wheel

检查以下项目：

- ZIP 是否与 Linux 或 Windows 匹配。
- `sm86`、`sm89` 或 `sm120` 是否与 GPU 匹配。
- 文件名是否包含 `cu130`。
- ComfyUI 是否为 0.38.x、0.39.x、0.40.x 或 0.42.x。
- 安装命令是否使用了 ComfyUI 自己的 Python。

安装器会先检查 ZIP 内的 wheel，再查找匹配的 Release wheel。若不希望在找不到 wheel 时自动尝试源码构建，可添加：

```bash
/path/to/ComfyUI/.venv/bin/python \
  ComfyUI-Spark-H3/install.py --no-source-fallback
```

Windows 用户请把前面的 Python 路径替换为当前 ComfyUI 实际使用的解释器。

### 运行时报错或结果异常

若仅 `fidelity` 模式出现显存不足（OOM），而相同工作流的 `dense` 和 `speed` 模式均可运行，可尝试在**启动 ComfyUI 时**加入 `--disable-cuda-malloc`，作为临时规避办法；这是 ComfyUI 启动参数，不是 Spark 节点参数。

1. 将 `mode` 切换为 `dense`，确认原始工作流能够运行。
2. 恢复 `mode=fidelity`、`dense_backend=inherit` 和默认 Top-K 设置。
3. 如果使用 KJNodes，确认 `Low VRAM Attention` 位于 Spark 前面。
4. 查看 ComfyUI 启动终端中的完整报错，而不只看网页弹窗。

### 加速效果不明显

- 只比较同一工作流、同一输入和同一 GPU 的采样耗时。
- 模型加载、VAE 解码和视频编码不属于 Spark 采样加速范围。
- 检查运行日志，确认没有因为不支持的输入回退到 dense attention。
- 可以尝试 `mode=speed`，但应同时检查输出质量。
- 建议使用 ComfyUI 0.39.2 或更新的 0.40.x。

### 是否包含模型、LoRA 和第三方节点

不包含。MiniMax-H3 模型权重、LoRA 以及部分示例工作流需要的第三方节点必须另行下载。模型放置方式和示例工作流说明见 [ComfyUI 节点文档](README.md)。

## 高级用法

以下内容仅用于自动检测失败、特殊 wheel 标签、开发验证或本地适配。普通安装不需要这些参数。

### 手动指定 comfy-kitchen 基础版本

```bash
# ComfyUI 0.38.x
/path/to/ComfyUI/.venv/bin/python \
  ComfyUI-Spark-H3/install.py --kitchen-base 0.2.36

# ComfyUI 0.39.x 或 0.40.x
/path/to/ComfyUI/.venv/bin/python \
  ComfyUI-Spark-H3/install.py --kitchen-base 0.2.37
```

### 手动指定 wheel

```bash
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py \
  --kitchen-base 0.2.37 \
  --wheel /path/to/comfy_kitchen-0.2.37+spark.h3.sm89.cu130.1-<wheel-tag>.whl
```

提供的 wheel 必须与操作系统、Python wheel tag、GPU 架构和 comfy-kitchen 版本一致。PyTorch CUDA 高于 13.0 时，安装器可直接选用 cu130 wheel。

### 在较新的 PyTorch 或 CUDA 环境中试用 cu130 wheel

安装包内已有对应架构和 comfy-kitchen 版本的 cu130 wheel 时，可以运行：

```bash
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py
```

也可同时传入 `--wheel /path/to/comfy_kitchen-...cu130...whl`。安装器会核对 wheel 的操作系统、Python 标签、GPU 架构和 comfy-kitchen 版本，并在安装后检查原生 `spark_attn` 是否可导入；不会修改 PyTorch。若包内没有匹配 wheel，安装器会尝试查找对应 Release wheel。安装成功只表示基本加载检查通过，完整工作流和性能仍需在该环境实测。

### CUDA 13.0 源码构建

预编译 wheel 不适用于当前 Python wheel tag 时，可以显式从源码构建：

```bash
/path/to/ComfyUI/.venv/bin/python \
  ComfyUI-Spark-H3/install.py --source
```

源码构建需要：

- Git、CMake 和 Ninja。
- 可用的 C++ 编译器。
- 与目标环境匹配的 `nvcc`。

CUDA 13.0 源码构建目标为：SM86 使用 `86`，SM89 使用 `89`，SM120 使用 `120f`。

### CUDA 12.8 实验通道

CUDA 12.8 暂不作为正式安装包发布。开发者可以显式提供本地 CU128 wheel：

```bash
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py \
  --experimental-cuda \
  --wheel /path/to/cu128/comfy_kitchen-<version>+spark.h3.sm120.cu128.1-<wheel-tag>.whl
```

也可以显式从源码构建：

```bash
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py \
  --experimental-cuda --source
```

SM120 在 CUDA 12.8 下使用 `120a` 架构目标。该通道仅用于本地适配和正确性验证，其性能不能代表 CUDA 13.0 正式发布包。

## 发布包与后端身份说明

- 每个正式 ZIP 同时支持 ComfyUI 0.38.x、0.39.x 和 0.40.x。
- ZIP 文件名和包内 `spark_h3_build.json` 会记录目标平台。
- Linux 包只接受 `linux_x86_64` 或兼容的 manylinux wheel；Windows 包只接受 `win_amd64` wheel。
- 正式后端身份分别为 `+spark.h3.sm86.cu130.1`、`+spark.h3.sm89.cu130.1` 和 `+spark.h3.sm120.cu130.1`。
- 安装器会在安装后端前检查操作系统、GPU 架构、CUDA 版本和 wheel 平台，不同架构的后端不能混用。
