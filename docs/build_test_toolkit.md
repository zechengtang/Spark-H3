# Spark-H3 编译与验证 Kit

本工具分两步：仓库中的生成器制作独立 kit；目标机器使用 kit 编译
`spark-h3-kitchen` wheel、验证 Spark-H3，并打包安装用发布 ZIP。**每份 kit
只对应一个操作系统和一个 GPU 架构**。生成后，目标机器不需要本仓库的完整源码。

## 1. 在仓库中生成 kit

在 Spark-H3 仓库根目录运行：

```bash
python tools/build_spark_release_toolkit.py \
  --platform linux \
  --architecture sm120 \
  --publisher-id zechengtang
```

`--platform` 可选 `linux`、`windows`；`--architecture` 可选 `sm86`、
`sm89`、`sm120`。将两项改成目标机器的系统与 GPU 架构。生成器默认使用
`windows`，建议每次显式填写。它输出目录和同名 ZIP，例如
`dist/spark-release-toolkit-linux-sm120-cu130/` 及 `.zip`。可用 `--output`
指定位置；已有目标目录时，只有显式加 `--force` 才会覆盖。已有匹配
wheel 可用 `--wheel /path/to/wheel.whl` 加入 kit。

生成器读取当前仓库的节点源码、工作流、补丁和工具模板，因此**修改生成器
或模板后，必须重新生成 kit**；旧的 `dist/` 文件不会自动变化。kit 的
`release.json` 记录完整平台标签（如 `linux-x86_64`）、架构、版本和
CUDA 标签。命令行只使用简短的 `linux`／`windows`。

## 2. 将完整 kit ZIP 送到目标机器

在目标机器解压生成的 `spark-release-toolkit-*.zip`，进入其中的同名
目录。它包含 `ComfyUI-Spark-H3/` 节点源码和补丁、`build`／`validate`
入口、`bundle.py`、验证工作流和 `wheelhouse/`。**不要把最后生成的安装用
发布 ZIP 当作编译验证 kit**：安装用 ZIP 不含这套验证入口。

目标机器还需准备：匹配架构的 NVIDIA GPU、CUDA 13.0 工具链、
ComfyUI 及其 Python 环境、torch 2.12.x、兼容的 Triton 3.7.x、
`git` 和网络访问。编译脚本会从上游下载固定版本的 `comfy-kitchen`
及其子模块，使用 kit 内的补丁编译。Linux 还需 `g++`；Windows 需
Visual Studio 2022 x64 开发者环境中的 `cl` 和 `triton-windows`。
完整视频验证还需要 `ffprobe`、ComfyUI 模型权重及 Python 包
`requests`、`websocket-client`。模型文件名列在 kit 的 `README.md`。

## 3. 编译与验证

使用 **ComfyUI 的同一个 Python** 执行编译和验证。kit 的构建、验证
必须在其目标操作系统上运行；生成 kit 的机器可以是另一系统。

### Linux

在解压后的 kit 目录：

```bash
./build.sh --python /path/to/comfyui/bin/python
./validate.sh --python /path/to/comfyui/bin/python --mode audit
./validate.sh --python /path/to/comfyui/bin/python \
  --comfyui-root /path/to/ComfyUI --mode full
/path/to/comfyui/bin/python bundle.py
```

### Windows

在 Visual Studio x64 开发者 PowerShell 中，进入解压后的 kit 目录：

```powershell
./build.ps1 -Python C:\path\to\python.exe
./validate.ps1 -Python C:\path\to\python.exe -ComfyUIRoot C:\path\to\ComfyUI -Mode audit
./validate.ps1 -Python C:\path\to\python.exe -ComfyUIRoot C:\path\to\ComfyUI -Mode full
& C:\path\to\python.exe ./bundle.py
```

`audit` 校验 wheel 平台、GPU 架构、原生 `spark_attn` 和基本执行，并
生成 `results-audit/wheel-validation.json`；`bundle.py` 要求该报告
通过且 wheel 的 SHA-256 未变化。`full` 运行 Dense、Fidelity、Speed
LightX2V 测试；`compatibility` 是较短的同形状兼容性检查。已有匹配
wheel 时，可放入 `wheelhouse/`，跳过编译，仍需执行验证。仅需一个独立实现时，
Linux 编译可加 `--kitchen-implementation 0.2.37`，Windows 编译可加
`-KitchenImplementation 0.2.37`。

当前验证范围是 ComfyUI 0.38.x–0.40.x 和 0.42.x–0.44.x；所有环境默认验证
独立的 0.2.37 实现，也可显式选择 0.2.39。验证和 wheel 选择不读取宿主官方
`comfy-kitchen` 版本。更换 ComfyUI、独立实现、CUDA 或验证工作流时，应先更新
仓库里的工具源码和模板，再重新生成 kit。

## 4. 输出与边界

编译好的 wheel 存放在 `wheelhouse/`；验证记录存放在
`results-audit/`、`results-full/` 或 `results-compatibility/`；
`bundle.py` 输出 `release/ComfyUI-Spark-H3-<platform>-<arch>-cu130.zip`
及同名 JSON 校验报告。此发布 ZIP 用于安装，完整 kit ZIP 用于继续
编译和验证。

仓库测试可运行：

```bash
python -m pytest -q tests/test_spark_release_toolkit.py
```

该测试检查 kit 生成、平台选择和归档组装；实际 CUDA 编译与 GPU
推理结果须在目标机器上执行上述编译和验证命令取得。

使用 uv 开发环境时，也可在仓库根目录运行：

```bash
uv run --locked --extra test python -m pytest -q \
  tests/test_spark_release_toolkit.py
```
