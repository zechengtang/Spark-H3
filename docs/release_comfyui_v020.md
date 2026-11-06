# ComfyUI Spark-H3 v0.2.0

## 中文

v0.2.0 将 Spark-H3 与 ComfyUI 官方 `comfy-kitchen` 依赖解耦。Spark 原生后端现以独立的 `spark-h3-kitchen` 发行包安装，并使用 `spark_h3_kitchen` Python 命名空间。它可与 ComfyUI 官方 `comfy-kitchen` 并存，不再替换、降级或锁定官方包。

本版本支持 ComfyUI **0.38.x–0.40.x** 和 **0.42.x–0.44.x**；ComfyUI 0.41.x 不在支持范围内。每个平台包内置两个独立后端 wheel：`spark-h3-kitchen 0.2.37` 和 `0.2.39`。所有受支持的 ComfyUI 版本默认使用独立的 0.2.37 实现，高级用户可通过 `--kitchen-implementation 0.2.39` 显式选择 0.2.39。

从 v0.1.5 或更早版本升级时，安装器会识别曾以 `+spark.h3...` 本地版本标记覆盖安装的旧 `comfy-kitchen`，先恢复相同基础版本的官方包，再安装独立的 `spark-h3-kitchen`。已经是官方版本的 `comfy-kitchen` 不会被修改。

### 发布资产

| 平台 | GPU 架构 | 安装包 |
| --- | --- | --- |
| Linux x86_64 | SM86 · RTX 30 系 | `ComfyUI-Spark-H3-0.2.0-linux-x86_64-sm86-cu130.zip` |
| Linux x86_64 | SM89 · RTX 40 系 | `ComfyUI-Spark-H3-0.2.0-linux-x86_64-sm89-cu130.zip` |
| Linux x86_64 | SM120 · RTX 50 系 / RTX PRO Blackwell | `ComfyUI-Spark-H3-0.2.0-linux-x86_64-sm120-cu130.zip` |
| Windows x86_64 | SM86 · RTX 30 系 | `ComfyUI-Spark-H3-0.2.0-windows-x86_64-sm86-cu130.zip` |
| Windows x86_64 | SM89 · RTX 40 系 | `ComfyUI-Spark-H3-0.2.0-windows-x86_64-sm89-cu130.zip` |
| Windows x86_64 | SM120 · RTX 50 系 / RTX PRO Blackwell | `ComfyUI-Spark-H3-0.2.0-windows-x86_64-sm120-cu130.zip` |

wheel 使用 CUDA 13.0 编译。请同时根据操作系统和 GPU 架构选择 ZIP，并使用随发布提供的 `SHA256SUMS.txt` 校验。发布包不包含 ComfyUI、MiniMax-H3 模型权重或 LoRA。

## English

v0.2.0 decouples Spark-H3 from ComfyUI's official `comfy-kitchen` dependency. The native Spark backend is now installed as the separate `spark-h3-kitchen` distribution under the `spark_h3_kitchen` Python namespace. It can coexist with ComfyUI's official `comfy-kitchen` package and no longer replaces, downgrades, or pins that package.

This release supports ComfyUI **0.38.x–0.40.x** and **0.42.x–0.44.x**; ComfyUI 0.41.x is not supported. Each platform archive bundles isolated `spark-h3-kitchen` 0.2.37 and 0.2.39 wheels. All supported ComfyUI versions default to the isolated 0.2.37 implementation. Advanced users may explicitly select 0.2.39 with `--kitchen-implementation 0.2.39`.

When upgrading from v0.1.5 or earlier, the installer detects a legacy `comfy-kitchen` replacement carrying a `+spark.h3...` local version. It restores the clean official package at the same base version before installing the isolated `spark-h3-kitchen` backend. An already-clean official `comfy-kitchen` installation is left unchanged.

The wheels are built with CUDA 13.0. Select the ZIP matching both the operating system and GPU architecture, and verify it with the accompanying `SHA256SUMS.txt`. ComfyUI, MiniMax-H3 model weights, and LoRAs are not included.
