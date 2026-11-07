# ComfyUI Spark-H3 v0.2.0

> [!IMPORTANT]
> **推荐使用默认的 `spark-h3-kitchen` 0.2.37：它在本版本发布测试中的速度和稳定性均优于 0.2.39。**
> **We recommend the default `spark-h3-kitchen` 0.2.37 for production use because it delivered better speed and stability in this release's testing.**

## 中文

v0.2.0 将 Spark-H3 与 ComfyUI 官方 `comfy-kitchen` 依赖解耦。Spark 原生后端现以独立的 `spark-h3-kitchen` 发行包安装，并使用 `spark_h3_kitchen` Python 命名空间。它可与 ComfyUI 官方 `comfy-kitchen` 并存，不再替换、降级或锁定官方包。

此前我们过于乐观地希望在同一版本中同时完成 SelfLift-Avatar 项目的分块显存优化功能适配、Kitchen attention 潜在性能劣化的调查，以及新版 ComfyUI 的适配，因此不应该地延迟了兼容更新，对此我们深表抱歉。v0.2.0 优先完成底层解耦，让后续功能开发和性能调查不再阻塞基础兼容性更新。

本版本支持 ComfyUI **0.38.x–0.40.x** 和 **0.42.x–0.44.x**。除 Windows SM89（RTX 40 系）安装包当前仅内置 `spark-h3-kitchen 0.2.37` 外，其余平台包均内置 `0.2.37` 和 `0.2.39` 两个独立后端 wheel。**我们明确推荐发布版使用默认的 0.2.37 实现；在本版本的发布测试中，它的速度和稳定性均优于 0.2.39。** 包含 0.2.39 wheel 的平台仍可通过 `--kitchen-implementation 0.2.39` 显式选择 0.2.39。

在 Spark 节点中选择 `dense_backend=kitchen` 时，dense 阶段会直接使用安装包内置 `spark-h3-kitchen` 提供的 Kitchen 实现，**无需再连接外部的“模型注意力后端”节点**。

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

We had previously been too optimistic about adapting chunked memory-saving support for the SelfLift-Avatar project, investigating a potential Kitchen attention performance regression, and adapting newer ComfyUI releases in one combined update. That delayed the compatibility release longer than it should have, and we apologize. v0.2.0 prioritizes the underlying decoupling so future feature work and performance investigations no longer block baseline compatibility updates.

This release supports ComfyUI **0.38.x–0.40.x** and **0.42.x–0.44.x**. The Windows SM89 (RTX 40 series) archive currently bundles only `spark-h3-kitchen` 0.2.37; every other platform archive bundles isolated 0.2.37 and 0.2.39 wheels. **We explicitly recommend the default 0.2.37 implementation for production use because it delivered better speed and stability in this release's testing.** Platforms that include the 0.2.39 wheel may still select it explicitly with `--kitchen-implementation 0.2.39`.

When `dense_backend=kitchen` is selected in the Spark node, dense stages directly use the Kitchen implementation provided by the bundled `spark-h3-kitchen` package, so **no external model attention backend node needs to be connected**.

When upgrading from v0.1.5 or earlier, the installer detects a legacy `comfy-kitchen` replacement carrying a `+spark.h3...` local version. It restores the clean official package at the same base version before installing the isolated `spark-h3-kitchen` backend. An already-clean official `comfy-kitchen` installation is left unchanged.

The wheels are built with CUDA 13.0. Select the ZIP matching both the operating system and GPU architecture, and verify it with the accompanying `SHA256SUMS.txt`. ComfyUI, MiniMax-H3 model weights, and LoRAs are not included.
