# ComfyUI Spark-H3 v0.2.0

## 中文

v0.2.0 将 Spark-H3 与 ComfyUI 官方 `comfy-kitchen` 依赖解耦。Spark 原生后端现以独立的 `spark-h3-kitchen` 发行包安装，并使用 `spark_h3_kitchen` Python 命名空间。它可与 ComfyUI 官方 `comfy-kitchen` 并存，不再替换、降级或锁定官方包。

### 为什么发布 v0.2.0

社区在 [Issue #5](https://github.com/zechengtang/Spark-H3/issues/5) 中指出，旧的集成方式会把 Spark 后端安装到 ComfyUI 官方 `comfy-kitchen` 中：ComfyUI 升级时可能直接覆盖 Spark 改动，而为了恢复 Spark 手动覆盖新版 Kitchen，又可能使依赖新接口的 ComfyUI 功能失效。这让 Spark-H3、ComfyUI 和 `comfy-kitchen` 的升级节奏被不必要地绑在一起。

v0.2.0 将 Spark 原生扩展放入独立发行包，使 ComfyUI 可以继续管理自己的官方 Kitchen，Spark-H3 也可以独立更新和验证后端。这项改造发生在包和安装器内部：节点用法保持一致，现有工作流仍然兼容。

此前我们过于乐观地希望在同一版本中同时完成 SelfLift-Avatar 分块显存优化、Kitchen attention 潜在性能劣化的调查，以及新版 ComfyUI 的适配，因此不应该地延迟了兼容更新，对此我们深表抱歉。v0.2.0 优先完成底层解耦，让后续功能开发和性能调查不再阻塞基础兼容性更新。

本版本支持 ComfyUI **0.38.x–0.40.x** 和 **0.42.x–0.44.x**。每个平台包内置两个独立后端 wheel：`spark-h3-kitchen 0.2.37` 和 `0.2.39`。所有受支持的 ComfyUI 版本默认使用独立的 0.2.37 实现，用户可通过 `--kitchen-implementation 0.2.39` 显式选择 0.2.39。

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

### Why v0.2.0

As discussed in [Issue #5](https://github.com/zechengtang/Spark-H3/issues/5), the previous integration installed the Spark backend into ComfyUI's official `comfy-kitchen` package. A ComfyUI upgrade could overwrite the Spark changes; conversely, restoring Spark by replacing a newer official Kitchen build could break ComfyUI features that depend on newer Kitchen APIs. This unnecessarily coupled the release cadence of Spark-H3, ComfyUI, and `comfy-kitchen`.

v0.2.0 moves the native Spark extension into a separate distribution. ComfyUI can continue managing its official Kitchen package while Spark-H3 updates and validates its backend independently. The change is internal to the package and installer: node usage remains the same, and existing workflows remain compatible.

We had previously been too optimistic about completing SelfLift-Avatar chunked-memory work, investigating a potential Kitchen attention performance regression, and adapting newer ComfyUI releases in one combined update. That delayed the compatibility release longer than it should have, and we apologize. v0.2.0 prioritizes the underlying decoupling so future feature work and performance investigations no longer block baseline compatibility updates.

This release supports ComfyUI **0.38.x–0.40.x** and **0.42.x–0.44.x**. Each platform archive bundles isolated `spark-h3-kitchen` 0.2.37 and 0.2.39 wheels. All supported ComfyUI versions default to the isolated 0.2.37 implementation. Users may explicitly select 0.2.39 with `--kitchen-implementation 0.2.39`.

When upgrading from v0.1.5 or earlier, the installer detects a legacy `comfy-kitchen` replacement carrying a `+spark.h3...` local version. It restores the clean official package at the same base version before installing the isolated `spark-h3-kitchen` backend. An already-clean official `comfy-kitchen` installation is left unchanged.

The wheels are built with CUDA 13.0. Select the ZIP matching both the operating system and GPU architecture, and verify it with the accompanying `SHA256SUMS.txt`. ComfyUI, MiniMax-H3 model weights, and LoRAs are not included.
