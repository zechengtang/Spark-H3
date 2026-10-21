# Spark-H3 Windows 打包与验证工具

入口是 [`tools/build_spark_release_toolkit.py`](../tools/build_spark_release_toolkit.py)。它从当前仓库源码生成独立的 Windows/CUDA 13.0 工具包，可选择 `sm86`、`sm89` 或 `sm120`。架构选择贯穿编译参数、wheel 版本、ComfyUI 节点名、GPU capability、cubin 审计与运行日志断言。

```bash
python tools/build_spark_release_toolkit.py \
  --architecture sm86 \
  --publisher-id YOUR_COMFY_REGISTRY_PUBLISHER
```

将 `sm86` 换成 `sm89` 或 `sm120` 即可生成对应包。默认输出在 `dist/spark-release-toolkit-windows-<架构>-cu130/`，同名 `.zip` 便于传到 Windows。若已有 Windows wheel，可重复传入 `--wheel 路径`，工具会先检查文件名中的架构、CUDA 标签和平台。默认不会覆盖已有输出；需要重新生成时加 `--force`。

Windows 上解压后，使用 **同一个 ComfyUI Python** 执行：

```powershell
.\build.ps1 -Python C:\path\to\python.exe
.\validate.ps1 -ComfyUIRoot C:\path\to\ComfyUI -Python C:\path\to\python.exe -Mode full
& C:\path\to\python.exe .\bundle.py
```

`full` 已包含 wheel 审计，`bundle.py` 默认读取 `results-full/wheel-validation.json`。只需要检查 wheel 时可单独运行 `-Mode audit`，此时打包脚本会读取 `results-audit/wheel-validation.json`。若用 `-Results` 指定其他目录，打包时通过 `--audit-report` 指向该目录中的 `wheel-validation.json`。

默认使用 LightX2V v1.0 768p 8-step LoRA。先把对应权重放入 ComfyUI 的 `models/loras/`，再运行：

```powershell
.\validate.ps1 -ComfyUIRoot C:\path\to\ComfyUI -Python C:\path\to\python.exe -Mode full -Steps 8 -LoraName minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors -Results .\results-full-8step
```

`-Steps` 同时修改采样器和 Spark 节点，`-LoraName` 只修改 LoRA 文件名。Fidelity 按 20% 比例向上取整设置 dense 预热：4 步为 1 步，8 步为 2 步；不同步数或 LoRA 的加速比不能直接横比。

`full` 模式默认使用 **VDN selected prompt 06**，运行结束后还会生成 `outputs/dense_fidelity_speed_side_by_side.mp4`（左至右 Dense、Fidelity、Speed），并在 `summary.json` 的 `comparison_video` 字段记录路径、SHA-256 和媒体属性。完整验证需要 `ffmpeg`（含 `drawtext` 与 `libx264`）和 `ffprobe`。

完整验证结束时，控制台会显示 `SPEEDUP_VS_DENSE fidelity=...x speed=...x`；`speedup_report.json` 单独记录 Dense/Fidelity/Speed 的正式采样器耗时、相对 Dense 的加速倍数和节省时间百分比。每个变体的 short warmup 耗时不计入这些数字。

`build.ps1` 默认可为 comfy-kitchen 0.2.36、0.2.37 和 0.2.39 编译本机架构 wheel。`validate.ps1` 有 `audit`、`full` 两种模式：分别是原生 wheel 审计和 LightX2V 三变体完整测速。脚本支持 ComfyUI 0.38.x（comfy-kitchen 0.2.36）、0.39.x/0.40.x（0.2.37）及 0.42.x（0.2.39），并检查 `comfyui_version.py` 和 `requirements.txt`。它隔离加载对应 wheel，不覆盖 ComfyUI 已安装的 comfy-kitchen。`bundle.py` 将已编译的 wheel 与节点源码组成 ComfyUI 包，并产生 SHA256 报告。正式发布时需使用自己的 publisher ID 重新生成工具包。

独立的短兼容测试已移除：`full` 在同一个 ComfyUI 进程中，为 Dense/Fidelity/Speed 各做一次 1/3/1 步 short warmup，再立即运行各自正式 8 步推理。无需单独运行预热。

验证 prompt 使用当前 Spark 节点的 `mode=dense/fidelity/speed`、`warmup_mode=warmup_steps`；默认 8 步对应的 `warmup_steps=0/2/0`。工具生成时复制已维护的 LightX2V 8 步工作流。生成过程的测试会检查 prompt 输入名称与当前节点 `patch()` 签名一致，避免旧版 `enabled` 参数混入。

前置检查要求 torch CUDA 13.0、目标 GPU 架构和可用的 Triton 3.7 JIT；缺少 Triton 会直接失败。工具还校验规划源码哈希与关键模块的直接 Triton 导入，拒绝把无 Triton 的 Torch 规划回退结果计入发布测速。三变体共用一个 ComfyUI 进程，每个变体正式计时前各跑一次同尺寸 short warmup（Dense/Speed 各 1 步，8 步 Fidelity 为 3 步），warmup 耗时不计入加速比。默认保持 DynamicVRAM 并使用 `--reserve-vram 1.5`；可用 `SPARK_EXTRA_COMFY_ARGS` 改写启动参数。输出保存在工具包内的 `results-*` 目录。

`nvidia-smi` 所显示的 CUDA 版本是驱动支持上限，可以高于 PyTorch 的 `torch.version.cuda=13.0`。已编译 wheel 的验证不要求 `nvcc`，但 cubin 审计仍需 `cuobjdump`。发布验证要求 PyTorch CUDA 13.0、torch 2.12.x 和 Triton 3.7.x。

编译 `cu130` wheel 时还要求实际 CUDA Toolkit 为 13.0：构建预检核对 PyTorch CUDA 版本、PyTorch 检测到的 Toolkit 路径、`PATH` 上的 `nvcc` 和 `nvcc --version`，并固定编译进程使用该 Toolkit。驱动显示 13.2 不影响构建；若机器只安装了 13.2 Toolkit，构建会在开始前失败。`audit`/`full` 验证已有 wheel 时不执行这项编译器检查。

目前已在 Linux 上验证三种架构工具包的生成、源码身份、Python 语法、zip 完整性，并用现有 SM89 Windows wheel 检查归档流程。Windows 原生编译、Triton JIT、cubin 审计和完整 LightX2V 测试仍需在各架构 Windows GPU 上执行。
