# Spark-H3 ComfyUI 节点

为 ComfyUI 原生 MiniMax-H3 提供 **MiniMax H3 Spark Attention (SM120)** 节点。将它接在模型加载器和 `BasicGuider` 之间；同一模型只使用一个注意力补丁。目前支持 Linux、NVIDIA SM120 显卡和 CUDA BF16。

## 安装

先按 [ComfyUI 官方说明](https://docs.comfy.org/installation/manual_install)安装支持 MiniMax-H3 的版本（0.30.0 或更新版本）。正式 Release 提供该资产后，推荐下载独立的
`ComfyUI-Spark-H3-<version>.zip`。该包仅包含节点、reblock 运行时代码、示例
workflow 和匹配的 kernel wheel，不包含模型权重或主仓库中的研究资料。

将 ZIP 直接解压到 `ComfyUI/custom_nodes`，再用**启动 ComfyUI 的同一个
Python**运行安装器：

```bash
cd /path/to/ComfyUI/custom_nodes
unzip /path/to/ComfyUI-Spark-H3-<version>.zip
/path/to/ComfyUI/.venv/bin/python ComfyUI-Spark-H3/install.py
```

安装器优先使用已经安装的 Spark 后端、包内 wheel 或固定 GitHub Release
中的兼容 wheel；只有找不到 wheel 时才会拉取固定版本的
`comfy-kitchen 0.2.36`、应用 Spark patch 并源码编译。源码回退需要 Git、
CMake、Ninja、C++ 编译器和 CUDA 12.8 或更新版本的 `nvcc`。安装后重启
ComfyUI；如果 Python 不在示例路径，请换成实际路径。

当前已验证的预编译配置为 Linux x86_64、Python 3.12+ 和 NVIDIA SM120。
Python 3.10/3.11 需要对应 wheel，否则会进入源码编译。Windows 打包路径已
准备好，但 Windows wheel 和 SM120 运行仍待单独验证。非 SM120 显卡只能
测试节点加载和 dense fallback，不能验证 Spark kernel。

如果暂时没有 Release ZIP，也可以直接克隆当前仓库；这种方式功能相同，但
会下载主仓库中的其他内容：

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/zechengtang/Spark-H3.git
/path/to/ComfyUI/.venv/bin/python Spark-H3/comfyui/install.py
```

维护者可以从当前主仓库生成一个不包含模型权重和研究资料的独立节点包：

```bash
python tools/build_comfyui_package.py \
  --publisher-id YOUR_COMFY_REGISTRY_PUBLISHER_ID
```

输出位于 `dist/comfyui/ComfyUI-Spark-H3`，同时生成可直接解压到
`custom_nodes` 的 ZIP。发布正式版本前，可用 `comfyui/kernel_builder.py`
构建 SM120 backend wheel，并通过 `--kernel-wheel` 将其装入离线安装包。

## 模型与工作流

从 [Comfy-Org 的 MiniMax-H3 仓库](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main)下载原生模型，按 ComfyUI 的目录放置：扩散模型放 `models/diffusion_models`，文本编码器放 `models/text_encoders`，视频和音频 VAE 放 `models/vae`。也可以通过 `extra_model_paths.yaml` 使用已有模型目录。

将工作流 JSON 拖入 ComfyUI 画布，并检查模型文件名。四个示例均生成约 14.4 秒的视频：

| 工作流示例 | 配置 |
| --- | --- |
| [原始 MiniMax-H3](../workflows/spark_h3_vdn8_14p4s_t2va.json) | 20 步，无 LoRA |
| [LightX2V 8-step LoRA](../workflows/spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json) | [下载 LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/main) |
| [Larryvrh 8-step LoRA](../workflows/spark_h3_larryvrh_8step_lora_14p4s_t2va.json) | [下载 LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)；需安装 [自定义节点](https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo) |
| [ComfyUI 8-step LoRA](../workflows/spark_h3_minimax_h3_comfyui_8step_lora_14p4s_t2va.json) | [下载 LoRA](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/loras) |

将 LoRA 放入 `models/loras`。Spark 节点参数见[中文说明](../docs/comfyui_spark_node_parameters.md)。

## 实现说明

这是 Spark-Attn 的初步 ComfyUI 实现，端到端效率仍在优化。Sol-Engine 的 Sol 近似分支只下采样 K/V、保留逐条 Q；ComfyUI 原生 Sol 同时按块下采样 Q 和 K/V。Spark 节点默认使用逐 Q 模式，但两套实现的路由、INT8 数值路径、模型权重和推理步数仍有差异，因此输出和加速比例尚未完全对齐。
