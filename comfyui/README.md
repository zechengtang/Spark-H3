# Spark-H3 ComfyUI 节点

为 ComfyUI 原生 MiniMax-H3 提供 **MiniMax H3 Spark Attention (SM120)** 节点。将它接在模型加载器和 `BasicGuider` 之间；同一模型只使用一个注意力补丁。目前支持 Linux、NVIDIA SM120 显卡和 CUDA BF16。

## 安装

先按 [ComfyUI 官方说明](https://docs.comfy.org/installation/manual_install)安装支持 MiniMax-H3 的版本（0.30.0 或更新版本），并准备 CUDA 12.8 或更新版本的 `nvcc`。使用**启动 ComfyUI 的同一个 Python**运行安装脚本：

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/zechengtang/Spark-H3.git
cd Spark-H3
bash comfyui/install.sh /path/to/ComfyUI/.venv/bin/python
```

脚本会安装节点依赖，并为 ComfyUI 使用的 `comfy-kitchen 0.2.36` 编译 Spark 扩展。安装后重启 ComfyUI。首次安装需要联网和编译；如果你的 Python 不在示例路径，请换成实际路径。

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
