# Spark-H3 ComfyUI 节点

为 ComfyUI 原生 MiniMax-H3 提供 **MiniMax H3 Spark Attention
(SM120)** 节点。将节点接在模型加载器和 `BasicGuider` 之间；
同一模型只使用一个注意力补丁。目前发布版面向 Linux、NVIDIA
SM120（RTX 50 系）和 CUDA BF16。

## 安装

请参阅 **[ComfyUI 中文安装指南](INSTALL.zh-CN.md)**。指南包含
ComfyUI 0.38.x/0.39.x 的版本选择、ZIP 安装命令、手动指定版本和
常见问题排查。

## 模型与工作流

从 [Comfy-Org 的 MiniMax-H3 仓库](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main)下载原生模型，按 ComfyUI 的目录放置：扩散模型放 `models/diffusion_models`，文本编码器放 `models/text_encoders`，视频和音频 VAE 放 `models/vae`。也可以通过 `extra_model_paths.yaml` 使用已有模型目录。示例默认使用官方 `minimax_h3_video_vae_fp16.safetensors`。VAE 选择不改变 Spark attention 算法。

将工作流 JSON 拖入 ComfyUI 画布，并检查模型文件名。三个示例均生成约 14.4 秒的视频：

| 工作流示例 | 配置 |
| --- | --- |
| [原始 MiniMax-H3](../workflows/spark_h3_vdn8_14p4s_t2va.json) | 核心示例；20 步，无 LoRA，无额外节点 |
| [LightX2V 8-step LoRA](../workflows/spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json) | 扩展示例；需另行[下载 LightX2V LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/main) |
| [Larryvrh 8-step LoRA](../workflows/spark_h3_larryvrh_8step_lora_14p4s_t2va.json) | 扩展示例；需[下载 LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)并安装 [Larryvrh 自定义节点](https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo) |

LightX2V 和 Larryvrh 工作流会继续随独立 ZIP 提供，但它们的额外 LoRA/节点
不会由 Spark-H3 静默安装。将 LoRA 放入 `models/loras`。所有示例均显式固定
`reblock_layout=q_reuse_k`，以避免节点默认值未来变化影响复现。Spark 节点参数
见[中文说明](../docs/comfyui_spark_node_parameters.md)。

## 实现说明

这是 Spark-Attn 的初步 ComfyUI 实现，端到端效率仍在优化。Sol-Engine 的 Sol 近似分支只下采样 K/V、保留逐条 Q；ComfyUI 原生 Sol 同时按块下采样 Q 和 K/V。Spark 节点默认使用逐 Q 模式，但两套实现的路由、INT8 数值路径、模型权重和推理步数仍有差异，因此输出和加速比例尚未完全对齐。
