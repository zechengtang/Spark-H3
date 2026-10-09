# Spark-H3 ComfyUI 节点

为 ComfyUI 原生 MiniMax-H3 提供 **MiniMax H3 Spark Attention
(SM89)** 与 **MiniMax H3 Spark Attention (SM120)** 节点。将对应节点接在
模型加载器和 `BasicGuider` 之间；同一模型只使用一个注意力补丁。目前实现
面向 Linux、NVIDIA SM89（RTX 4090）或 SM120（RTX 50 系）和 CUDA BF16。

## 安装

请参阅 **[ComfyUI 中文安装指南](INSTALL.zh-CN.md)**。指南包含
ComfyUI 0.38.x/0.39.x 的版本选择、ZIP 安装命令、手动指定版本和
常见问题排查。

## 模型与工作流

从 [Comfy-Org 的 MiniMax-H3 仓库](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main)下载原生模型，按 ComfyUI 的目录放置：扩散模型放 `models/diffusion_models`，文本编码器放 `models/text_encoders`，视频和音频 VAE 放 `models/vae`。也可以通过 `extra_model_paths.yaml` 使用已有模型目录。示例默认使用官方 `minimax_h3_video_vae_fp16.safetensors`。VAE 选择不改变 Spark attention 算法。

将工作流 JSON 拖入 ComfyUI 画布，并检查模型文件名。DMAD 与 LBH 示例生成约
5.2 秒的视频，其余示例均生成约 14.4 秒的视频：

| 工作流示例 | 配置 |
| --- | --- |
| [原始 MiniMax-H3](../workflows/spark_h3_vdn8_14p4s_t2va.json) | 核心示例；20 步，无 LoRA，无额外节点 |
| [LightX2V 8-step LoRA](../workflows/spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json) | 扩展示例；需另行[下载 LightX2V LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/main) |
| [Larryvrh 8-step LoRA](../workflows/spark_h3_larryvrh_8step_lora_14p4s_t2va.json) | 扩展示例；需[下载 LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)并先转换为 ComfyUI 通用格式 |
| [DMAD 4-step LoRA](../workflows/spark_h3_dmad_4step_lora_5p2s_t2va.json) | 使用 DMAD re-noise sampler；需先转换 DMAD LoRA |
| [LBH 官方 two-pass + LightX2V 4-step](../workflows/spark_h3_lbh_official_lightx2v_4step_5p2s_i2va.json) | 需安装 LBH 官方 ComfyUI 节点并下载 latent upscaler |

LightX2V 和 Larryvrh 工作流会继续随独立 ZIP 提供，但它们的额外 LoRA
不会由 Spark-H3 静默安装。将 LoRA 放入 `models/loras`。所有示例均显式固定
`topk_mode=topk_ratio` 和 `reblock_layout=q_reuse_k`。常规、LightX2V、
Larryvrh 与 DMAD 示例使用 `topk_ratio=0.2`；LBH 示例为对齐 Diffusers
`h3_lbh` 路径使用 `topk_ratio=0.1`。Spark 节点参数见
[中文说明](../docs/comfyui_spark_node_parameters.md)。

Larryvrh 原始权重需使用 `tools/convert_larryvrh_lora_comfyui.py` 转换。转换会保留
208 个主干 LoRA，并将 51 个 AdaLN 更新投影到 pruned 模型的 8 维时间曲线；之后
工作流与 LightX2V 一样使用原生 `LoraLoaderModelOnly`，无需 Larryvrh 自定义节点，
也没有逐步 runtime-LoRA 矩阵乘法。

LBH 示例使用官方 two-pass 结构：0.2 MP 的低分辨率阶段执行 4 次模型评估，
取 `denoised_output` 后分离音视频 latent，只提升视频 latent；随后在 0.98 MP
重新加噪，并用 `0.9035, 0.6316, 0.3158, 0.0` 完成 3 次高分辨率评估。
Spark 只作用于第二阶段，设置为零 warmup、10% Top-K。安装
`LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`，并将
`minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors` 放入
`ComfyUI/models/latent_upscale_models/`。

## 实现说明

这是 Spark-Attn 的初步 ComfyUI 实现，端到端效率仍在优化。Sol-Engine 的 Sol 近似分支只下采样 K/V、保留逐条 Q；ComfyUI 原生 Sol 同时按块下采样 Q 和 K/V。Spark 节点默认使用逐 Q 模式，但两套实现的路由、INT8 数值路径、模型权重和推理步数仍有差异，因此输出和加速比例尚未完全对齐。
