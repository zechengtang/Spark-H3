# Spark-H3 ComfyUI 节点

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md"><strong>简体中文</strong></a>
</p>

为 ComfyUI 原生 MiniMax-H3 提供 **MiniMax H3 Spark Attention
(SM86)**、**MiniMax H3 Spark Attention (SM89)** 与 **MiniMax H3 Spark Attention (SM120)** 节点。将对应节点接在
模型加载器和 `BasicGuider` 之间；同一模型只使用一个注意力补丁。目前实现
面向 Linux 或 Windows x86_64、NVIDIA SM86（GeForce RTX 30 系）、SM89（GeForce RTX 40 系）或 SM120（GeForce RTX 50 系及
RTX PRO 5000/6000 Blackwell）和 CUDA BF16。

## 安装

请参阅 **[ComfyUI 中文安装指南](INSTALL.zh-CN.md)**。指南包含
ComfyUI 0.38.x/0.39.x/0.40.x 的版本选择、ZIP 安装命令、手动指定版本和
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

Spark 节点最前面的 `mode` 按 `fidelity / speed / custom / dense` 排列。
`fidelity`（默认）填入 20% dense 预热，`speed` 填入
`warmup_mode=warmup_ratio, warmup_ratio=0`，`custom` 使用手动 warmup 配置，
`dense` 原样输出未打补丁的模型。原 `enabled` 开关被移除，但 `warmup_mode`、
`warmup_ratio`、`warmup_steps` 都保留。手动修改当前生效的 warmup 参数使其
不符合预设时，mode 自动变为 `custom`；修改未生效的参数不会触发切换。
切换到 `dense` 也不会清空 warmup 值，便于之后切回 Spark。
从其他 mode 切换到 `fidelity` 时，会把未连接成外部输入的 warmup mode 和比例
重新写为 `warmup_ratio` 和 `0.2`。`fidelity` 下还会显示一个不参与序列化的提醒
按钮：“双采高分辨率二阶段：建议使用 mode=speed → 一键切换”。点击即可应用
speed 预设。

`sparse_scope` 默认是 `target-video`，完整保留原有 target-only 布局和 direct-output
路径。Ref2VA 可以选择 `joint-video`，把具有多个时间位置的参考视频视觉 token 和
生成目标视频放入同一个稀疏域，统一执行 Reblock、Top-K 和 Reweight；静态参考图、
文本、参考音频和目标音频仍保持 exact。没有参考视频视觉 token 时，`joint-video`
自动使用完全相同的 target-video 路径。由于 `joint-video` 会扩大候选集合，建议优先使用
`topk_ratio`；固定 `topk_blocks` 在更大候选集合中对应的实际比例会降低。
当未连接外部输入且仍选择 `target-video` 时，节点会显示一个不参与序列化的提醒按钮，
显示“带视频条件的 Ref2VA：建议使用 sparse_scope=joint-video → 一键切换”；点击即可切换。

`producer_chunk` 是 Spark Q/K/V producer 每次处理的 token 数，默认 `16384`，
必须是 64 的正整数倍。减小它可能降低 producer 阶段的临时显存，但会增加分块调用
开销；未经实测建议保持默认值。该设置只影响 Python 运行时分块，不需要重新编译 wheel。

`dense_backend` 控制 dense 预热、固定 dense 层、短序列和非严格模式回退。
默认 `inherit` 不强制覆盖，沿用模型或 ComfyUI 当前的注意力后端；也可以选择
`kitchen`、`sage` 或 `sdpa`，只对这些 dense 调用强制使用对应后端。该切换发生在 Python
分发层，不需要重新编译各架构的 Spark wheel；`sage` 需要安装 SageAttention，
`kitchen` 使用独立 `spark-h3-kitchen` 包提供的 INT8 attention，不会覆盖
ComfyUI 官方 `comfy-kitchen`。
`inherit` 不注入新的 attention override：模型、上游节点或 ComfyUI 已经选择的
后端会原样保留，包括官方 `comfy-kitchen` 提供的 Kitchen Attention；但 Spark-H3
不会主动选择它。
当保持 `inherit` 且该参数未连接外部输入时，节点会显示一个不参与序列化的提示按钮：
“Dense warmup 加速：可使用 dense_backend=kitchen → 一键切换”。点击即可应用 kitchen，
默认值仍保持 `inherit`。

### KJNodes 低显存节点

Spark 可与 ComfyUI-KJNodes 的 `MiniMax H3 Low VRAM Attention` 和
`MiniMax H3 Chunk FeedForward` 组合，推荐且受支持的 MODEL 连接顺序是：

```text
模型加载器 → MiniMax H3 Low VRAM Attention → MiniMax H3 Chunk FeedForward
          → MiniMax H3 Spark Attention → BasicGuider
```

必须把 Low VRAM Attention 放在 Spark 前面：Spark 会在安装补丁时检测它的
`sol_take_forward` 协作标记，并选择独立的 head-chunk 兼容路径。不连接该节点时
仍安装原有 Spark block patch，没有每层运行时兼容判断。Chunk FeedForward 只分块
执行 MLP，与 attention patch 相互独立。两种适配均复用现有 Spark CUDA 接口，
不需要重新编译 SM86、SM89 或 SM120 wheel。更多分块通常降低峰值显存，但会增加
kernel 调用次数和耗时。

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
