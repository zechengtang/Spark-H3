<div align="center">

<h1><img src="assets/spark-h3-wordmark.png" alt="Spark-H3" width="270"></h1>
<h3>适用于 MiniMax-H3 的自适应分块稀疏注意力</h3>

<p><strong>相似 token 重组块 · 尾部 token 重加权</strong></p>

<p>
  <a href="https://zechengtang.github.io/Spark-H3/"><img src="https://img.shields.io/badge/Blog-Spark--H3-f97316?style=flat-square" alt="Spark-H3 技术博客"></a>
  <img src="https://img.shields.io/badge/arXiv-Spark--H3-b31b1b?style=flat-square" alt="Spark-H3 arXiv">
  <a href="https://huggingface.co/Aazeus/Spark-H3"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Spark--H3-ffc107?style=flat-square" alt="Spark-H3 Hugging Face 页面"></a>
  <a href="comfyui/README.md"><img src="https://img.shields.io/badge/ComfyUI-Spark_Node-f97316?style=flat-square" alt="Spark-H3 ComfyUI 节点"></a>
  <a href="#special-thanks"><img src="https://img.shields.io/badge/Special_Thanks-Community_Contributors-e11d48?style=flat-square" alt="特别感谢社区贡献者"></a>
  <a href="h3_sparse_attention/README.md"><img src="https://img.shields.io/badge/Docs-Usage_%26_Configuration-2563eb?style=flat-square" alt="使用与配置指南"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-15803d?style=flat-square" alt="Apache 2.0 许可证"></a>
  <a href="https://github.com/zechengtang/Spark-H3/releases"><img src="https://img.shields.io/github/downloads/zechengtang/Spark-H3/total?style=flat-square&amp;label=Downloads" alt="GitHub Release 累计下载量"></a>
</p>

<p>
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p>
  <a href="#demo">🎬 演示</a> &nbsp;·&nbsp;
  <a href="#quick-start">🚀 快速开始</a> &nbsp;·&nbsp;
  <a href="#comfyui-installation">🧩 ComfyUI</a> &nbsp;·&nbsp;
  <a href="https://zechengtang.github.io/Spark-H3/">📖 技术博客</a> &nbsp;·&nbsp;
  <a href="#project-status">🧭 项目进展</a>
</p>

</div>

> [!WARNING]
> **兼容性与更新计划提醒（2026-10-25）**
>
> 近期我们收到社区用户反馈：新版`comfy-kitchen`的实现可能在部分硬件和工作流中引起性能下降。目前相关问题仍在排查，实际影响可能因运行环境而异。
>
> 在问题得到进一步确认和解决前，建议用户暂时安装或继续使用经过验证、运行较稳定的**ComfyUI 0.38.x**，并谨慎升级至更新版本。相关现象与测试可参考这段**[社区反馈视频](https://www.bilibili.com/video/BV1RaHK6cEHt/)**。
>
> 受此影响，原计划于**2026年10月27日**发布的新版本将暂时推迟。我们会优先配合排查`comfy-kitchen`的相关问题，待兼容性和性能表现确认稳定后，再恢复版本发布。后续进展将在本项目中及时同步，感谢大家的理解与反馈。

> **ComfyUI v0.1.5：** 新增对 ComfyUI 0.42.x（comfy-kitchen 0.2.39）的支持，
> 提供 Linux 和 Windows 的 SM86、SM89、SM120 安装包；每包也包含适配
> ComfyUI 0.38.x–0.40.x 的 wheel。
> **[查看发布说明并下载](https://github.com/zechengtang/Spark-H3/releases/tag/comfyui-v0.1.5)**

> **VPack v0.1.0：** 独立的 ComfyUI 插件，用于压缩 Ref2VA 任务中的参考视频；
> 推荐使用默认的 `2/5`。
> **[VPack GitHub 仓库](https://github.com/zechengtang/VPack)**

> 👉 [Ref2VA 任务加速解决方案](docs/sparse_scope.md#ref2va-任务加速解决方案)

---

<a id="demo"></a>

## 🎬 演示

原生 MiniMax-H3 · VDN Selected Prompt 06 · 1344×768 · 14.4 秒。
三栏对比 Dense、Spark-H3 Fidelity（2.19×）和 Spark-H3 Speed（3.12×）。
下方动态预览和点击后的视频片段从第 9 秒开始，覆盖最后约 5.4 秒。

[![原生 MiniMax-H3 阁楼场景动态对比：Dense、Spark-H3 Fidelity 和 Spark-H3 Speed](assets/h3-native-attic-comparison-9s-to-end.webp)](assets/h3-native-attic-comparison-9s-to-end.mp4)

<p align="center"><a href="https://huggingface.co/Aazeus/Spark-H3/resolve/main/comfyui/demos/H3_native_vdn_selected_0006_dense_fidelity_speed_native_resolution.mp4">观看完整 14.4 秒对比视频</a></p>

- [Veda / Sol / Spark：主观画质与速度对比（B 站）](https://www.bilibili.com/video/BV1zynA6aEZG/)
- [Spark-H3 与 VPack 联合用于 Ref2VA 的演示（B 站）](https://www.bilibili.com/video/BV1i6J163EGG/)

## ✨ Spark-Attn

Spark-Attn 是面向 MiniMax-H3 视频生成的分块稀疏注意力方法，包含两个操作：

- **🧩 重组块（Reblocking）**：将相似的 token 归入同一块，让稀疏注意力选中更相关的交互。
- **⚖️ 重加权（Reweighting）**：构建 key/value 摘要时，对尾部块中的 token 赋予不同权重，并修正注意力权重总量。

Spark-H3 提供 Spark 注意力算子及其与 MiniMax-H3 推理流程的集成。

<a id="quick-start"></a>

## 🚀 快速开始

先按照 [MiniMax-H3 上游说明](https://github.com/MiniMax-AI/MiniMax-H3) 配置 H3 推理流程，再在同一环境中使用兼容的 PyTorch/CUDA 版本安装 Spark：

```bash
python -m pip install -e '.[cuda]'
```

在已加载的 pipeline 中使用 Spark 安装器：

```python
from h3_sparse_attention import install_h3_spark_attn

# MiniMaxH3Scheduler 会将末尾的 sigma=0 计入 num_inference_steps，
# 但这个时间点不会运行 transformer。
num_denoise_steps = 19
num_inference_steps = num_denoise_steps + 1

with install_h3_spark_attn(
    pipe.transformer,
    num_denoise_steps=num_denoise_steps,
    warmup_mode="warmup_steps",
    warmup_steps=4,
    dense_layers=0,
    topk_ratio=0.1,
):
    result = pipe(**inputs, num_inference_steps=num_inference_steps)
```

<a id="comfyui-installation"></a>

## 🧩 ComfyUI 安装

Spark-H3 为 ComfyUI 0.38.x–0.40.x 和 0.42.x 提供独立节点。[v0.1.5](https://github.com/zechengtang/Spark-H3/releases/tag/comfyui-v0.1.5) 新增 ComfyUI 0.42.x 支持，提供 Linux 和 Windows x86_64 的 SM86、SM89、SM120 安装包；每包也支持 ComfyUI 0.38.x–0.40.x。请选择与操作系统和 GPU 架构匹配的 CUDA 13.0 安装包。发布包不包含 ComfyUI 和模型权重。

安装方法及示例工作流请参阅 [ComfyUI 指南](comfyui/README.md) 或 [中文安装指南](comfyui/INSTALL.zh-CN.md)。

如需压缩 Ref2VA 任务中的参考视频，可另行安装独立插件 **[VPack v0.1.0](https://github.com/zechengtang/VPack)**。

## 步数约定

`MiniMaxH3Scheduler` 会把最后的 `σ=0` 时间点计入 `num_inference_steps`，因此默认的 `num_inference_steps=50` 实际运行 49 个去噪步骤。详见 [Hugging Face 调度器文档](https://huggingface.co/docs/diffusers/main/api/schedulers/minimax_h3)。

<a id="todo"></a>
<a id="project-status"></a>

## 🧭 项目进展与计划

Diffusers 与 ComfyUI 是两条独立的实现路线。下面分开记录两者的进展，
避免将后端算子、ComfyUI 打包和兼容性工作混为一谈。

### Diffusers / Python

| 状态 | 项目 | 资料 |
|---|---|---|
| ✅ 已实现 | MiniMax-H3 的 Spark attention 集成，以及支持 SM80、SM86、SM89 和 SM120 的融合算子 | [使用与算子文档](h3_sparse_attention/README.md) · [性能对照表](docs/kernel_performance_matrix.md) |
| ✅ 已实现 | Ref2VA pipeline 支持及已发布的推理示例 | [Spark-Ref2VA 预览](https://zechengtang.github.io/Spark-H3/#spark-ref2va-preview) |
| 🗓️ 计划中 | 支持 SM90 和 SM100 的融合 Spark 算子 | 待硬件资源到位 |
| 🗓️ 计划中 | 发布技术报告 | — |

### ComfyUI

| 版本 | 更新日期 | 状态 | 主要改进 / 预计更新 | 发布 / 跟踪 |
|---|---|---|---|---|
| v0.1.1 | 2026-10-09 | ✅ 已发布 | 首个稳定的 SM89 / SM120 Linux 安装包；修复 v0.1.0 的 CUDA Graph 分配生命周期问题；内置 5 组代表性工作流 | [发布说明](https://github.com/zechengtang/Spark-H3/releases/tag/comfyui-v0.1.1) |
| v0.1.2 | 2026-10-11 | ✅ 已发布 | 新增经过验证的 SM89 / SM120 Windows 安装包；修正内置工作流并优化 LBH 双采样示例 | [发布说明](https://github.com/zechengtang/Spark-H3/releases/tag/comfyui-v0.1.2) |
| v0.1.3 | 2026-10-13 | ✅ 已发布 | 新增经过验证的 SM86 安装包；适配 KJNodes 低显存节点；新增 `fidelity`、`speed`、`custom`、`dense` 预设及 dense 后端选择 | [发布说明](https://github.com/zechengtang/Spark-H3/releases/tag/comfyui-v0.1.3) |
| v0.1.4 | 2026-10-15 | ✅ 已发布 | 新增 ComfyUI 0.40.x 支持、面向 Ref2VA 的 `sparse_scope`、可配置 `producer_chunk` 及 Ref2VA 节点内提示 | [发布说明](https://github.com/zechengtang/Spark-H3/releases/tag/comfyui-v0.1.4) |
| v0.1.5 | 2026-10-19 | ✅ 已发布 | 通过 comfy-kitchen 0.2.39 新增 ComfyUI 0.42.x 支持；发布 Linux 和 Windows 的 SM86/SM89/SM120 安装包，每包也内置适配 ComfyUI 0.38.x–0.40.x 的 wheel | [发布说明](https://github.com/zechengtang/Spark-H3/releases/tag/comfyui-v0.1.5) |
| v0.1.6 | 待定 | 🛠️ 计划中 | 增加 Spark-H3 对 [SelfLift-Avatar](https://github.com/slmonker/selflift-Avatar) 的兼容支持，包括集成适配与工作流验证；新增对 ComfyUI 0.37.x 及 Python 3.10.x–3.11.x 的支持 | [SelfLift-Avatar 仓库](https://github.com/slmonker/selflift-Avatar) |

当前 v0.1.5 的 Linux 和 Windows x86_64 SM86/SM89/SM120 安装包支持 ComfyUI 0.38.x–0.40.x 和 0.42.x。
工作流请参阅 [ComfyUI 指南](comfyui/README.md)，
安装包详情请参阅[安装指南](comfyui/standalone/README.md)。参考视频压缩由独立的
[VPack](https://github.com/zechengtang/VPack) 提供。

## 🤝 致谢

感谢 [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) 和 [Sol-Attn / Sol-Engine](https://github.com/NVlabs/Sana/tree/sol-engine) 提供模型与注意力基础设施。算子归属和所含许可证信息请参阅[第三方声明](sol_attn/THIRD_PARTY_NOTICES.md)。

<a id="special-thanks"></a>

## ❤️ 特别感谢

- 感谢 [Smthem](https://space.bilibili.com/439655233) 在 Spark-H3 的 ComfyUI 插件开发过程中给予的支持与帮助；
- 感谢 [AI绘画KK](https://space.bilibili.com/313528845) 反馈了社区对 Windows 适配的需求，并为插件测试提供了宝贵的 workflow 案例；
- 感谢 [AIwood爱屋研究室](https://space.bilibili.com/503934057) 指出了 Spark-H3 应用于双采工作流时出现的一个 Bug，并提出了将 dense backend 切换为 Kitchen 的宝贵改进建议，使插件的加速比得到进一步提升。

## ⭐ Star History

<p align="center">
  <a href="https://www.star-history.com/?repos=zechengtang%2FSpark-H3&amp;type=date&amp;legend=top-left">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=zechengtang/Spark-H3&amp;type=date&amp;theme=dark&amp;legend=top-left">
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=zechengtang/Spark-H3&amp;type=date&amp;legend=top-left">
      <img alt="Spark-H3 GitHub Star 历史曲线" src="https://api.star-history.com/chart?repos=zechengtang/Spark-H3&amp;type=date&amp;legend=top-left" width="720">
    </picture>
  </a>
</p>

## 📄 许可证

Spark-Attn 的原创代码和文档采用 [Apache License 2.0](LICENSE)。第三方组件保留其原有许可证和声明，详见[第三方声明](sol_attn/THIRD_PARTY_NOTICES.md)。MiniMax-H3 模型权重仍受其上游许可证约束。
