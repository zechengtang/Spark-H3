# Reblock：5s768p 独立成本与完整生成验证

状态：独立构造成本探针和 3 prompts 的完整生成验证已完成；
两张 GPU 已释放。当前不更改生产默认配置，尚未验证 10s 迁移。

## 结论

完整生成验证没有给出足够强的默认替换理由：Both-side swap4 的平均
PSNR 仅增加 0.092 dB，SSIM 增加 0.00883，去噪耗时却增加 38.53%。
因此本轮结论是**保留 baseline，不将局部交换替换为默认实现**。

## 独立构造成本探针

在空闲 GPU 上使用完整 56-head、37,296-token 形状，3 次预跑、10 次
CUDA-event 测量取中位数。为填充形状，将真实输入的 8 heads 重复 7 次；
这些重复数据只用于成本探针。

| 操作 | 中位耗时 ms |
|---|---:|
| 生产 reblock | 13.047 |
| 双侧 swap4，预先计算特征，未融合参考代码 | 71.449 |
| global anchor 构造 | 0.462 |
| Q64-local anchors 构造 | 0.381 |
| global 原生摘要构造 | 0.778 |
| 582 个 Q64-local 原生摘要，一次物化 | 11.010 |
| K32 双摘要，未融合 Torch 参考代码 | 4.735 |

局部 anchor 的主要额外工作在**按各 anchor 构造 K/V 摘要**，不是求 Q 均值。
全物化 Q64-local 摘要约需 9.8 GB，而 global 仅约 16.8 MB；原生摘要构造
探针约慢 14.2 倍。实际生产 consumer 使用有界分块和流式处理，且 attention
读取/缓存行为也变化，不能把该比值解释为端到端加速或减速倍数。

参考 swap4 的 71ms 不含额外 M2/变换工作，包含 Torch 实现的中间张量、
kernel launch 和同步开销；这不是融合实现的理论下界。当前参考版的
完整生成耗时已显示约 40% 增加，因此即使质量验证有收益也不能直接替换默认。

原始记录为实验根目录的 `cost_probe.json`；脚本为
[benchmark_reblock_summary_cost_5s768p_20261002.py](/autodl-fs/data/h3_repos/MiniMax-H3/scripts/benchmark_reblock_summary_cost_5s768p_20261002.py)。

## 三样本完整生成验证

使用 prompts 1、15、44，seed42、120 requested frames（内部对齐 124，
解码后取前 120 帧），其余参数与生产 baseline 相同。每个 worker 排除一次
完整 baseline 预跑，然后分别从相同 seed 和 conditioning 生成 Dense、baseline
和 Both-side swap4。9 次生成的 video/audio latents 全部有限。

相对于重新生成的匹配 Dense RGB，视频像素无压缩保存为 uint8 NumPy 数组。
每段视频 PSNR 使用全部 RGB 像素的 pooled MSE；SSIM 使用逐帧
Gaussian 11×11、sigma1.5、valid convolution，最后按 prompt 等权平均。
这些指标衡量与 Dense 的接近程度，不是官方 VBench 感知质量排名。

| 方法 | 平均 PSNR dB ↑ | 平均 SSIM ↑ | 平均去噪秒 ↓ |
|---|---:|---:|---:|
| baseline | 22.5323 | 0.771330 | 141.624 |
| Both-side swap4 | 22.6244 | 0.780162 | 196.191 |

| Prompt | ΔPSNR dB | ΔSSIM |
|---|---:|---:|
| 1 | −0.3829 | +0.008572 |
| 15 | +0.4035 | +0.014752 |
| 44 | +0.2556 | +0.003173 |

PSNR 在 2/3 prompts 改善，SSIM 在 3/3 改善。平均增益很小，3 prompts、
单 seed 的小规模验证不足以作确认性质量结论。去噪只测一次，
参考适配器未融合，并重复构造 M2/变换特征；38.53% 是本实现的描述性成本，
不是优化后算法的必然成本。也不以该计时取代已有速度协议的加速比。

原始记录：实验根目录的 `generation_check/results.json`、`latents/`、`rgb/`。
若继续布局方向，先融合或减少交换轮数，再用完整生成评估是否扩展 10s 实验。
本轮结果不证明任意更优布局的全局上限只剩这些小幅收益。

## 复现与记录

完整生成脚本：
[verify_reblock_screen_generation_5s768p_20261002.py](/autodl-fs/data/h3_repos/MiniMax-H3/scripts/verify_reblock_screen_generation_5s768p_20261002.py)。
使用 prompts 1、15、44，比较 Dense、baseline 和 Both-side swap4；
输出 RGB PSNR/SSIM，并记录未融合参考适配器的去噪耗时。
这是小规模验证，不是 held-out 确认。
