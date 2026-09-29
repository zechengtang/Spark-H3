# ComfyUI 一小时提速迭代（2026-09-28）

## 依据与方案

最新本地 10s/768p、20 次模型求值的完整 profile 位于
`/autodl-fs/data/h3_experiments/comfyui_current_fullpath_profile_20260928/`。
同设置未加 profile 的采样时间为 Sol tau=1 252.419s、Spark block
244.880s，即 Spark 快 1.031 倍。Spark 稀疏侧的 reblock plan parent
累计 12.192s，之后的 preprocess/route/tail/exact 为 37.396s；QKV
投影累计 22.455s（4136 次），独立 V 缓冲拷贝 1.234s（3920 次）。
plan 内 Q/K 已在不同 stream 上执行；既有的 direct output scatter 已在
另一项完整 A/B 中省下平均 1.279s 且 latent 逐位一致。不能把重叠的
`plan_graph_replay` exclusive range 与 plan parent 相加。

本轮选择两个不改稀疏算法的候选：先调 QKV producer chunk，减少投影调用；
再尝试把 V copy 放到辅助 CUDA stream，与同 chunk 的 Q/K RMSNorm/RoPE
及后续工作重叠。两者的可见收益以完整 ComfyUI 采样计时为准。

## 验证方法

固定归档 prompt 1、seed 42、10s/768p、20 步、Spark block、TopK10、
full reblock、global reweight、direct output。每个方法及 GPU 独立启动
ComfyUI，先做排除计时的 5 步预热，再测一次完整 20 步 sampler；GPU0/1
采用相反方法顺序。每次都保存 latent SHA256。启动、模型加载、解码、保存
均不计入 sampler_seconds。

原始记录：

- 分块：`/autodl-fs/data/h3_experiments/comfyui_producer_chunk_paired_20260928/summary.json`
- V copy：`/autodl-fs/data/h3_experiments/comfyui_async_v_copy_ab_20260928/summary.json`

## 结果

| 候选 | GPU0 采样秒 | GPU1 采样秒 | 与 16K 同卡对照 |
| --- | ---: | ---: | --- |
| 16K chunk | 247.538 | 243.716 | 基线 |
| 24K chunk | 249.466 | 245.226 | 两卡分别慢 1.928/1.509s |
| 32K chunk | 247.430 | 243.502 | 两卡分别快 0.108/0.215s |

32K 平均只快 0.161s（约 0.07%），不足以从单提示词单次重复中认定稳定收益；
生产默认仍为 16K。六份 latent 的 SHA256 完全相同。

| V copy 调度，固定 16K | GPU0 秒 | GPU1 秒 | 同卡差值 |
| --- | ---: | ---: | --- |
| 同步基线 | 247.490 | 243.858 | — |
| 辅助 stream | 247.654 | 243.599 | GPU0 慢 0.164s；GPU1 快 0.259s |

异步方案平均仅快 0.047s（约 0.02%），方向不一致。四份 latent 的
SHA256 完全相同。实验性异步实现已从源码移除，生产路径保持原来的同步拷贝。

完整路径的主要剩余成本仍是 reblock plan、稀疏 route/exact 和 QKV
投影。上述分块和 V copy 调度都没有验证出足以影响生产默认值的收益。
后续若继续追求原定 1.05 倍目标，应优先验证能减少 plan 或 producer
大块读写的融合实现，并使用相同的同卡完整采样与结果校验。

本轮代码通过 `PYTHONPATH=. pytest -q tests/test_comfyui_plugin.py`，15 项通过。
