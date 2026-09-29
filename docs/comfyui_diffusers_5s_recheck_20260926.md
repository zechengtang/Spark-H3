# ComfyUI/Diffusers 5s 速度复核与 ComfyUI 热点筛选

> **历史 Diffusers 对照已退役（2026-09-27）：**下方 5s Diffusers 数字来自未有效开启
> `torch.compile` 的旧实验，原始生成产物已清理。它们仅用于追溯当时的排查过程，
> 不能作为当前 Sol/Spark 速度结论；ComfyUI 数据的有效性需独立判断。

## 端到端对照

同一四条 VBench prompt、seed=42、1344×768、120 帧、请求 20 步；复用已验证的
Diffusers conditioning。每种方法先做一次排除计时的完整去噪预热，同一 GPU 上测
两轮，共 8 条记录。已退役 Diffusers 实现的历史同步去噪时间（不含加载、文本编码、VAE）：

| 方法 | 5s 平均耗时 | 相对 Sol τ=1 |
| --- | ---: | ---: |
| Sol τ=1 | 146.759 s | 1.000× |
| Spark-H3-10pct（fanout16、global reweight） | 146.137 s | 1.004× |

原始记录、每条耗时和配置：
该组旧 Diffusers 原始产物属于未开启有效 `torch.compile` 的清理候选，逐文件路径见 `docs/uncompiled_delete_preview_20260927.tsv`；
脚本：`scripts/diffusers_sol_spark_4prompt_5s768p_20260926.py`。
Diffusers 实际为 19 次 transformer 评估，前 4 次 dense。此前 ComfyUI 四 prompt
实验是 20 次评估，且两侧模型 pipeline 和内核不同，因此**只能比较各自内部的
Sol/Spark 加速比例，不能直接比较绝对秒数**。

在此前 ComfyUI 四 prompt、两轮实测中，5s Sol 为 93.694 s，Spark block 为
96.357 s（0.972×），Spark query 为 97.481 s（0.961×）；10s 分别是
251.235 s、243.170 s（1.033×）、246.977 s（1.017×）。数据：
`/autodl-fs/data/h3_experiments/comfyui_producer_validate_20260926/results.json`。
5s Diffusers 这次只快 0.4%，不能把先前 10s Diffusers 约 1.034× 当作 5s 目标。

## 完整路径热点

来自 `comfyui_pipeline_audit_v2_20260926` 的真实去噪嵌套 CUDA event；下面
同一阶段数字是 exclusive 口径，父子范围不相加。选定 prompt 的每次稀疏调用：

| 阶段 | 5s Spark block | 10s Spark query | 说明 |
| --- | ---: | ---: | --- |
| reblock plan | 9.84 ms | 15.11 ms | 784 次/生成；5s 约 7.7 s，10s 约 11.8 s |
| Spark route/tail/exact | 15.57 ms | 52.09 ms | 已比同长度 Sol core 快 |
| Sol route/tail/exact | 19.92 ms | 69.45 ms | 对照项 |
| QKV buffer copy | 累计 1.73 s | 累计 3.34 s | 次级目标；不能单独解释 5s 差距 |

5s plan 的图重放约 8.06 ms/次；一次 kernel trace 中主要是 score 1.85 ms、
融合节点划分 1.78 ms、BF16 矩阵乘法 1.47 ms、kth 选择约 0.98 ms、
compact key 0.79 ms、范数 0.77 ms、方向生成 0.76 ms。trace 的活跃 kernel
时间可能重叠，不与完整路径均值直接相加。

5s 的视频 token 数为 37296，除以 64 余 48。当前算法按特征范数选尾部
48 个 token；余下 root 索引不是连续的。因此，直接启用连续索引的 root
快路会改变路由语义。此前未对齐 root 试验的 latent 差异很大，不能作为优化。

## 无损候选筛选

在重编译后的 comfy-kitchen SM120 扩展上，同一 5s prompt/seed、20 次评估，
以默认 score tile 128、融合节点 2 warp 为基线：

| 候选 | plan 均值 | 相对基线 | latent SHA-256 |
| --- | ---: | ---: | --- |
| 默认：score 128、节点 2 warp | 9.834 ms | — | `60bc4513…` |
| score tile 64 | 9.824 ms | 波动范围内 | 相同 |
| score tile 32 | 9.828 ms | 波动范围内 | 相同 |
| 融合节点 4 warp | 10.025 ms | 更慢 | 相同 |
| 融合节点 8 warp | 11.548 ms | 更慢 | 相同 |
| FP8 特征表（默认 tile/warp） | 10.632 ms | 更慢 | 不同 |

原始结果分别在 `/autodl-fs/data/h3_experiments/comfyui_5s_plan_score_bm{32,64,128_rebuilt}_20260926/`
及 `comfyui_5s_plan_fused_warps{4,8}_20260926/`。单 prompt 的 sampler 时间
受并行任务和加载影响，不作为调参成功依据。上述候选均未接入生产默认值。
FP8 候选的记录在 `comfyui_5s_plan_fp8_features_20260926/`，额外增加
约 0.80 ms/次 plan 且输出不逐位一致，没有继续做昂贵的视频质量评估。

## 代码状态与正确性

本轮期间另有 `force_local_blocks` 默认策略相关代码改动写入工作树；本任务未
修改或回滚这些文件。新源码与旧二进制曾发生签名不匹配，已按 SM120 重新编译
comfy-kitchen 扩展，并验证导出签名包含 `force_local_blocks`。仓库 ComfyUI 插件及
reblock 的 15 项测试通过。comfy-kitchen 新增的 60 项 tail 测试中 42 项通过、
18 项失败；失败集中在未 reblock 时默认强制相邻块 exact、而参考计算假设
all-summary 的情形。显式 `force_local_blocks=False` 的对应参考抽检余弦为
0.99849，超过测试阈值 0.997。应由该策略变更的维护者统一调整测试参考，
不把这些失败误认为本次 tile/warp 调参回归，也不宣布整套测试通过。

Diffusers 进程在上述默认策略改动前加载了实现，原始协议保存了运行配置；
ComfyUI 新候选使用改动后源码和重编译扩展。进一步的跨实现比较须固定代码
快照。下一优先级是减少 5s 非连续 root 的实际数据读取与路由工作量，再做
同 prompt 的数值/视频质量和端到端配对验证；仅改 launch tile/warp 无效。
