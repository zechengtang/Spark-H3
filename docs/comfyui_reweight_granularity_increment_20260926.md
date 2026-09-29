# 同粒度 global reweight 增量测量（2026-09-26）

包含 plan、reblock、TopK 收益及完整调用的后续统一分母对照见
[最新阶段比较](comfyui_latest_stage_comparison_20260926.md)。

## 测量口径

重放真实 10s768p 单层输入 `[1,73565,56,128]`，video=72576。
TopK10、FP32 global anchor、fanout16；GPU0–1 反向顺序，每项预热4次、
CUDA event测量12次，报告两张卡的中位数平均。不是20步视频实验。

本次对照是 **普通均值 KV summary → global-anchor 校正 KV summary**。
两侧保留相同 query 粒度、相同 Spark 选块与 sinks、同一消费 kernel。
原生 `_C.spark_attn(reweight=False)` 是新增消融开关，默认 True，
未暴露给 ComfyUI 节点；不改动既有工作流。基线含均值 summary carrier 的适配成本，
不是官方 Sol 原始融合 route-tail 的直接测速，也不是关闭整个 tail 的 exact-only 基线。

固定 plan 只为排除本次测量之外的规划耗时，并未改变生产 reuse=1 策略。
开启 reblock 的两臂使用同一对预先生成的排列，计入 fused gather/scatter。

## ComfyUI 结果

| 粒度 | reblock | 普通 KV/ms | reweight KV/ms | 增量/ms | 增量比例 |
| --- | --- | ---: | ---: | ---: | ---: |
| block | 无 | 45.162 | 46.046 | 0.884 | 1.96% |
| query | 无 | 49.912 | 50.814 | 0.902 | 1.81% |
| block | fused，plan不计时 | 45.458 | 46.336 | 0.878 | 1.93% |
| query | fused，plan不计时 | 50.114 | 51.017 | 0.902 | 1.80% |

这是总路径的净增量，**不是 summary kernel 自身仅耗时0.9ms**。
例如 GPU0 fused 测量中的 summary kernel约1.62ms、anchor约0.23ms，
辅助 stream 与 transpose/route 重叠，且对照也有普通 summary 构建成本。
单次 profiler attribution只用于解释，CUDA event重复测量才是总耗时依据。

## 是否重复计算近似分支

源代码和 profiler 一致：route 的 tail 参数固定为0，只有一次 route QK 和
一次 TopK finalize；global anchor和weighted summary各构建一次。

- block：一次 `spark_block_tail_kernel`（本次约0.16ms），exact继承其共享状态，
  virtual-token数量设为0，不再执行逐query summary。
- query：没有 block tail consumer，summary只在 exact kernel中逐query累积一次。

普通 pooled K/V统计仍用于路由；其存在不等于重复计算一次 attention tail。

## Diffusers 当前实现与历史参考复测

另行以当前仓库 Diffusers 实现运行相同脚本（`--current-diffusers`），GPU1：
固定同一plan为 **77.765 → 81.743ms，增量3.978ms（5.12%）**。
它计入显式tensor重排而不计plan；ComfyUI对应计入fused重排而不计plan。
这是各自路径内的边际开销比较，并非两边每个kernel或数值精度完全一致。

同一真实输入、TopK10、global reweight、fanout16，GPU1，复测20260921冻结快照：

- 含动态plan：94.527 → 99.103ms，增量4.576ms；另一轮为4.545ms。
- 固定同一plan、仍计入显式tensor重排：77.477 → 81.810ms，增量4.333ms（5.59%）。

因此，按本次同粒度对照的**边际延迟**，ComfyUI两种粒度均低于历史约4.72ms，
也低于本轮当前和冻结Diffusers的复测增量。但不能据此宣称算法/精度/全部执行工作量一致：
ComfyUI INT8路径与Diffusers CuTe路径、路由实现、显式/融合重排均仍不同；
尤其Diffusers是query粒度，block结果不属于同粒度跨后端比较。

**这不是把旧6–7ms“优化到”0.9ms。** 旧数值是相对关闭tail的基线，包含逐query近似计算。
现有query路径总耗时仍约51ms；本轮新增的是可靠消融基线，不是query路径的大幅提速。
block模式降低的是query侧近似粒度对应的工作量，质量影响需另外测量。

## 回归与产物

173项CUDA kernel测试通过；48个历史Spark边界用例逐位不变。
新增验证普通summary浮点参考，以及切换reweight/粒度后route bitmap不变。
Diffusers生产源码未修改。

- 脚本：`scripts/profile_comfy_reweight_increment.py`。
- Diffusers复测：`scripts/profile_diffusers_vs_comfy_spark_20260925.py`。
- 结果目录：`/autodl-fs/data/h3_experiments/spark_cross_pipeline_profile_20260925/`。
- Comfy原始记录：`comfy_reweight_increment_gpu{0,1}_20260926.json`。
- 冻结Diffusers：`diffusers_reweight_grain_refresh_20260926.json`、
  `diffusers_reweight_grain_cached_20260926.json`。
- 当前Diffusers：`diffusers_current_reweight_grain_20260926.json`。
