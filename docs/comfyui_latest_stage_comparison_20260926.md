# 最新 attention 阶段对照（2026-09-26）

## 清理记录

已删除旧报告中以 exact-only 为基线计算的 reweight 增量列、百分比及“未达标”结论。
从 `summary_gemm_v17_gpu{0,1}.json`、`summary_register_v18_gpu{0,1}.json`、
`reweight_tail_first_v19_gpu{0,1}.json` 中移除共24个 `increment_ms` 字段，
生成脚本也停止输出它们。原始CUDA event计时、质量检查、正确的候选A/B结论保留，
未删除模型、生成视频或有效的同粒度实验。被删除的是可由原始计时重算但归因错误的派生量。

## 当前口径

真实10s768p单层capture：BTHD=[1,73565,56,128]，video=72576，TopK10、
fanout16、FP32 global anchor、reuse=1。ComfyUI为GPU0–1反向执行的中位数平均，
Diffusers为当前仓库实现GPU1实测；每项4次warmup、12次CUDA event计时。
不是完整视频或20步去噪计时，不含PSNR/VBench。

ComfyUI tau与TopK诊断对照使用相同无强制local的Spark路由、相同sink及同粒度
普通均值KV consumer；不是直接把官方Sol默认block消费与Spark query消费相减。
官方未改路由的Sol tau=1另测为72.68ms，列作旁证，不混入匹配消融。
Diffusers是query粒度；其CuTe数值路径、选块细节及重排方式不与ComfyUI逐位一致。

下表括号中的百分比统一除以本列“不含reblock/reweight的TopK10普通近似基线”。

| 项目 | ComfyUI block | ComfyUI query | 当前Diffusers query |
| --- | ---: | ---: | ---: |
| tau=1，同粒度普通近似 | 72.75ms | 77.55ms | 115.19ms |
| TopK10普通近似基线 | 45.26ms（100%） | 50.04ms（100%） | 72.48ms（100%） |
| reblock plan，独立计时 | 14.89ms（32.89%） | 14.89ms（29.75%） | 18.13ms（25.02%） |
| reblock路径净增量，固定plan | 0.28ms（0.62%） | 0.13ms（0.27%） | 5.29ms（7.30%） |
| reweight净增量，固定同一plan/排列 | 0.90ms（1.98%） | 0.90ms（1.79%） | 3.98ms（5.49%） |
| 完整Spark调用，含实时plan | 59.88ms | 64.36ms | 97.80ms |

tau→TopK的收益另以tau耗时为分母：

| 项目 | ComfyUI block | ComfyUI query | 当前Diffusers query |
| --- | ---: | ---: | ---: |
| 耗时减少 | 27.50ms / 37.79% | 27.50ms / 35.47% | 42.72ms / 37.08% |
| 含plan完整Spark相比同粒度tau减少 | 17.70% | 17.00% | 15.10% |

## 不应混淆的量

- reblock净增量=重排后普通TopK耗时−未重排普通TopK耗时，排除plan。
  它同时包含重排和布局变化对后续attention的影响，**不是独立搬运kernel耗时**。
  Diffusers另有独立显式搬运测量5.90ms；ComfyUI搬运已融合，
  无法把上述0.28/0.13ms称为纯gather/scatter时间。
- reweight净增量=同一排列、同一粒度下，weighted summary−普通summary。
  本表比例以未reblock的TopK10为分母，故与上一报告以重排后基线为分母的百分比略有不同。
- plan独立计时、其他路径净差值和完整调用来自不同measurement boundary；
  不是一条trace的可加和分段。完整调用耗时是直接测量，不是把各行相加。
- block对照不与Diffusers query构成同粒度质量对齐。不能由本表推导视频质量。
- 新结果未改变之前的结论：query总耗时没有突然优化数毫秒；
  修正的是reweight归因，新增block模式减少的是近似query工作量。

## 结论

reweight绝对和相对增量均低于当前Diffusers；reblock融合后的路径净增量也较小。
plan绝对时间更低，但占TopK基线的比例仍更高：block约32.89%、query约29.75%，
Diffusers约25.02%。当前值得关注的附加开销仍是plan，而不是把逐query近似成本再次记作reweight。

原始文件位于 `/autodl-fs/data/h3_experiments/spark_cross_pipeline_profile_20260925/`：

- `comfy_latest_stages_gpu0_20260926.json`
- `comfy_latest_stages_gpu1_20260926.json`
- `diffusers_current_reweight_grain_20260926.json`

脚本：`scripts/profile_comfy_reweight_increment.py`、
`scripts/profile_diffusers_vs_comfy_spark_20260925.py --current-diffusers`。
