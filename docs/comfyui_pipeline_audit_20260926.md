# ComfyUI 完整路径复核（进行中）

本轮要解释真实去噪耗时，不把独立 capture 上的微基准差值相加作为端到端预测。
已有微基准仍只代表其标注边界；不能据此排除 producer、调度和布局转换的开销。

## 固定条件

- 复用上一轮四条 prompt 和本地 conditioning；seed=42、1344×768、5s/10s。
- 20次模型评估（21个 sigma 网格点）、前4次 dense、layer 0 dense，无4步 LoRA。
- Sol tau=1、extra=0；Spark TopK10、fanout16、reuse=1、FP32 global anchor，分别测 block/query。
- GPU0–3 各负责一条 prompt；同一 GPU 顺序测试三个实现，方法顺序轮换。
- 每个方法/长度先执行排除在计时外的5步预热，再做无插桩与有插桩的20步配对运行；奇偶 GPU 反转配对顺序。

## 计时与归因

运行脚本：`scripts/comfyui_pipeline_audit_20260926.py`。
正式复核目录：`/autodl-fs/data/h3_experiments/comfyui_pipeline_audit_v2_20260926`。
汇总工具：`scripts/summarize_comfyui_pipeline_audit.py`。

临时 opt-in profiler 位于 ComfyUI 的 `custom_nodes/h3_pipeline_audit.py`，只在
`H3_PIPELINE_AUDIT=1` 的服务器进程安装运行时 hook；默认不启用计时。
记录同一次执行中嵌套 CUDA event 范围、父子关系、模型评估序号、层号。

分解覆盖 QKV projection、Spark producer 中的布局 gather/缓冲区 copy、
reblock plan、Spark attention、Sol fused producer/core、输出恢复与投影、dense attention、MLP。
未归属部分保留为 residual，不武断归为某个 kernel。

必须分别报告：

1. 无插桩的真实 sampler 耗时，用于判定加速。
2. 插桩开销、每步/层分布及计时树闭合。
3. 同一父范围的 exclusive 时间；不得把父范围与子范围相加。

CUDA event 区间包含 host launch 或调度造成的 GPU 时间线空洞，exclusive 不等于
孤立 kernel 活跃时间。Spark attention 范围包含辅助 stream 完成后的 join。
需要继续细分时，对选定真实层/步补充 kernel trace，而不是用独立测试替代。

## 优化约束与后续

先从实际完整路径定位，再验证单项优化，最后重新进行未插桩配对测试。
不修改 Diffusers 生产路径；不降低稀疏预算、不引入 reuse=2、不偷偷改变量化或近似语义。
候选实现需通过数值正确性和生成结果检查，收益不足或数值异常则不作为默认实现。

此前48次生成的 latent/计时全部保留；其 VAE 解码与质量后处理已暂缓，以释放 GPU
用于当前优先复核。此文件尚不报告任何本轮性能结论，也不表示 PSNR/VBench 已完成。

## 测量脚本自检与候选测试

初版发现连续相同工作流只改变保存路径时，sampler 会命中缓存。脚本已拒绝这类
小于10秒的无效记录；初版不是完整配对结果。v2 在两次正式执行之间插入排除计时的
5步预热，改变 sampler 上游调度器签名来触发重新去噪，正式20步参数不变。
初版保留的真正执行记录仍可用于诊断，不当作完整四prompt统计。

初版单条10s Spark block trace：784次稀疏调用，producer 平均31.55ms，
plan14.77ms，native attention47.23ms；producer 中 buffer copy 合计平均4.26ms，
布局 gather1.03ms，输出布局恢复1.42ms。这些是同一执行的嵌套计时，不是消融净增量；
尚需与同长度 Sol 配对解释差值。

独立候选 `scripts/probe_comfy_qkv_direct_destination.py` 使用现有 CUDA RMS/RoPE
显式输出地址，省去 Q/K 中间结果的两次复制。tokens=19/4096/73565、rot=32/64/128
的9种情况中 Q/K/V 均逐位一致。73565 tokens、rot128 的后投影片段中位数约
7.07→5.06ms；只说明该片段有可测收益，不是整层或完整去噪速度。
原始结果保存在初版实验目录 `qkv_direct_probe_gpu3.json`。尚未改为生产默认实现。

## 无人值守实验链

已启动三个有依赖关系的进程，后续阶段等待前一阶段明确完成，不抢占其 GPU：

1. 完整路径复核：`comfyui_pipeline_audit_20260926.py`，四prompt、两长度、三实现，配对插桩/无插桩。
2. 候选筛选：`comfyui_producer_screen_20260926.py`，10s query、同一prompt，四卡分别比较4K/8K/16K/32K chunk，均有同卡 Sol/旧Spark 对照、正反顺序两轮和 latent 数值检查。
3. 四prompt验证：`comfyui_producer_validate_20260926.py`，选取最快且输出有限的候选，复测5s/10s的 Sol、旧/候选 Spark block/query，各两轮，并记录完整20次模型评估。

筛选只产生待验证候选；有限值检查不等于质量达标。四prompt验证也不会自动修改生产
默认实现。若候选改变数值，保留差异，不能仅凭速度宣布优化成功或把 latent 指标称作
视频 PSNR/VBench。达到 Diffusers 的加速比例仍需看结果，预设实验链不是无条件保证。
上游失败时下游停止，已有结果保留；复核阶段允许一次自动重试，避免重复写入成功记录。

结果分别写入实验根目录下 `comfyui_producer_screen_20260926` 和
`comfyui_producer_validate_20260926`；`status.json` 区分等待、运行、失败和完成。

## 完成的四 prompt 实测与生产接入

上述三个实验阶段均已完成。以下为四条 prompt、各两轮无插桩 Sampler 耗时的平均值，
单位秒。16K 表示实验候选，在正式源代码改动前测得。

| 时长 | Sol tau=1 | 旧 Spark block | 16K Spark block | 旧 Spark query | 16K Spark query |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5s | 93.69 | 97.77 | 96.36 | 98.83 | 97.48 |
| 10s | 251.24 | 245.83 | 243.17 | 249.95 | 246.98 |

10s 的 16K block 相对 Sol 加速 1.033×，与此前 Diffusers 的约 1.034×
量级接近。两个 pipeline 的实现和数据不完全相同，这只能比较加速比例，不能
解读为绝对耗时对齐。10s query 为 1.017×；5s 两种模式仍慢于 Sol。
四条 prompt 的 5s/10s、block/query 候选与旧 Spark 的 audio/video latent
均逐位相同；实验程序另验证了每个正式测量都有20次模型评估。

真实路径计时中，5s 的 Spark reblock plan 约9.83ms/次，10s 约15.0ms/次。
5s 官方 Sol 的 route/tail/exact 约19.93ms/次；Spark block 核心约15.37ms/次，
query 核心约16.62ms/次。5s 的 plan 开销超过核心节省，是剩余减速的主要来源。
10s Spark block 核心约47.27ms/次，plan约15.02ms/次；Sol 核心约69.29ms/次。
这些数字来自完整去噪执行中的同一层级计时，不能与父范围重复求和。

16K 候选已接入本仓库 ComfyUI 生产 producer：连续视频区间直接切片，
Q/K RMSNorm+RoPE 写入最终缓冲区，保持原有 reblock、route、reweight 配置。
`tests/test_comfyui_plugin.py` 的11项测试已通过。未启用实验插件的生产路径
独立复测：5s query 96.99s，10s block 243.39s；两个结果的 audio/video latent
均与实验候选逐位一致。原始文件见
`/autodl-fs/data/h3_experiments/comfyui_producer_production_smoke_20260926/results.json`。

5s 额外分项测试表明，9.83ms/次的 plan 中约8.06ms为聚类图重放，
方向矩阵计算约0.26ms。5s 的视频 token 数不被64整除，因而未走10s使用的
双路融合根节点。试过让完整块走融合根节点，同时保留尾部处理；单prompt
约95.06s，仍慢于Sol，且相对原Spark的video/audio latent相对L2误差
分别约30%/13%。已撤回该实验代码，保留原始结果于
`/autodl-fs/data/h3_experiments/comfyui_unaligned_root_trial_20260926/results.json`。

另试过把非对齐长度的 Q/K 聚类计划拆成两张独立图，并行提交，保持每行的
分数和选块过程不变。四条prompt、各两轮的video/audio latent全部逐位一致，
但5s block仅96.36→96.18s（相对旧版7/8次胜），query仅97.48→97.40s
（6/8次胜）；两者相对Sol仍为0/8次胜。收益不足，已撤回实验代码。
原始结果保存在
`/autodl-fs/data/h3_experiments/comfyui_split_plan_validate_20260926/results.json`。

## 5s 聚类内核热点与下一步

在真实5s ComfyUI去噪第7次模型评估、第25层的 reblock plan 采集了一次
CUDA kernel trace；同次完整去噪的784次plan均值为9.83ms，其中聚类图
重放8.06ms。trace中的活跃 kernel 时间（不同stream可能重叠）主要分布：

| Kernel 类别 | 调用数 | 累计活跃时间 |
| --- | ---: | ---: |
| 分数计算 `_score` | 3 | 1.85ms |
| 节点划分 `_fused_node_split_kernel` | 3 | 1.78ms |
| BF16矩阵乘法 | 2 | 1.47ms |
| `gatherKthValue` | 28+4 | 0.98ms |
| 路由键压缩 `_compact_keys` | 9 | 0.79ms |
| 范数 `_squared_norm_kernel` | 1 | 0.77ms |
| 中点方向 `_fused_midpoint_directions_kernel` | 3 | 0.76ms |

原始trace：
`/autodl-fs/data/h3_experiments/comfyui_5s_plan_cuda_trace_20260926/plan_trace.json`。
这些是一次选定层/步的kernel活跃时间，不能把它们与多层均值直接相加。

以当前5s四prompt均值计，block要追平Sol至少还需减少约2.66s，折合784次
稀疏调用每次约3.4ms；要达到10s/Diffusers的约1.034×加速比例，则需
每次约7.3ms。query要达到同一比例需每次约8.8ms。因此调度级0.1–0.2s的
收益不足。下一阶段应针对分数计算、节点划分和路由选择做ComfyUI独立的
内核优化，并以相同四prompt配对测量与latent/视频质量检查为准；不能靠增加
plan跨层复用或减少topk预算凑速度。
