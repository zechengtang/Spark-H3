# Sol 近似 tail 的粒度差异与兼容实现

## 本地代码核对结论（2026-09-26 更新）

该差异成立，属于近似算法粒度差异，不只是 kernel 调度差异。
已进一步核对本地官方仓库原始代码：NVlabs/Sana 的 sol-engine 分支
`8e0db4fa562d727ea28b8d63015c196db7d97cae` 和 Comfy-Org/comfy-kitchen
`f61028a7b0f4be4beb3595c64ee919e0c345f96d`，确认前者为 query 粒度，后者为 block 粒度。

设序列长度为 N，block size=64，block 数 M=ceil(N/64)。忽略 batch/head：

| 路径 | 路由依据 | 未选中 block 的近似分支 |
| --- | --- | --- |
| ComfyUI 原有 Sol | query 块均值与平均 key 的分数 | 按 query 块计算，共享 tail 的 softmax/output 累积状态 |
| Sol-Engine 官方原始实现及本地 Diffusers 路径 | 真实 query 与平均 key 点积，再沿 query 块归约取均值；外部 route 模式另行提供选择 | 保留逐 query 点积，计算每行的近似 softmax/PV |

两者路由分数在实数运算下都基于 query 均值，但量化、归约顺序、
阈值比较、TopK 边界、强制 local/sink 等差异可能改变实际选块，不能声称逐位相同。
ComfyUI 共享的是 **tail 状态**，不是加上 exact 分支之后的最终输出。

逻辑近似分数规模分别约为 M×M 与 N×M；不意味着完整矩阵一定会
写入显存，也不代表整个 attention 都只有该复杂度。

源代码依据：

- `../Sana/techniques/sparse_backends/sol_attn/sm120/mainloop.py` 的上述提交：
  真实 Q 与 pooled K 的 GEMM → `reduce_route_columns`/`col_mean` 路由归约；
  原逐 query 分数继续经 `apply_route_mask` → `online_softmax_route` → pooled PV。
- `../comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_route.cu`：
  文件开头的 centroid/shared-tail 说明、`cen8` QK、pooled PV 与 `o_part/m_part/l_part`。
- `/autodl-fs/data/h3_experiments/attn_kernel_speedup_20260921/snapshot/sol_attn/sm120/mainloop.py`：
  `gemm_smem_zero_acc(... tSrS, tSrQ, ...)` 产生逐 query 分数；
  `reduce_route_columns` 和 `col_mean` 用于路由；未选中的列在
  `apply_route_mask(tSrS, ...)` 后继续进入 `online_softmax_route` 和 pooled PV。

## 对既有性能结论的修正

ComfyUI 相对这个 Diffusers 快照的速度优势包含更粗粒度近似带来的
计算量优势，不能全部归因于实现效率。加入逐 query global reweight 后，
两侧新增工作不同。已有约 6.77 ms 与 4.72 ms 不能直接作为同等工作量的
kernel 效率比较；前者还使用主动关闭 tail 的诊断基线，后者是替换已有
近似分支的增量，reblock 条件也不完全相同。

应先对齐 tail 语义、路由/强制 exact 规则、reblock 输入和测量范围，
再解释剩余性能差距，并测量近似粒度对质量的影响。以前将差异归结为
“只缺高效调度，做好即可达到 4.72 ms”的解释不成立。

## 新增兼容接口

### Spark：K/V reweight 与 query 粒度解耦

**reweight 是 K/V 侧校正，不是逐 query 计算的同义词。** 两种模式共用
`launch_spark_reweight_tokens`：同一 global anchor、同一加权 K/V summary、
同一 log-mass 修正和 INT8 carrier；summary 构建函数不接收 query 粒度参数。

`comfy_kitchen.backends.cuda.spark_attn(..., tail_granularity="query" | "block")`：

- `query`：保持已有默认值，真实 query 消费校正后的 summary，融合在 exact kernel 中。
- `block`：新增 centroid consumer，每个 query block 只消费一次校正后的 summary，
  将共享 `o/m/l` 交给原有 exact kernel。exact 分支仍用真实 query；不退回逐 query tail。

ComfyUI Spark 节点与 `ComfySparkConfig` 同步暴露 `tail_granularity`，默认 query，
避免静默改变已有工作流。它独立于 `global_anchor_dtype`，也不改变视频尾块策略。
ComfyUI 实际固定采用 dense 尾块：从 `video_tokens // 64` 开始的 KV/query
block 均作为 dense sink。`video_tail_mode` 仅保留 `dense` 以兼容已有工作流；
此前界面中的 `pad` 没有实现，只保存了配置，实际仍执行 dense。
现已移除该选项，并对程序化传入 `pad` 显式报错。这里的尾块策略不是
`tail_granularity` 所控制的近似粒度。Diffusers 路径不改动。

两种消费模式保留各自数值路径：block 使用 centroid INT8 Q、BF16 中间共享
状态，再计算 exact；query 在 exact kernel 中累计 summary。因此不要求两种模式
输出逐位一致，也不能把差异只解释成 reweight 精度。普通 Sol 的接口如下。

CUDA `sol_attn` 和 `sol_attn_chunked` 增加 `tail_granularity`：

- `"block"`：默认，保持原有官方共享 tail 行为。
- `"query"`：路由不变，每个真实 query 消费平均 K/V summary。
- `tail=False` 时不开启任何近似分支；此时粒度不影响结果。
- 首版 query 模式不支持 `token_aug>0`，显式报错，不静默退回 block 模式。

公共 `comfy_kitchen.sol_attn` 及 eager reference 也支持该选项；HIP 的新模式
尚未实现，不会声称其可用。示例：

```python
import comfy_kitchen as ck

block_out = ck.sol_attn(q, k, v, tau=1.0, tail_granularity="block")
query_out = ck.sol_attn(q, k, v, tau=1.0, tail_granularity="query")
```

首版沿用官方的 centroid 路由，仅替换近似分支。用平均 V 加
`log(block_length)` 的 summary logit bias 保持块质量，含非完整尾块。
summary 与 exact 在同一个现有 CUDA attention kernel 中合并，
不分配 N×M 的逐 query score 矩阵，不依赖 Diffusers/Triton/CuTe。

这是 **粒度语义兼容**，不是 Diffusers 的数值或性能复刻：仍采用
ComfyUI 的 INT8 Q/K/P/V 计算路径，summary V 也经过量化；路由仍从
centroid 分数产生，没有复现 Diffusers 从逐 query QK 归约路由并复用
该分数的调度。因此不能仅凭启用 query 模式就声称两边所有工作量完全对齐。
默认配置和 Spark global reweight 算法不因该选项改变。

chunked query 模式另行使用 producer 实际采用的 K centering（可为上一轮
统计量）构造 pooled K；路由仍用原有 centroids。这样 exact/tail 的中心化
平移一致，不会在合并 softmax 时引入额外偏置。原有 block 模式不改动。

## 验证结果

### Spark 双粒度 reweight（2026-09-26）

后续同粒度 reweight 开销消融见
[独立测量记录](comfyui_reweight_granularity_increment_20260926.md)，
该实验不再以关闭整个近似 tail 作为 reweight 的对照。

- 新增 26 项测试，连同普通 Sol 测试共 170 项通过；插件 10 项通过。
- 覆盖 BF16/FP16 输入、BF16/FP32 anchor、batch/head、ragged 尾块、
  全近似和 exact/summary 混合的独立浮点参考、sink、fused reblock。
- 两种模式的 route bitmap、anchor、有效 INT8 K/V summary 和 scale/bias
  逐位一致；block 全近似输出块内共享；默认 query 与 48 个历史边界输出逐位一致。
- CUDA 扩展已编译。没有修改 Diffusers 源码。

GPU0–1 重放同一真实 10s768p 输入 `[1,73565,56,128]`，video=72576，
TopK10、FP32 global anchor、相同 conditioning sinks；**无 reblock/plan**。
预热 4 次、CUDA event 测量 12 次，两卡反向执行，取两卡中位数平均：

| Spark 消费粒度 | attention 耗时 | 相对 dense 的单层 MSE | 相对 L2 |
| --- | ---: | ---: | ---: |
| block + global reweight | 45.96 ms | 0.035184 | 0.072908 |
| query + global reweight | 50.59 ms | 0.032550 | 0.070127 |

block 耗时下降约 9.1%，但该层误差更大；不是两种粒度等价或视频质量相同的证明。
此表也不是单独 summary 构建耗时、完整 Spark-H3 pipeline 耗时，
更不是生成视频 PSNR/VBench；不应与不同路由规则的普通 Sol TopK 数字直接相减。

复现实验：`scripts/profile_comfy_spark_tail_granularity.py`。
原始结果：`/autodl-fs/data/h3_experiments/spark_cross_pipeline_profile_20260925/`
下的 `spark_tail_granularity_gpu0.json` 和 `spark_tail_granularity_gpu1.json`。

### 此前普通 Sol 双粒度验证

- comfy-kitchen：原有 113 项测试与新增 31 项测试，共 144 项。
  新增覆盖 eager query 参考、BF16/FP16、257/1000/4097 长度、
  tau/TopK、跨步与 batch 输入、block_len/key bias、sink query、
  chunked 启动及故意偏移的 stale centering、公开 fake dispatch。
- 原生 workspace 中读取 route 的 idx/cnt，验证 block/query 两种模式
  在 tau 和 TopK 下选中的列表完全一致。
- 默认 block、全 exact、tail=False 的相关输出逐位不变；
  单独的 48 个 Spark 边界用例也与本轮之前的保存结果逐位一致。
- MiniMax-H3 的 9 项 ComfyUI 插件测试通过。Diffusers 源码未修改。

### 单层真实输入 smoke benchmark

GPU0–1（SM120）重放同一份已有 10s768p attention 输入：
`[1, 73565, 56, 128]`，video tokens=72576；conditioning KV/query sink
规则保持一致。每个模式预热 4 次、CUDA event 测量 12 次；两张卡的
测试顺序相反，下表为两张卡各自中位数的平均。没有 reblock。

| 路由设置 | block tail | query tail | 增量 |
| --- | ---: | ---: | ---: |
| tau=1 | 72.46 ms | 77.62 ms | +5.15 ms（+7.11%） |
| TopK10 | 44.88 ms | 50.35 ms | +5.46 ms（+12.18%） |

同一输入与 fused Flash SDPA 输出比较（FP64 指标累积）：

| 路由设置 | block → query 的 MSE | block → query 的相对 L2 |
| --- | --- | --- |
| tau=1 | 0.017413 → 0.016718 | 0.051291 → 0.050257 |
| TopK10 | 0.041158 → 0.040198 | 0.078856 → 0.077931 |

该输入上 query 模式稍微降低误差，同时增加耗时。两张 GPU 的四种输出
哈希完全一致，默认 tau=1 block 输出哈希与修改前一致。
**这只是一个真实 attention 层的验证，不是整段视频的 PSNR/VBench，
也不能推广为所有 prompt 的质量结论。**

复现脚本：`scripts/profile_comfy_sol_tail_granularity.py`。
原始结果：`/autodl-fs/data/h3_experiments/spark_cross_pipeline_profile_20260925/sol_tail_granularity_gpu{0,1}.json`。
核心代码位于相邻 `../comfy-kitchen` 工作树；本仓库记录说明及测量脚本。

### 后续对照要求

新的 query 模式让近似分支粒度可对齐，但比较 Spark/Diffusers 时还必须
统一 local 强制 exact、TopK 边界与预算、conditioning sink、reblock 输入
和量化方案。尤其不能直接用本表的 query TopK 耗时减历史 Spark 耗时，
然后宣称已实现同口径的 reweight 提速。当前仍保留官方 TopK 的阈值路由
流程；该模式没有顺带改变其重复 route 或边界策略。
