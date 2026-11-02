# Spark chunked 实现说明

本文记录当前保留的两种实现及其分块范围。两种方案都通过 chunk producer 分块生成 K；区别在于 reblock 规划期间全局保存原始 K，还是保存变换后的 K2。

## 1. 总体设计与“分块”的含义

### SelfLift-Avatar drop_k 接入

`prepare_split_attention` 独立完成 reblock，返回排列、统计及 RoPE 等状态；drop_k 的 K2 在函数返回前释放。随后 `execute_split_attention` 才分配工作区，通过共享 Sol producer 执行 QKV 重放，再调用 Spark BSA/reweight。原有 `spark_split_attention` 保持兼容，顺序调用这两个入口。

Spark attention callback 暴露 `selflift_drop_k`，SelfLift 通过它接入自己的分块 norm、QKV 投影、输出投影、residual 和 MLP；规划采样与重放均保持原始位置及逐 token 调制参数对应。`global_stream` 沿用 Sol 的默认 8-head 分组和 block 权重准备，显式 KJ head 分组优先，避免重复分组。该深度接入仅用于 drop_k；keep_k 和普通 Spark 保留之前的 callback 路径。RAM QKV 暂存不在本次范围内。

从功能上看，Spark 由 BSA、reblock 和 reweight 组成。沿用 Sol chunked 的 producer 思路时，需要单独处理的是具有全局依赖的 reblock plan：不能简单地把各个 producer chunk 独立聚类，再将局部结果拼接成同一个全局规划。

当前执行方式为：

1. 从少量采样 Q 构造变换 T，准备规划所需的 K 或 K2。
2. 对整个稀疏视频域完成 reblock plan，得到 token permutation；`q_reuse_k` 模式下 Q、K 共用该排列。
3. 分块投影和预处理后续所需的 QKV 或 Q/V，将结果写入全局量化工作区。
4. 在原生计算核中应用 permutation，执行路由、BSA 和 Spark 的 reweight。

这里需要区分三个层次：

| 层次 | 当前实现 |
|---|---|
| 线性投影与 producer | 按 token chunk 执行，避免完整浮点 QKV 同时驻留 |
| reblock plan | 保持全局规划语义；其内部的变换、评分计算仍可以分块 |
| attention 计算 | producer 填充全局量化工作区后，由原生块级计算核执行 |

因此，“reblock 单独处理”指全局规划独立于后续 producer 阶段，并不表示规划内部所有操作都禁止分块。当前也不是每生成一个 QKV/QV chunk，就立即完成该 chunk 的全部 BSA 和 reweight。全局 permutation 先得到，量化张量的实际重排在后续原生核中完成。

两种方案使用与 `sol_attn_chunked` 相同的 `sol_attn_plan` 工作区格式，复用 Sol 的 route/exact attention 主干。Spark 另外执行排列相关处理和 reweight，因此整个后续阶段并非与 Sol 完全相同。

每层用于下一步的 K 均值和 V scale 是跨步缓存，必须通过 comfy-kitchen 的 `allocation_context()` 分配；ComfyUI 将该上下文映射为暂停临时内存编译图。若直接在 block 的临时分配域创建这两个统计张量，它们返回后仍被层状态引用，会导致反复图失效和采样过程中的显存累积。此处理只改变统计量的分配生命周期，不改变其计算，也不暂停整个 reblock 或 attention 的内存编译。

## 2. 术语

- **原始 K**：完成线性投影、norm/RoPE，但尚未经过聚类变换 T 的 K。它是 attention 使用的 K，不是投影后未经归一化的位置编码前张量。
- **变换 T**：由少量采样 Q 构造的逐 head 小矩阵。当前 Q 采样数约为稀疏视频 token 数的 1/64。
- **K2**：原始 K 经 T 变换后用于聚类的特征。它不能直接替代 attention 的原始 K。
- **全局量化 K**：原始 K 经 BSA producer 量化后写入工作区的表示，包含相应尺度；它与聚类特征 K2 是两种不同的表示。

以下流程聚焦两种方案的差异。BSA 所需的量化尺度及统计初始化属于共同机制，不作为 Sol/Spark 或这两种分块方案的区分依据。

## 3. 方案 1：`drop_k`

全局保存 K2，规划后重新分块投影 QKV。

1. 仅投影采样位置的 Q，完成 norm/RoPE，构造 T。
2. 按 chunk 投影 K，完成 norm/RoPE。
3. 将当前 K chunk 中属于稀疏视频域的部分立即变换为 K2，写入全局 K2 表，然后释放临时 K chunk。
4. 基于完整 K2 表完成全局 reblock plan，得到 permutation。
5. 释放 K2 和 T，再分配后续 attention 工作区。
6. 重新按 chunk 投影完整 QKV，使用 Sol producer 进行预处理并写入量化工作区。
7. 执行量化张量重排、路由、BSA 和 reweight，输出结果。

**此方案不会先保存完整原始 K，再额外分配完整 K2。** 规划输入构建期间，全局保留的是 K2，原始 K 仅以临时小块存在。K2 表只覆盖参与聚类的视频域，后续 attention 工作区覆盖完整序列。

代价是 K 的线性投影需要在后续 QKV producer 中再次执行。优势是聚类直接读取已经计算好的 K2，无需在树划分过程中反复执行变换。

## 4. 方案 2：`keep_k`

全局保存原始 K，聚类按需变换，后续仅分块投影 Q/V。

1. 仅投影采样位置的 Q，完成 norm/RoPE，构造 T。
2. 按 chunk 投影 K，完成 norm/RoPE，将每块结果写入全局 K 缓冲区，最终保存完整 K。
3. 全局规划过程中，根据当前所需的 token，通过 K 和 T 按需计算变换特征及划分指标，得到 permutation。
4. 释放 T；从保留的原始 K 生成全局量化 K，写入 attention 工作区。
5. 释放原始 BF16 K。
6. 仅按 chunk 投影 Q、V，并直接读取各自的投影结果完成 producer；不重新投影 K，也不额外拼接 Q/V 张量。
7. 执行量化张量重排、路由、BSA 和 reweight，输出结果。

按需变换不是完全融合成“不产生任何 K2 临时张量”的评分核。当前实现会生成有界的 K2 特征块，交给评分/划分核使用后释放；默认特征块预算为 262144 行，即 head_dim=128、BF16 下约 64 MiB。这样可以沿用参考评分核的计算顺序，保持排列一致性，同时避免保存完整 K2 表。

**完整原始 K 只保留到量化 K 生成完毕。** 后续 Q/V producer 和 attention 使用工作区中的全局量化 K 及其尺度，不继续保留完整 BF16 K。

## 5. 两种方案的共同点与取舍

| 项目 | 方案 1：`drop_k` | 方案 2：`keep_k` |
|---|---|---|
| K 的生成方式 | 按 chunk 投影 | 按 chunk 投影 |
| 规划期间全局保留的特征张量 | 变换 K2 | 原始 K |
| 是否保存完整 K2 表 | 是，仅稀疏视频域 | 否，按需暂存小块 |
| 后续投影 | 分块 QKV，重新生成 K | 分块 Q/V，复用 K |
| 原始 BF16 K 的生命周期 | 仅临时 chunk | 规划及 K 量化期间 |
| 最终 attention 工作区 | Sol 格式量化工作区，加 Spark 辅助状态 | 相同 |

方案 2 省去了后续重复 K 投影，并减少 Q/V producer 的临时输出；代价是聚类中按需变换可能重复计算，以及分别投影 Q、V 的调度成本。因此，它不保证在所有输入上同时拥有更低峰值和更快总速度。

如果 reblock 阶段的峰值已经低于后续 BSA/reweight 阶段，那么最终峰值落在共同的 attention 工作区和输出上，符合这次拆分的目标。此时，两种方案的 attention 峰值可能相同，不能仅凭这一点判断规划优化无效，也不能用“一份 K 相对完整 QKV 的 1/3”直接推断整个模型的显存降幅。

## 6. 当前入口与代码

底层同时保留两种方案。Spark 节点统一使用 `chunk_path`：`none`（默认）表示非分块 Spark，`drop_k` 表示方案 1；节点不再提供 `use_chunked` 和 `chunked_scheme`，暂不开放 `keep_k`。旧 Python `spark_attn_chunked` 调度已从源码删除，共享的底层 BSA/reweight 原生核继续保留。

两种方案均可与 KJNodes 的 `MiniMaxLowVRAMAttention` 和 `MiniMaxChunkFeedForward` 组合。ChunkFeedForward 保持原有 token 分块；LowVRAMAttention 提供的 `minimax_head_chunks` 由 split producer 读取，形成“外层 head group、内层 token chunk”的二维分块。每个 head group 独立保存 K2 或原始 K、完成全局视频域 reblock 和 Spark attention，结果按原 head 位置合并后只执行一次输出投影。统计缓存键包含 head 范围，组间不复用 K 均值、V scale 或排列。

非 chunk Spark 仍使用原有 KJ 投影后 Q/K/V compose hook；chunked Spark 则保留 KJ 的 block-level 早释放 forward，但必须在投影前接管 QKV，以便执行 K-only、Q/V-only 或 QKV replay。当前实现优先验证显存和语义，head group 会分别运行 reblock/producer，调度开销可能明显增加；它不是性能优化承诺。

- [comfyui_spark_split.py](../comfyui_spark_split.py)：采样 Q、分块 K 生成、两种方案的生命周期、后续 producer 和原生核调用。
- [landmark_lazy_k.py](../h3_sparse_attention/landmark_lazy_k.py)：方案 2 的按需变换与全局划分。
- [producer.cu](../native/spark_split/producer.cu)：单路 norm/RoPE、独立 K 与 Q/V producer。
- [节点配置与构建说明](spark_chunked.md)：支持配置、动态库构建、测试与回放方法。
- [5s768p 实验报告](spark_split_5s768p_20261031.md)：两种方案的显存、耗时和正确性结果；报告中的端到端结果使用当时的环境变量选择入口。
