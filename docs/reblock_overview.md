# Reblock 的高层运行逻辑

本文描述当前 Spark-H3 的 landmark-tree-v2 reblock 基线：fanout 最大值 16、flat 初始化、M2 加权 cosine、32 个 midpoint landmarks、2 次中心更新、最终 64-token 块。

## 1. Reblock 做什么？

Reblock 不修改 attention 使用的 Q/K/V 数值，而是重新组织 token，让适合放在一起的 token 进入同一个 64-token 块。

这样做有两个目的：让重要 attention mass 更集中在少量块中，使固定 Top-K 预算更有效；让未选中块的压缩近似更容易准确。它不保证两个目标一定同时改善，需要实验验证。

整个过程可以理解为：

```text
当前层、当前步的原始 Q/K/V
          │
          ▼
采样 Q/K，构造对侧二阶矩 M2
          │
          ▼
生成仅供聚类使用的 Q/K 特征
          │
          ▼
从 flat 顺序开始，按容量递归划分
  小规模 landmarks 拟合分裂方向
  → 当前节点全部 token 按方向分配
          │
          ▼
得到各自的 Q64、K64 布局及置换索引
          │
          ▼
用原始 Q/K/V 做路由、摘要和 attention
          │
          ▼
把 Q 侧输出恢复到原始 token 顺序
```

Q 和 K 各自构造布局，V 跟随 K；不是用同一套排列同时重排 Q/K/V。各 batch/head 的布局根据其数据计算。在启用 sparse attention 的层和步上使用当前输入重算；缓存执行计划不等于复用旧 token 分组。

## 2. 为什么先做 M2 变换？

Q 的聚类应考虑“这些 Q 与 K 交互时有多不同”，而不仅是 Q 自身的几何距离。因此 Q 使用对侧 K 的二阶矩，K 使用对侧 Q 的二阶矩。

概念上：

\[
M_K=\mathbb E[kk^\top],\qquad M_Q=\mathbb E[qq^\top].
\]

加上按 trace／维度缩放的 ridge 后，通过 Cholesky 得到因子，构造聚类特征：

\[
X_Q=Q L_K,\qquad X_K=K L_Q,
\quad L_KL_K^\top=M_K+\lambda_K I.
\]

当前使用非中心化二阶矩，不是减去均值后的协方差，也不是逆协方差 whitening。cosine 距离在这些变换后的特征上计算。

M2 并非用所有 token 直接构造：当前入口沿全局 Hilbert 顺序取均匀 midpoint 样本，样本数为完整 64-token video 块数。这与下面“每个树节点的 32 个 landmarks”是两套不同采样。

**flat 指树的初始 token 顺序；它不意味着 M2 采样也必须使用 flat。** 原始 attention Q/K/V 不被替换成变换后的特征。

## 3. 如何递归生成 64-token 块？

先确定最终需要多少个完整块。例如 37,296 个 video tokens 对应 582 个完整块和 48 个余数 token。

树的每个节点拥有一组 token 和一个“最终叶块预算”。划分时先确定子节点各自需要承载多少个叶块，每个子节点容量必须是 64 的整数倍。

fanout=16 表示一次外层划分最多产生 16 个子节点，不表示每个节点一定产生 16 个。当前 `power_of_two_fanout` 调度按剩余容量选择实际分支数。递归到一个节点只需一个叶块时停止，得到严格的 64-token 块。

因此这不是普通无容量限制的 k-means：即使自然类簇大小不同，也必须满足规定容量，而不是先自由聚类再大量 padding。

余数 token 单独放入 exact 保护尾部。当前实现根据聚类特征范数选择排除 token，不能把它理解为原始序列最后的几个 token。context 不参与 video reblock，继续按既有规则保持 exact。

## 4. 每个节点的 landmarks、种子和 iteration

直接在一个大节点的全部 token 上反复更新中心太贵，因此先用少量 landmarks 拟合分裂方向，再一次性处理全部 token。

当前每个节点沿其继承顺序划分区间，取区间 midpoint 特征作为 landmarks，并以区间大小作为权重；默认目标数量为 32，实际数量可按节点规模调整。它们是 token 特征代表，不是视频 prompts，也不是最终的 64-token blocks。

产生多个子节点的方向拟合，内部通过一棵小型二分 proxy 树完成。每个 proxy 二分节点：

1. 从当前有效 landmarks 中选择 cosine 距离最远的一对，作为初始左右中心。
2. 根据到两个中心的 cosine 差值给 landmarks 排序，按左右容量分配权重。
3. 用分配后的权重计算原始变换特征的左右均值中心。
4. 重复第 2、3 步一次，即总共 2 次中心更新。
5. 将最终两个中心归一化，构造分裂方向；子 proxy 节点继续处理对应的 landmark 权重。

这里的中心更新保留特征模长，是 `raw` 加权均值，不是先将每个 landmark 单位化再求均值。cosine 比较时再做中心归一化。

随后用拟合得到的方向给当前外层节点的全部 token 打分，并通过容量约束路由生成子节点。`parent_order` 保留每个子节点中继承的相对 token 顺序。

**“2 次更新”是 2 次小规模 landmark 分配与中心更新，不是对全部 token 做 2 次完整聚类。** 最终全 token 打分和容量划分另外执行。fanout=16 时，proxy 树包含 15 个二分节点，而不是一次普通的 16 中心 Lloyd 更新。

若消融 0 次更新，应直接使用初始种子构造方向，但仍正确完成子 proxy 节点的 landmark 分配和最终 token 容量划分；不能简单跳过当前循环中的所有操作。

## 5. Reblock 和路由、近似支路有什么区别？

| 模块 | 回答的问题 | 当前基线 |
|---|---|---|
| Reblock | 哪些 token 放进同一个块？ | M2＋cosine 递归容量划分 |
| Route | 哪些 Q64×K64 块值得精确计算？ | 块均值打分，Top-K 10% |
| 近似支路 | 未选中块如何贡献 attention？ | global-anchor 加权摘要 |

Reblock 输出布局和置换索引，随后用该布局中的原始 Q/K/V 构造路由和摘要。实现可以融合索引读取与打包，不必每一步都显式复制全量张量。

同样的 Top-K 预算下，布局不同，选中的块及覆盖的 attention mass 会不同；计算密度相同不等于质量相同。布局 oracle 也各不相同，不能把某一个布局的 oracle 当作所有布局的上界。

## 6. 哪些工作贵，哪些工作小？

- M2 构造及特征变换：包括采样二阶矩、矩阵分解、全 token 特征变换。
- Landmark 拟合：在少量代表上选种子、排序、分配和更新中心；大节点上算术成本通常较小。
- 全 token 分配：对树中各层的全部活跃 token 打分、按容量路由并维护索引；它不会因 landmark 只有 32 个而消失。
- 布局应用：原始 Q/K/V 的索引读取／打包，以及 Q 输出的逆置换。

每个 proxy 二分节点的中心更新复杂度约为 \(O(I(LD+L\log L))\)，其中默认 \(I=2,L=32,D=128\)。一个外层节点若分出 \(F\) 个子节点，需 \(F-1\) 个这样的 proxy 节点。另外还存在 landmark 两两距离与种子选择成本。

增加少量中心更新可能比增加全 token 聚类轮数便宜，但深层节点数量、排序及 GPU 调度也影响实际耗时。因此质量与完整 reblock 耗时应一起测，不能只按 FLOPs 判断。

## 代码入口

- [spark_integration.py](../h3_sparse_attention/spark_integration.py)：采样 M2、特征变换与 prepared plan 调用。
- [landmark_tree_v2.py](../h3_sparse_attention/landmark_tree_v2.py)：外层递归、叶块容量、landmarks 和 token 划分。
- [landmark_v2_cosine_fast.py](../h3_sparse_attention/landmark_v2_cosine_fast.py)：小规模 proxy 二分树与 2 次中心更新。
- [processor.py](../h3_sparse_attention/processor.py)：Spark preset 和 reblock 配置。

以上描述当前基线；环境变量和后端优化可能改变精度或实现路径，实验应保存实际生效配置。后续消融计划见 [prompt_reblock.md](prompt_reblock.md)。
