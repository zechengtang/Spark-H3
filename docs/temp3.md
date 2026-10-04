# 逐 Query 归一化路由方案

核心做法是：先在每个 query 或 query-child 内，对所有 K 父块做 Softmax
归一化；然后在概率空间对同一 Q64 父块内的 query 求平均。不能先把 Q
子块一起 LogSumExp。

## 精确目标

对于 query \(i\) 和 K64 父块 \(b\)，先计算该块的未归一化质量：

\[
L_{i,b}=\log\sum_{j\in b}\exp(q_i^\top k_j/\sqrt d)
\]

然后逐 query 在所有 K 块间归一化：

\[
P_{i,b}=\frac{\exp L_{i,b}}{\sum_{b'}\exp L_{i,b'}}
=\operatorname{softmax}_b(L_{i,b})
\]

一个 Q64 父块 \(A\) 共用同一路由，因此最终路由分数应该是：

\[
R_{A,b}=\frac{1}{|A|}\sum_{i\in A}P_{i,b}
\]

最后对 \(R_{A,b}\) 做 Top-K。这直接最大化选中块所保留的平均注意力质量：

\[
\max_{|S|=K}\frac1{|A|}\sum_{i\in A}\sum_{b\in S}P_{i,b}
\]

最优解正是选择 \(R_{A,b}\) 最大的 K 个块。

## 保持 Q32×K32 复杂度的实现

不必真的对 64 个 query token 分别计算，可以使用两个 Q32 代表：

\[
z_{a,b,c}=\bar q_{A,a}^{\top}\bar k_{b,c}/\sqrt d
\]

其中 \(a\in\{0,1\}\) 是 Q child，\(c\in\{0,1\}\) 是 K child。

第一步，只在 K-child 内聚合：

\[
L_{a,b}=\log\sum_c n_{b,c}\exp z_{a,b,c}
\]

第二步，对每个 Q-child 分别沿 K 父块归一化：

\[
P_{a,b}=\operatorname{softmax}_b(L_{a,b})
\]

第三步，在概率空间平均两个 Q-child：

\[
R_{A,b}=\frac{n_{A,0}P_{0,b}+n_{A,1}P_{1,b}}
{n_{A,0}+n_{A,1}}
\]

然后对 \(R_{A,b}\) 做 Top-K。对应伪代码：

```python
# child_scores:
# [B, q_parent, H, k_parent, q_child, k_child]

log_k_mass = torch.logsumexp(
    child_scores + log_k_counts,
    dim=-1,
)

log_partition = torch.logsumexp(
    log_k_mass,
    dim=3,
    keepdim=True,
)

log_probability = log_k_mass - log_partition

log_route_score = torch.logsumexp(
    log_probability + log_q_counts,
    dim=-1,
) - log_total_q_count

# Top-K is invariant under log, so no final exp is required.
route = topk(log_route_score)
```

当前联合聚合直接执行：

```python
logsumexp(child_scores, dim=(q_child, k_child))
```

它会让 logit 整体较大的 Q-child 获得指数级更高的权重。新方案先对每个
Q-child 消除其 partition function，再平均其概率分布，因此每个
query-child 按 token 数公平贡献。

## 是否加入文本与上下文 token

条件视频归一化只在候选视频块之间计算 partition，衡量“已经决定关注视频
token 时，质量落在哪个视频块”。完整注意力归一化还应加入始终稠密的文本、
sink 和其他上下文 key：

\[
Z_a=\sum_{\text{video }b}\exp L_{a,b}
+\sum_{j\in\text{context}}\exp(q_a^\top k_j)
\]

完整归一化会使主要关注上下文的 query 对视频路由贡献更小，更接近真实
Dense attention mass。实际实现可以对上下文同样采用相邻 K32 均值，以避免
引入昂贵的精确 token 支路。

## 重要性质

Q64×K32 只有一个 Q-child。逐行归一化对该行所有块只是共同常数，不改变
Top-K，因此这种方法只可能改变 Q32×K64 和 Q32×K32 的选择。最值得测试的
方案是 `Q32×K32 normalized-mass`；Max 不对应可加的概率质量目标。

这套方案可称为 `row-normalized attention-mass routing`。它保持 Q32×K32 的
路由点积规模，但新增一次沿所有 K 父块的归一化。实现时应尽量融合
Softmax/LogSoftmax，避免多个独立的 `exp + sum + log` kernel。
