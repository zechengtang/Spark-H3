# Spark reweight 原理与推导

本文对应当前仓库的 `h3_sparse_attention` 实现。可在 VS Code 中打开本文件并使用 Markdown 预览（`Ctrl+Shift+V`）阅读公式。

## 1. 问题：稀疏注意力如何处理未精算的块

对一个 query $q\in\mathbb R^d$，标准注意力是

$$
O(q)=\frac{\sum_{i=1}^{T}\exp(s_i(q))v_i}
           {\sum_{i=1}^{T}\exp(s_i(q))},
\qquad
s_i(q)=\frac{q^\top k_i}{\sqrt d}.
$$

本实现中 $d=128$，物理 K/V 块通常含 64 个 token。Top-K 路由决定哪些块逐 token **精确计算**。若直接丢弃其余块，这些块对 softmax 分母及输出的贡献都会消失。Reweight 为它们建立摘要，再与精确部分合并。

下文固定一个 batch、一个 attention head 和一个未精算的 K/V 块 $B$；其他维度的计算彼此独立。

## 2. 为一组 query 建立代表点

先把 query 分成若干组（代码称 *virtual parents*）。组 $P$ 的代表 query 或 *anchor* 为

$$
a_P=\frac{1}{|P|}\sum_{q_j\in P}q_j.
$$

同一组 query 共享基于 $a_P$ 建立的 K/V 块摘要。Spark 默认预设将目标视频 query 汇成每个 head 的一个全局代表；也可以设置多个组。代表点的构造见 [`_anchors`](../h3_sparse_attention/sol_numerator_virtual_q.py) 和 [`build_virtual_anchors`](../h3_sparse_attention/sol_numerator_virtual_q.py)。

注意：**query 组与 K/V 物理块是两种划分**。前者决定共用哪个 anchor；后者决定路由和摘要的单位。每个 query 组都会针对各个 K/V 块建立摘要。

## 3. 在代表点处计算块摘要

对 K/V 块 $B$，先计算 anchor 对块内 token 的 softmax：

$$
z_i=\frac{a_P^\top k_i}{\sqrt d},\qquad
Z_B(a_P)=\sum_{i\in B}e^{z_i},\qquad
p_i=\frac{e^{z_i}}{Z_B(a_P)}.
$$

然后保存三个量：

$$
\bar k_{P,B}=\sum_{i\in B}p_i k_i,
\qquad
\bar v_{P,B}=\sum_{i\in B}p_i v_i,
$$

$$
m_{P,B}=\log Z_B(a_P)-\frac{a_P^\top\bar k_{P,B}}{\sqrt d}.
$$

它们在代码里分别叫 `AK`、`AV`、`LM`，由 [`virtual_summaries`](../h3_sparse_attention/sol_numerator_virtual_q.py) 生成。末尾不满 64 个 token 的块只统计实际 token，不把填充行计入 $Z_B$。

### 从熵理解 log-mass

把整个块压缩成一个代表 key 时，$a_P^\top\bar k_{P,B}/\sqrt d$ 描述这个代表的**响应强度**，但一个代表本身无法表达“块里有多少 token 在共同响应”。$m_{P,B}$ 补上这部分丢失的 softmax 质量。

这个修正项有一个很直观的身份。因为 $\log p_i=z_i-\log Z_B(a_P)$，且 $\sum_i p_i z_i=a_P^\top\bar k_{P,B}/\sqrt d$，所以在精确算术下：

$$
\boxed{\quad m_{P,B}
=\log Z_B(a_P)-\sum_{i\in B}p_i z_i
=-\sum_{i\in B}p_i\log p_i
=H(p_B).\quad}
$$

也就是说，**log-mass 修正恰好是 anchor 在该块内产生的 softmax 分布的熵**。$\exp(m_{P,B})$ 可以理解为块内的“有效 token 数”：

| 块内响应 | softmax 权重 | log-mass $m_{P,B}$ | 有效 token 数 $e^{m_{P,B}}$ |
| --- | --- | ---: | ---: |
| 一个 token 几乎独占 | 接近 $(1,0,\ldots,0)$ | 接近 $0$ | 接近 $1$ |
| $n$ 个 token 响应相同 | 每个都是 $1/n$ | $\log n$ | $n$ |

例如，一个块有 64 个得分都为 0 的 key，旁边还有一个得分也为 0 的单独 key。真实 softmax 中，这个块贡献 64 份质量，单独 key 贡献 1 份，块占 $64/65$。若把块压成一个得分为 0 的代表，却不加修正，两边只会各占一半；加上 $\log 64$ 后，块的份量恢复为 64 份。

反过来，若块内只有一个 key 明显占优，其余 key 的权重近乎 0，修正项便接近 0。**它描述的是权重的集中程度，而不是 key 向量本身的方差，也不是响应的绝对高低**：给块内所有得分同时加上同一常数，熵不变；整体响应强度仍由代表 key 的得分承担。由于当前实现会将加权 key 存为 BF16/FP16，并可选择不同的舍入方式，上述熵等式在实际张量上可能有数值偏差。

## 4. 为什么也能用于组内其他 query

对任意真实 query $q\in P$，块的精确 log-mass 是

$$
F_B(q)=\log\sum_{i\in B}
  \exp\!\left(\frac{q^\top k_i}{\sqrt d}\right).
$$

它在 $a_P$ 处的梯度为

$$
\nabla F_B(a_P)
=\sum_{i\in B}p_i\frac{k_i}{\sqrt d}
=\frac{\bar k_{P,B}}{\sqrt d}.
$$

因此，一阶泰勒展开给出

$$
F_B(q)\approx F_B(a_P)
 +(q-a_P)^\top\frac{\bar k_{P,B}}{\sqrt d}
=\underbrace{\frac{q^\top\bar k_{P,B}}{\sqrt d}
 +m_{P,B}}_{\widehat F_B(q)}.
$$

这正是实际 query 访问摘要块时使用的分数。它在 $q=a_P$ 处的 **log-mass 值和梯度均与精确计算一致**；偏离 anchor 越远，一阶近似通常越可能不准。

更具体地，$F_B$ 的 Hessian 为

$$
\nabla^2F_B(q)=\frac{1}{d}\operatorname{Cov}_{p(q)}(k_i),
$$

所以二阶误差取决于块内 key 在 $q-a_P$ 方向上的离散程度。精确算术下，log-sum-exp 是凸函数，上述切线满足 $\widehat F_B(q)\le F_B(q)$。**这个下界只针对块的 log-mass**，并不保证最终 attention 输出是下界，也不保证有限精度实现逐位满足该关系。

## 5. 块的 value 如何近似

精确块的 softmax 加权 value 会随 $q$ 改变：

$$
V_B(q)=
\frac{\sum_{i\in B}e^{s_i(q)}v_i}
     {\sum_{i\in B}e^{s_i(q)}}.
$$

Reweight 使用在 anchor 处求得的 $\bar v_{P,B}=V_B(a_P)$ 作为该块对组内 query 的 value。因此，块的近似分母与分子为

$$
\widehat D_B(q)=e^{\widehat F_B(q)},
\qquad
\widehat N_B(q)=\widehat D_B(q)\,\bar v_{P,B}.
$$

在精确算术下，两者在 $q=a_P$ 都等于真实块的分母与分子。但 $V_B(q)$ 被固定为 $V_B(a_P)$，因此**分子的导数一般并不精确**；它漏掉了 key 与 value 在块内的协方差效应。不能把整个输出称为一阶精确近似。

## 6. 与精确块合并

令 $E(q)$ 为路由选中、逐 token 计算的精确 key 集合，$S(q)$ 为未精算的 K/V 块集合。最终输出近似为

$$
\widehat O(q)=
\frac{
  \displaystyle\sum_{i\in E(q)}e^{s_i(q)}v_i
  +\displaystyle\sum_{B\in S(q)}e^{\widehat F_B(q)}\bar v_{P,B}
}{
  \displaystyle\sum_{i\in E(q)}e^{s_i(q)}
  +\displaystyle\sum_{B\in S(q)}e^{\widehat F_B(q)}
}.
$$

实现使用稳定的 softmax / log-sum-exp 累积。流式路径先计算精确部分，再按两部分各自的 log-sum-exp 权重合并；见 [`_skip_merge_chunk`](../h3_sparse_attention/sol_numerator_virtual_q.py)。生产长度上的融合路径把精确块与摘要块放进同一 attention mainloop；见 [`_fused_virtual`](../h3_sparse_attention/sol_numerator_virtual_q.py)。两条路径表达的是同一个“精确块 + 摘要块”结构。

## 7. 与路由、重排的区别

| 步骤 | 决定什么 | 结果 |
| --- | --- | --- |
| `reblock` | 哪些 token 放在同一个物理块 | 改变 64-token 块的组成 |
| Top-K route | 哪些物理块需要精确计算 | 生成精确块集合 $E(q)$ |
| `reweight` | 未精算块如何代表其 K/V 与 softmax 质量 | 生成 $\bar k$、$\bar v$、$m$ |
| merge | 精确与近似结果占多少权重 | 生成最终 $\widehat O(q)$ |

单独调用 `spark_reweight(a, k, v)` 只生成摘要，**不会**选择路由或生成最终 attention 输出。

## 8. 当前实现的可选变体

- `sol_reweight_components="full"`：使用加权 K/V 和 log-mass 修正，即上面的完整公式。
- `"weights_only"`：使用加权 K/V，但质量项退回 $\log |B|$。
- `"bias_only"`：K/V 使用普通均值，但仍计算质量修正；修正项中的 key 使用这个普通均值。
- `"none"`：普通均值 K/V 加 $\log |B|$，作为同一 Spark 框架内的对照。

这些开关的实现见 [`_summaries`](../h3_sparse_attention/sol_numerator_virtual_q.py)。`sol_tail_granularity` 则控制摘要分支按每个 query、每 64 个 query，或每 8 个 query 计算近似输出；它不改变 Top-K 路由选出的精确块。FP32/BF16 anchor、summary 计算模式以及 `stored` / `pre_round` 的 log-mass 计算会改变有限精度数值，见 [`H3SparseAttentionConfig`](../h3_sparse_attention/processor.py)。
