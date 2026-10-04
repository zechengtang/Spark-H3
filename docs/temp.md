这并不真正违反 Taylor 理论，因为“中心点展开更准确”需要满足几个条件，而当前 route 场景没有完全满足。最关键的是：**均值 route 和 global-anchor Taylor route 实际上在逼近不同的目标。**

原始 mean route：

\[
s^{mean}_{ij}=\bar q_i^\top \bar k_j
\]

它不仅可以解释成零点附近的近似；由于点积是双线性的，它还是该 block 所有 Q–K token pair 平均 logit 的精确值：

\[
\bar q_i^\top\bar k_j
=
\mathbb E_{q\in Q_i,k\in K_j}[q^\top k]
\]

所以 avgpool 对“平均相关性”完全没有近似误差，而且方差很低、排序稳定。

修正后的 Taylor route 逼近的是另一个非线性量：

\[
G_{ij}=
\log\mathbb E_{q\in Q_i,k\in K_j}
\exp(q^\top k/\sqrt d)
\]

其近似为：

\[
\tilde q_i^\top\tilde k_j/\sqrt d+b_{Q,i}+b_{K,j}
\]

它更接近 attention mass，但不必更适合决定“哪些 block 值得 exact”。

## 1. Global mean 并不是每个 block pair 的局部中心

所谓中心是全视频、单个 head 的全局 Q/K 均值，而不是当前 \(Q_i,K_j\) 的中心。实际评估点是每个局部 block；运动主体、局部纹理等 block 可能离 global mean 很远。

Taylor 展开只保证在展开点附近误差较小，并不存在：

\[
|G-T_{\text{global mean}}|
<
|G-T_0|
\]

对所有 block 成立的保证。尤其经过 RoPE 后，全局平均还会发生方向抵消，它未必是具有实际语义的 anchor。

## 2. Route 的目标不是预测 attention mass，而是选择需要 exact 的 block

Global reweight 已经使用同一个全局 anchor 去近似未选中的 block。因此，与 global anchor 高度匹配的 block，往往正是 reweight 近似得最好的 block。

新的 route 又按照 global-anchor Taylor mass 选择 exact block，可能导致：

- 将 exact 预算花在已经能被 reweight 良好近似的全局常见模式上；
- 漏掉 attention mass 未必最高、但 reweight 残差很大的局部运动或主体 block。

理想 route 应排序的是：

\[
\text{exact attention error}
-
\text{reweight approximation error}
\]

而不只是 attention mass。

## 3. 双侧 reweight 会强化共同模式、削弱 block 区分度

\[
\tilde q_i=\operatorname{softmax}(\bar k_{\rm global}^\top q)Q_i
\]

会把 Q block 拉向全局常见 K 方向；K 侧也做同样的事情。随后再点积，背景、静态结构和全局共性会被两次强化，而局部特异 token 被压低。

bias 修复了每个 block 的总 mass，但不能恢复被 centroid 改变的方向和 block 间排序。这也符合结果：

- 加入 bias 后 PSNR、SSIM、LPIPS 全部明显恢复；
- 但仍没有追上 mean route。

说明“缺少 bias”是一个问题，但“两侧 global-anchor centroid 改变 route 排序”是更主要的问题。

## 4. Top-K 排序和迭代扩散会放大很小的近似误差

Taylor 误差可能在数值上不大，但 Top-K=10% 是离散选择。cutoff 附近一次很小的分数变化，就会交换 exact/approximate block。

这种交换发生在：

- 多个 head；
- 多层；
- 15个稀疏 denoising evaluation；
- 总计735次 route 调用。

因此，即使 Taylor score 的平均标量误差更小，也可能有更差的 Top-K recall，并在扩散过程中逐步积累成明显 PSNR/LPIPS差异。

所以目前最合理的结论是：

> Global-anchor Taylor 可能更准确地估计了全局 attention mass，但 mean route 更准确地选择了“必须进行 exact attention 的 block”。

下一步最有信息量的不是再跑50条，而是抽取少量真实 Q/K，直接比较三类 route 对 oracle 的排序：

- exact block attention mass；
- exact 与 global-reweight 输出之间的 block residual；
- mean、key-only Taylor、two-sided Taylor 的 Top-K recall。

我尤其怀疑 **key-only + bias** 会优于当前 two-sided 版本：保留 query block 的原始均值和局部区分度，只对 K 侧使用完整 Taylor：

\[
s_{ij}
=
\bar q_i^\top\tilde k_j(\bar q_{\rm global})/\sqrt d+b_{K,j}
\]

它更符合 Q→K attention 的非对称结构，也不会让 Q/K 两侧同时向全局共同模式收缩。
