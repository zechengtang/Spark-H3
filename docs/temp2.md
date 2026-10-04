是的，你指出得对。之前写的

\[
\log \mathbb E[e^X]
\approx \mathbb E[X]+\frac12\operatorname{Var}(X)+\cdots
\]

只是累积量展开的截断形式。完整表达是：

\[
\log \mathbb E[e^{tX}]
=\sum_{n=1}^{\infty}\frac{\kappa_n}{n!}t^n
\]

令 \(t=1\) 后：

\[
\log \mathbb E[e^X]
=\mathbb E[X]
+\frac12\operatorname{Var}(X)
+\frac16\kappa_3
+\frac1{24}\kappa_4+\cdots
\]

其中 \(\kappa_n\) 是各阶累积量。因此“均值加一半方差”只是二阶截断；当 logit 分布偏斜、长尾或跨度较大时，高阶项可能并不小。

更重要的是，本次实验并不能说 LME 带来了明确的性能提升：

- 端到端耗时从 `341.98s` 增加到 `437.48s`，约 `+27.9%`。
- SSIM、LPIPS略有提升。
- PSNR 略降 `0.046 dB`。
- VBench 基本持平且总体偏弱，Background 下降最明显。

所以它的价值主要是诊断性的：证明 Taylor/global-anchor 近似确实损失了部分路由准确性；但即使把 K 侧 LME 算精确，这种“更准确的 attention-mass 代理”也没有转化成显著的视频质量收益。

原因很可能是目标错配：

1. LME 强调少数极高 logit，但路由会保留整个 block。
2. 当前仍使用 \(\bar q\)，没有覆盖不同 Q token 的需求。
3. TopK 是离散选择；分数更准确，但如果没有改变 cutoff 附近的 block 排名，生成结果不会变化。
4. 路由只观察 Q/K，没有考虑被选 block 的 V 是否真正重要。
5. `mean(qk)` 会压制局部极值，可能恰好起到了稳定化和覆盖正则的作用。

因此工程结论相当明确：**当前没有理由用精确 LME 替代均值路由。** 它显著增加计算量，却只带来微小且不一致的指标变化。若继续研究，更合理的是检查 cutoff 附近的换位率，或者尝试低成本的

\[
\operatorname{mean}(qk)+\lambda\operatorname{Var}(qk)
\]

其中 \(\lambda\) 可小于理论二阶系数 \(1/2\)，而不是继续追求完整 LME 或更高阶展开。
