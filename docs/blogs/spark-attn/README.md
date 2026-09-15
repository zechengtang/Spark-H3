# Spark-Attn: Better Sparse Attention for MiniMax-H3

September 13, 2026 · Updated September 15, 2026 · SparkH3 Team

> Adapted for [Spark-MiniMax-H3](https://github.com/zechengtang/Spark-MiniMax-H3)
> from the MiniMax-H3-Sparse research blog. Results below are historical source
> experiments, not measurements rerun in this repository. Cited reports are
> preserved as [source snapshots](references/README.md); implementation links
> point to this repository's port.

[Run Spark inference](#run-spark-inference) · [Preview the blog and animations](PREVIEW.md)


Block-sparse attention reduces the cost of long video sequences by computing selected blocks exactly while either discarding the rest or approximating their contribution through a compressed branch. However, fixed block partitioning and the Jensen gap introduced by block-mean estimation can compromise the fidelity of these methods.

We introduce **Spark-Attn** to mitigate these two limitations. It consists of two components:

- **Spark-Reblock** partitions query tokens and their corresponding key/value tokens according to the learned attention affinities of each layer and head: keys that attract similar queries are grouped together, as are queries that attend to similar keys. This organization aims to retain more attention mass for the same block budget, enabling greater sparsity at a given level of attention coverage.
- **Spark-Reweight** reduces the Jensen-gap bias introduced by block-mean scoring, improving the accuracy of the compressed branch.

## Spark-Reblock

### Why fixed blocks leave quality on the table

Block-mean scoring works best when tokens within a block have similar attention preferences, while different preferences fall into different blocks. Fixed spatial layouts only approximate this goal. Sol’s flat64 layout groups consecutive tokens; [FastH3’s VSA](https://haoailab.com/blogs/fasth3-preview/) uses local spatiotemporal tiles, including 4×4×4 tiles of 64 tokens. Neither layout adapts its block membership to the current attention head.

There are three reasons to reconsider this assumption for video diffusion.

**Noise weakens the spatial prior.** Denoising starts from independent Gaussian noise and gradually recovers structure. Nearby positions in the noise do not have an inherent similarity advantage. A spatial neighborhood that makes sense in the final image may be a poor group earlier in generation.

![Denoising from noise to data](https://yang-song.net/assets/img/score/denoise_vp.gif)

*Illustration: [Yang Song, Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/).*

**Attention preferences change across layers and heads.** Q/K projections, normalization and RoPE transform the representation before attention. The useful question is whether two queries attend to similar keys—not simply whether their image coordinates are close. A single fixed layout cannot capture all these preferences.

**Coarse tokens cover larger regions.** Wan2.1, Wan2.2 A14B and HunyuanVideo 1.0 use 4×8×8 VAE downsampling; Wan2.2 TI2V and HunyuanVideo 1.5 use 4×16×16. H3 also uses 4×16×16, followed by patchification to an effective 4×32×32 attention-token stride. A 4×4 spatial tile then spans a nominal 128×128 pixels, potentially crossing multiple structures. This motivates checking similarity in attention space rather than assuming it from proximity. Sources: [Wan2.1](https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_t2v_14B.py), [Wan2.2 A14B](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_t2v_A14B.py), [Wan2.2 TI2V](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py), [HunyuanVideo 1.0](https://huggingface.co/tencent/HunyuanVideo), [HunyuanVideo 1.5](https://huggingface.co/tencent/HunyuanVideo-1.5), [H3 VAE](../../../vae/config.json) and [patch configuration](../../../transformer/config.json).

### Group tokens by attention preference

Clustering methods such as K-means improve within-group similarity, but create uneven groups that need extra work to fit fixed GPU tiles. They also repeatedly compare every token with every center. With N/64 centers, the assignment cost is O(N²d·n_iter/64), where d is the head dimension.

Spark-Reblock replaces global clustering with **balanced hierarchical partitioning**. At each stage, a small set of representative tokens defines a few split directions. Tokens are divided by those directions under fixed capacity constraints, then each subsequence is split again until every leaf contains 64 tokens. With fixed branching and local iteration budgets, the partitioning scales as **O(N log N)**.

The distance follows attention responses. For example, two keys are close when they respond similarly to the current queries:

$$
\mathbb E_q[(q^\top(k_a-k_b))^2]
=(k_a-k_b)^\top\mathbb E[qq^\top]\,(k_a-k_b).
$$

We use the corresponding Mahalanobis-weighted cosine to group similar response directions, separately for each layer and head. Q is reordered independently; K and V move together, and outputs return to the original Q order. [Spark-Reblock implementation](../../../h3_sparse_attention/landmark_tree_v2.py).

### Watch the blocks reorganize

The animation keeps the attention values fixed and moves their rows and columns into more coherent blocks. With the same exact-block budget, more attention mass falls inside the selected blocks in this illustrative example. Colors represent attention preferences, not spatial positions; the small blocks stand in for the implementation's 64-token blocks.

<iframe class="spark-animation" src="animations/spark-reblock.html" title="Spark-Reblock: better blocks, same attention" loading="lazy"></iframe>

[Open the Reblock animation](animations/spark-reblock.html). Use Pause, Replay or the progress slider to inspect each stage. This shows the effect of the final permutation; it does not animate the recursive tree construction or report model measurements.

### Current comparison: 50 prompts at 768p

We compare **Dense**, **Sol-Attn**, **Sol-Attn + Spark-Reblock**, and **Sol-Attn (Top-K 10%) + Spark-Reblock** on the same 50 prompts. Videos are 10 seconds at 1344×768, with 240 output frames at 24 fps, seed 42 and 20 requested denoising steps. PSNR, SSIM and LPIPS use the matching Dense video as the reference.

| Method | Denoising s/video ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| Dense | 578.216 | ∞ | 1.000000 | 0.000000 |
| Sol-Attn | 373.626 | 19.6353 | 0.689055 | 0.229460 |
| Sol-Attn + Spark-Reblock | 392.001 | 23.1901 | 0.787919 | 0.145838 |
| Sol-Attn (Top-K 10%) + Spark-Reblock | 345.602 | 21.1202 | 0.729122 | 0.195131 |

The threshold-routed reblock configuration improves PSNR by **3.55 dB** over Sol-Attn, with **18.38 seconds** more denoising time per video. The fixed Top-K configuration improves PSNR by **1.48 dB** while taking **28.02 seconds less** than Sol-Attn. These are descriptive differences between separate generation runs; timings include first-use compilation and exclude model loading. The Top-K comparison changes the attention budget as well as the block organization.

Both reblock configurations use 32 midpoint landmarks, Cholesky metric factorization, normalized FP16 cosine scoring, BF16 feature tables and fanout (16,16). They also use a fixed temporal first split with a minimum of 10 latent frames, realized as six groups of 12 latent frames in this experiment, and disable forced local-neighbor blocks. Reblocking remains independent per head and **Spark-Reweight is disabled**. The threshold arm uses tau=1. The Top-K arm uses block-mean Q·K scores with a **GEMM + radix cutoff**, selecting 113 of 1,134 video blocks per query block; context/sink blocks remain exact outside that quota. The original Sol-Attn baseline retains its original tau=1 configuration.

VBench scores below use the same dimension-specific prompt memberships across all four methods: **14 subject, 17 background, 14 motion, 19 imaging and 19 aesthetic prompts** from the 50-prompt set. Scores are scaled to 0–100, higher is better. This is a five-dimension comparison, not an overall VBench score.

| VBench dimension ↑ | Dense | Sol-Attn | Sol-Attn + Spark-Reblock | Sol-Attn (Top-K 10%) + Spark-Reblock |
| --- | ---: | ---: | ---: | ---: |
| Subject consistency | 90.5170 | 90.4674 | 90.8370 | 90.6217 |
| Background consistency | 94.0076 | 94.1772 | 94.0178 | 94.0139 |
| Motion smoothness | 99.0153 | 98.9646 | 99.0085 | 98.9844 |
| Imaging quality | 72.1242 | 72.1476 | 71.9900 | 71.7848 |
| Aesthetic quality | 67.8895 | 67.5251 | 67.7571 | 67.9993 |

The VBench averages remain close to Dense, with mixed changes across dimensions. Higher fidelity to the Dense reference does not imply improvement in every VBench dimension or every prompt. LPIPS uses AlexNet v0.1, RGB in [-1,1] and an arithmetic mean over aligned frames.

[Four-way comparison](references/reports/min10_no_local_vbench20pct_20260915/REPORT.md) · [Threshold reblock results](references/reports/min10_tau1_no_local_vbench20pct_20260915/results.json) · [Top-K reblock results](references/reports/min10_topk10_no_local_vbench20pct_20260915/results.json) · [Exact configuration](references/reports/min10_topk10_no_local_vbench20pct_20260915/settings.json) · [Dense and Sol baseline](references/reports/vbench20pct_768p10s_seed42_20260913/report.md).

### Earlier comparison: 25 prompts at 480p

On 25 prompts, changing only the block organization substantially improves fidelity at similar sampled sparsity:

| Method | PSNR ↑ | SSIM ↑ | Sampled sparsity |
| --- | ---: | ---: | ---: |
| Sol | 18.96 | 0.663 | 69.45% |
| Sol + Spark-Reblock | 23.24 | 0.802 | 70.53% |

These are 5-second, 832×480 videos with seed 42 and 20 requested steps. The comparison uses flat initialization and Mahalanobis cosine. LPIPS was not recorded for this historical run; VBench did not show a uniform improvement across dimensions. [Full results and settings](references/reports/lmv2_initial_order_25/report.md).

## Spark-Reweight

### Why block means underestimate attention mass

Better blocks make their means more representative. But mean scoring still has a fundamental bias: the exponential of a mean is smaller than the mean of exponentials. For logits z within a block, the difference in log space is the **Jensen gap**:

$$
g=\log\operatorname{mean}(e^z)-\operatorname{mean}(z)\ge0.
$$

Why does mean scoring nevertheless work reasonably well for selecting blocks? Much of the gap is shared across key blocks in the same query row. A common shift disappears under softmax and leaves rankings unchanged. The residual differences between blocks matter more than the absolute bias.

Our local diagnostics support this explanation:

| Observation | Measured result |
| --- | ---: |
| Row-common share of squared gap, query-block level | 96.10% |
| Row-common share of squared gap, query-token level | 92.72% |
| Mean-score vs. exact-mass top-147 overlap | 81.66% |

These are averages over five real Q/K fixtures with the original contiguous layout. They explain why a biased mass estimate can still be a useful routing score; they do not imply exact rankings. [Common-shift study](references/SOL_JENSEN_COMMON_SHIFT_CAUSAL_VERDICT.md).

**The cancellation breaks when exact and compressed branches are combined.** Exact blocks retain their true mass, while mean-compressed blocks are discounted by their Jensen gap. In a separate diagnostic covering 61,440 query rows, the compressed branch’s median mass underestimate was **3.468 nats—about 32×**. Its contribution is therefore strongly suppressed relative to the exact branch. Simply increasing its weight is insufficient: keeping a poor mean V while correcting the mass worsened output error in the same study. [Branch-mismatch results](references/SOL_LSE_MECHANISM_DIAGNOSTIC_VERDICT.md).

### Restore the compressed branch’s contribution

Spark-Reweight addresses both problems with one shared query per block. For a query-block mean a, we compute its exact attention scores against every key in each K block B. Those within-block softmax weights produce weighted summaries k̃ and ṽ, while log-sum-exp preserves the block’s total mass ℓB(a). Each query q in the group then uses

$$
\widehat\ell_B(q)=\ell_B(a)+(q-a)^\top\widetilde k_B/\sqrt d
$$

as the compressed block's log-mass, paired with ṽ. These contributions are normalized together with the exact branch. At the shared query a, both mass and the value-weighted numerator are exact; nearby queries reuse a first-order mass approximation. Grouping queries with similar preferences makes this sharing more effective. [Spark-Reweight implementation](../../../h3_sparse_attention/sol_numerator_virtual_q.py).

### Watch the compressed contribution recover

The example below separates the two corrections for clarity. First it restores the compressed block's attention mass using log-sum-exp. Then it replaces the mean value with the within-block attention-weighted value. Correcting only the mass still leaves an inaccurate output. Both corrections together match the Dense output at the shared query a; other queries use an approximation.

<iframe class="spark-animation" src="animations/spark-reweight.html" title="Spark-Reweight: correct attention mass and value together" loading="lazy"></iframe>

[Open the Reweight animation](animations/spark-reweight.html). All displayed numbers come from the four-token toy example shown in the animation, not a model benchmark. The implementation computes the mass and weighted summaries together.

### Share higher in the tree

One representative per 64 queries requires **O(N²d/64)** anchor–key scoring: **1.56%** of the dot products in full QK attention. We can reuse Spark-Reblock’s hierarchy to share a representative across larger query groups while keeping 64-token exact routing.

In a balanced four-way tree, a parent covers 256 queries and a grandparent covers 1024. Their summary-scoring costs fall to O(N²d/256) and O(N²d/1024). Actual trees can have unequal groups, so the cost follows the number of shared representatives. These savings apply to summary construction; exact attention, summary aggregation and per-query evaluation still contribute to runtime.

### Results

Our 25-prompt reweighting ablation shows that sharing higher in the tree retains much of the fidelity gain:

| Compressed branch | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| Block mean | 20.87 | 0.724 | 0.190 |
| Spark-Reweight, 64-token groups | 21.50 | 0.746 | 0.174 |
| Spark-Reweight, parent groups | 21.54 | 0.748 | 0.175 |
| Spark-Reweight, grandparent groups | 21.47 | 0.742 | 0.178 |

This is a separate experiment with a matched mean baseline and an additional 10% exact-block budget, using the same video size, seed and requested steps as above. The measured hierarchy has average parent/grandparent group sizes of 112.5/450 tokens. The 64-token version improves PSNR on 20 of 25 prompts; coarser sharing preserves the average benefit but does not improve every sample. LPIPS uses AlexNet v0.1 on aligned RGB frames. [Spark-Reweight results, timings and protocol](references/reports/virtual_q_levels_20260913/REPORT.md), [hierarchy details](references/reports/virtual_q_levels_20260913/RESULT.md).

Together, Spark-Reblock makes fixed-size blocks better reflect attention preferences, and Spark-Reweight gives compressed blocks a more faithful contribution to the output. The current evidence supports improved fidelity. The next engineering question is how much of that benefit we can retain while reducing the complete attention pipeline’s latency.


## Run Spark inference

This repository provides the full Spark attention path through
`install_h3_spark_attn`, including reblocking, Top-K routing, and reweighting.
Install the optional attention package from the repository root:

```bash
python -m pip install -e '.[cuda]'
```

With an already loaded H3 pipeline:

```python
from h3_sparse_attention import install_h3_spark_attn

with install_h3_spark_attn(pipe.transformer, num_inference_steps=20) as attention:
    result = pipe(..., num_inference_steps=20)
    stats = attention.summary()
```

The current installer uses 10% native-mean Top-K routing, `(16, 16)` landmark
fanout, a 10-frame temporal minimum, and query representatives targeting 189
physical blocks (bounds 94–284). It disables forced local-neighbor blocks after
reblocking and keeps context keys and context queries exact. These defaults
include reweighting, so they differ from the reblock-only benchmark arms above
and from the historical 25-prompt reweighting ablation.

See the [installation and configuration guide](../../../h3_sparse_attention/README.md)
for grid requirements, overrides, warmup behavior, and standalone
`spark_reblock` / `spark_reweight` usage. Plain Sol-Attn remains available with
`install_h3_sol_attn`.
