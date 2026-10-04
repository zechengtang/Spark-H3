# Spark-H3 Reblock 消融：5s768p 完整视频结果与 50p/10s 复验

> 状态：23 个正式配置均已完成 25-prompts 完整视频生成与评价；六方法 5s768p/50-prompts 和
> 10s768p/25-prompts 的生成、PSNR/SSIM/LPIPS、无损视频归档及按 prompt 指定维度的 VBench
> 也均已完成。本文不使用 Dense-QKV replay 粗筛结果。

## 1. 摘要

- **所有候选判断均以 25-prompts 完整视频 PSNR/SSIM/LPIPS 和完整去噪时间为准。**
- 完整视频中，`landmarks64` 是最佳质量—成本点：相对 baseline，25 prompts 的 PSNR、SSIM、
  LPIPS 分别改善 `+0.189 dB`、`+0.0048`、`-0.0066`，去噪耗时增加 3.6%。
  `landmarks128` 的质量最好（`+0.455 dB`、`+0.0114`、`-0.0127`），但耗时增加 16.9%。
- M2×distance 2×2 的完整视频确认 cosine 的优势强依赖当前双侧 M2 变换：双方 M2 开启时，
  cosine 相对 Euclidean 的 all PSNR 高 `1.571 dB`；关闭 M2 后仅高 `0.049 dB`。三个非 baseline
  组合的 PSNR/SSIM/LPIPS 都显著退化，当前 `both M2 + cosine` 应保留。
- 初始 5s/25-prompts 中，`q_reuses_k_layout` 是混合结果：all split 三项质量均改善且去噪快 2.4%，但
  confirmation 仅 PSNR 改善，SSIM/LPIPS 略差。它有 Pareto 潜力，但不能表述为确认集全面胜出。
- 初始 5s/25-prompts 中，`k_reuses_q_layout` 的 all 与 confirmation 三项质量点估计均改善，去噪快 3.79% 且 25/25 prompts
  更快；但质量差值的 paired 95% CI 仍跨 0，因此尚不能表述为质量显著提升。
- 初始 tile 顺序和若干低成本 M2 估计在完整生成中未稳定胜过 baseline；增加 landmarks 是目前
  质量收益最稳定的改动。
- 严格 raw-Q/K 子序列均值并未改善压缩：完整视频 all PSNR/SSIM/LPIPS 分别退化
  `-0.451 dB`、`-0.0136`、`+0.0130`，去噪耗时增加 51.6%。
- Taylor 0+1 的完整视频结果全面退化：
  all 三项为 `-0.574 dB/-0.0180/+0.0216`，去噪耗时增加 67.3%。
- 扩展到 5s/50 prompts 后，`landmarks64` 相对 baseline 的全体质量差为
  `+0.057 dB/+0.0026/-0.0034`，耗时 `+3.7%`；新增 25-prompts confirmation 的 PSNR 点估计
  略降而 SSIM/LPIPS 近中性，因此 5s 扩展集上的收益小于初始 25 prompts。
- 10s768p/25 prompts 复验重新拉开了差异：`landmarks64` 的 PSNR/SSIM/LPIPS 为
  `+0.487 dB/+0.0108/-0.0111`，耗时仅 `+2.2%`，三项 paired 95% CI 均支持改善。
  `q_reuses_k_layout` 快 `4.8%` 且三项点估计改善；`k_reuses_q_layout` 虽快 `4.9%`，但三项
  质量点估计均退化，其中 SSIM 的 paired 95% CI 不跨 0，不能再把它列为质量保持型优先候选。

## 2. 评价口径和 25-prompts 覆盖范围

每个正式配置都生成并评价 25 个完整 5s768p 视频，与相同 prompt、seed 的 Dense 视频配对。报告只使用：

- PSNR、SSIM、LPIPS：最终解码视频相对匹配 Dense 视频的质量；
- denoise time：充分预热后的完整去噪耗时；
- 逐 prompt 胜场和 paired 95% CI：判断均值差异的稳定性。

覆盖范围如下：

| 消融组 | 完整视频配置 | 每配置 prompts | 状态 |
|---|---:|---:|---|
| baseline | 1 | 25 | 完成 |
| M2×distance | 3 个非 baseline 组合 | 25 | 完成 |
| 初始种子／中心更新规则 | 2 | 25 | 完成 |
| landmark 数量 | 2 | 25 | 完成 |
| proxy 更新次数 | 3 个非 baseline 值 | 25 | 完成 |
| M2 estimator | 6 | 25 | 完成 |
| Q/K 布局复用 | 2 | 25 | 完成 |
| landmark 压缩 | 2 | 25 | 完成 |
| root 初始顺序 | 2 | 25 | 完成 |

共计 **23 个正式配置×25 prompts=575 个 Dense-paired candidate videos**，全部完成。

## 3. 实验设置

### 3.1 基线来源与冻结项

实验以 `docs/blogs/spark-attn/README.md` 中 Table 3/4 的 `spark-h3-10pct` 路径和实际运行配置为
依据，而不是仅从表格名称反推参数。固定设置如下：

- 5s768p 代理：120 帧，768×1344，20 denoising steps，seed 42；
- Top-K 10%，Q64×K64 exact tile，mean route，原有近似 attention 支路；
- baseline reblock：flat root order、parent order、fanout=16、32 midpoint landmarks、
  farthest-pair seeds、mean center update、2 次 proxy 更新；
- 双侧对侧非中心化二阶矩 M2 变换、Hilbert midpoint M2 估计、cosine 距离；
- Q/K 独立重排，V 跟随 K，原始 attention Q/K/V 数值不变；
- ridge 系数 `1e-3`，按各自 M2 的 trace/维度缩放；
- 所有候选保持相同 prompt、seed、Dense 参考、context、Top-K 预算和 48 个保护 video-tail tokens。

“中心选择”和“中心更新”是两个不同环节：`farthest_pair`/`endpoint_order` 选择初始代表点；
`mean`/`medoid` 决定固定轮数内的中心更新规则。更新次数是每个 proxy 二分节点在 landmarks 上的
“分配＋更新”轮数；0 次路径仍会生成子节点 landmark 分配并完成最终容量约束划分。

历史实验曾增加 `landmark_compression`，测试 raw-Q/K mean 与 Taylor 0+1。
这两个实验实现及配置入口现已撤回；下文对应的结果仅作历史记录，不能用当前代码复跑。
当前只保留原有的 `landmark_tree_v2_landmark_mode`：默认 `midpoint`，可选
`mean` 表示在 M2 变换后的特征空间取区间均值，包括 `group_size>1` 的用法。

### 3.2 数据集与划分

Prompt 来自：

```text
/mnt/CFS/tangzecheng/repos/MiniMax-H3-Benchmark/
  vbench_core5_percent_subsets/10pct/samples.json
```

共 25 个 prompts：

- tuning（10）：case 1、2、3、8、9、10、11、17、18、19；
- confirmation（15）：case 4、5、6、7、12、13、14、15、16、20、21、22、23、24、25。

实验使用 GPU0–7，均为 NVIDIA A800-SXM4-80GB；每个计时任务只占用一个 GPU，未在同 GPU 上并发
计时。记录所对应的仓库提交为 `57d401b25a1c2c2eb78f3a2483b06999417e033f`，但运行时工作树包含本轮
未提交实验实现；复现时应同时保留当前源码、协议 JSON 和逐样本记录，而不能只依赖该提交号。

### 3.3 八个消融轴

除 M2×distance 的完整 2×2 外，其余均为相对共同 baseline 的单轴比较：

1. M2×distance：both/none × cosine/Euclidean；
2. 初始种子/中心更新规则：farthest-pair vs endpoint-order；mean vs weighted medoid；
3. landmark 数量：32/64/128；
4. proxy 更新次数：0/1/2/4；
5. M2 估计：Hilbert midpoint、flat64 block mean、flat64 midpoint 1/2/4、
   flat64 mean+exact diagonal、full M2；
6. 布局复用：independent、Q reuses K、K reuses Q；
7. landmark 压缩方式：midpoint、mean、Taylor 0+1；
8. root 初始顺序：flat、Hilbert THW、FastH3-compatible tile T4H4W4。

M2 估计均覆盖全部原始 video tokens（包含余数、不含 context），Q 使用 K 的矩阵，K 使用 Q 的矩阵，
定义为非中心化二阶矩而非逆协方差 whitening。flat64 midpoint 使用确定性的真实 token 索引；full M2
以分块 `X^T X/N` 累加，不构造 N×N 矩阵。

### 3.4 M2 估计实现

5s768p 的原始 video token 数为 `N=37296`。按原始 flat 顺序切 64-token block 后，包含 582 个完整块
和一个 48-token 余数块，共 583 块。各 estimator 的定义如下：

| Estimator | 实际样本/代表数 | 构造方法 |
|---|---:|---|
| Hilbert midpoint | 582 | 当前 Hilbert block 的 midpoint token |
| flat64 block mean | 583 | 582 个完整块均值＋48-token 余数块均值，按实际块长加权 |
| flat64 midpoint-1 | 583 | 每块 1 个确定性真实 token |
| flat64 midpoint-2 | 1166 | 每块 2 个确定性真实 token |
| flat64 midpoint-4 | 2332 | 每块 4 个确定性真实 token |
| flat64 mean+diag | 583 个块均值＋全部 37296 token 的对角统计 | block-mean M2 加精确对角方差补偿 |
| full M2 | 37296 | FP32 分块累加 `X^T X/N`，chunk size 4096 |

flat64 midpoint 的块内索引严格使用
`floor((j+0.5)*n_b/s_b)`，其中 `s_b=min(m,n_b)`；索引被保存并验证唯一、范围合法，余数块采用同一
公式。所有候选共享原始 flat 分块，不从 reblock 后布局采样。block mean 使用
`sum_b (n_b/N) mu_b mu_b^T`，因此明确丢弃块内方差。

所有新 estimator 先将输入转为 FP32；归约、矩阵乘、累加和后续 M2 对称化/分解均为 FP32。
full M2 的 `4096`-token chunks 只限制临时乘法输入，不改变非中心化定义，也不构造 N×N 矩阵。
mean+diag 的修正为
`mean_all(x*x) - sum_b (n_b/N) mu_b*mu_b`：以每 batch 精确对角绝对最大值（至少 1）为尺度，
只 clamp 位于 `2e-6 × scale` 容差内的负舍入误差；更显著的负值直接报错。clamp 数在数值测试中
于 clamp 前记录，CUDA 路径保留异步断言。

每种估计都按自己的 `trace(M2)/D` 加 ridge，系数固定为 `1e-3`；随后对称化并使用当前统一分解
路径生成特征因子。这里的 ridge 和分解作用于完整 D×D M2，低成本估计并没有把后续特征变换改成
低秩近似。

FastH3 tile-64 顺序已与官方 FastVideo `main` 提交
`9491c8638adeb266dd37656ba6224b2200db91ea` 的 `(4,4,4)` tile partition 实现逐索引核对；包含
`(9,10,13)`、生产 grid `(37,24,42)` 等边界网格时均完全一致。上游没有
`tile_t4h4w4` 这一同名配置字符串；这是本项目对上游 tile-size-64 顺序的描述性名称。

## 4. 主结果：完整视频质量—成本

### 4.1 评价协议与完整性

每个候选生成 25 个完整视频 latent，并与相同 prompt、seed 的 Dense latent 配对。官方 decoder 在单个
worker 内常驻；同一 case 的 Dense decode 只做一次并复用于候选。评价使用：

- pooled RGB PSNR；
- 逐帧 Gaussian SSIM（11×11，sigma 1.5，valid window）；
- LPIPS AlexNet v0.1，输入 RGB `[-1,1]`；
- 充分预热后的完整 denoising wall time。

LPIPS 使用 learned Alex v0.1 权重
`df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`，其 torchvision AlexNet
backbone 权重 hash 为
`7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`。同时归档了 LPIPS 实现
`lpips.py` hash `780d09b907cb9b661e0ae28b2d163ddfba92f9e870d7feba34d4790cc6590658`；所有 RGB 先验证
finite 且位于 `[0,1]`，再量化为 uint8，并映射到 `[-1,1]`。metric batch=4，TF32 关闭。

Primary、roll01–roll11、roll13 raw-space mean、roll14 Taylor 0+1、roll15 优先级前四项和 roll16
最后两项的生成与质量记录均为 25/25 complete，case、prompt SHA 和 Dense latent SHA 逐项匹配。
roll13 质量文件含 25
baseline＋25 raw-space mean records；roll14 和 roll15 分别只归档 25 条 Taylor、100 条优先级候选质量
记录，roll16 归档 50 条候选质量记录；三者的差值均复用 roll13 的同源码、同 prompt/seed baseline，
而不是声称各 root 内重复生成了 baseline。

完整视频正式集合按配置去重后为 primary 4 arms、roll01–roll11 的 11 arms、roll13 strict mean、
roll14 Taylor、roll15 的 4 arms 和 roll16 的 2 arms，共 23 个方案×25 prompts=575 个 Dense-paired candidate videos；
roll13 为同源码配对另生成的 25 个 baseline 不在此处重复计数。质量评价覆盖每个视频的全部 120 帧，
即 69000 个 paired frames。
all 是 25 prompts 的等权均值，tuning/confirmation 分别是 10/15 prompts 的等权均值；不是按视频
像素数或某个更大的 split 重新加权。

专项测试将数值实现与实验记录分开验收：M2 测试覆盖带余数的 block-mean 显式公式、midpoint-1/2/4
确定性索引的唯一性和范围、mean+diag 与 full M2 的对角一致性、full 分块累加与直接 FP32 参考及
Hilbert 指定索引；proxy 测试覆盖 0/1/2/4 次更新的容量约束树、
0 次更新仍生成有效子节点分配、冻结尾部、raw-space mean/Taylor 公式和所有分裂层生效。在这些
实现级测试之外，roll13 strict raw-space mean 和 roll14 Taylor 各自的 25/25 完整视频进一步验收了
端到端路径。

### 4.2 汇总结果

单元格为“绝对值（相对 baseline 差值）”。PSNR/SSIM 越高越好，LPIPS/时间越低越好。

| 方案 | All PSNR | All SSIM | All LPIPS | All denoise | Confirmation PSNR | Confirmation SSIM | Confirmation LPIPS | Confirmation denoise | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | 22.883 (+0.000) | 0.7915 (+0.0000) | 0.1618 (+0.0000) | 232.9s (+0.0%) | 23.588 (+0.000) | 0.8240 (+0.0000) | 0.1389 (+0.0000) | 232.4s (+0.0%) | 当前 baseline |
| `landmarks64` | 23.072 (+0.189) | 0.7963 (+0.0048) | 0.1552 (-0.0066) | 241.3s (+3.6%) | 23.959 (+0.371) | 0.8339 (+0.0099) | 0.1308 (-0.0080) | 240.9s (+3.6%) | 最佳质量—成本折中 |
| `landmarks128` | 23.338 (+0.455) | 0.8029 (+0.0114) | 0.1491 (-0.0127) | 272.2s (+16.9%) | 23.873 (+0.286) | 0.8315 (+0.0075) | 0.1294 (-0.0095) | 272.2s (+17.1%) | 质量最高，成本明显增加 |
| `k_reuses_q_layout` | 23.207 (+0.323) | 0.7950 (+0.0035) | 0.1571 (-0.0047) | 224.6s (-3.8%) | 24.036 (+0.449) | 0.8328 (+0.0088) | 0.1331 (-0.0057) | 224.6s (-3.8%) | 稳定加速；质量点估计改善但 CI 跨 0 |
| `m2_flat64_midpoint1` | 22.894 (+0.011) | 0.7905 (-0.0010) | 0.1626 (+0.0008) | 236.2s (+1.4%) | 23.550 (-0.037) | 0.8226 (-0.0014) | 0.1401 (+0.0012) | 236.3s (+1.7%) | 无稳定视频收益 |
| `initial_tile_t4h4w4` | 22.854 (-0.029) | 0.7898 (-0.0017) | 0.1623 (+0.0005) | 234.9s (+0.9%) | 23.508 (-0.080) | 0.8235 (-0.0005) | 0.1399 (+0.0010) | 234.8s (+1.0%) | 无稳定视频收益 |
| `m2_flat64_midpoint2` | 22.889 (+0.006) | 0.7901 (-0.0013) | 0.1626 (+0.0009) | 239.1s (+2.7%) | 23.425 (-0.163) | 0.8206 (-0.0034) | 0.1406 (+0.0017) | 239.0s (+2.8%) | confirmation 退化 |
| `m2_full` | 22.848 (-0.035) | 0.7896 (-0.0019) | 0.1636 (+0.0018) | 248.9s (+6.9%) | 23.526 (-0.062) | 0.8239 (-0.0000) | 0.1405 (+0.0016) | 248.8s (+7.0%) | 更慢且无质量收益 |
| `m2_flat64_midpoint4` | 22.957 (+0.073) | 0.7934 (+0.0019) | 0.1601 (-0.0017) | 239.8s (+3.0%) | 23.495 (-0.093) | 0.8217 (-0.0023) | 0.1403 (+0.0014) | 239.7s (+3.1%) | all 收益由 tuning 驱动 |
| `initial_hilbert_thw` | 22.889 (+0.006) | 0.7900 (-0.0015) | 0.1631 (+0.0013) | 234.9s (+0.9%) | 23.569 (-0.018) | 0.8251 (+0.0011) | 0.1385 (-0.0004) | 234.9s (+1.1%) | 接近中性，指标混合 |
| `updates4` | 22.760 (-0.123) | 0.7909 (-0.0006) | 0.1643 (+0.0025) | 332.9s (+43.0%) | 23.511 (-0.077) | 0.8264 (+0.0024) | 0.1403 (+0.0015) | 332.7s (+43.1%) | 成本不可接受 |
| `m2_flat64_mean_diag` | 22.522 (-0.361) | 0.7829 (-0.0086) | 0.1716 (+0.0098) | 280.4s (+20.4%) | 23.140 (-0.447) | 0.8141 (-0.0099) | 0.1500 (+0.0111) | 280.6s (+20.7%) | 全面退化 |
| `updates0` | 22.438 (-0.446) | 0.7750 (-0.0165) | 0.1804 (+0.0187) | 315.7s (+35.3%) | 23.057 (-0.531) | 0.8083 (-0.0157) | 0.1563 (+0.0175) | 315.3s (+35.0%) | 三项质量显著退化且明显更慢 |
| `m2_flat64_block_mean` | 21.920 (-0.964) | 0.7617 (-0.0298) | 0.1900 (+0.0282) | 278.6s (+19.6%) | 22.481 (-1.107) | 0.7935 (-0.0305) | 0.1685 (+0.0296) | 279.2s (+20.1%) | 最差方案之一 |
| `update_weighted_medoid` | 22.157 (-0.726) | 0.7659 (-0.0256) | 0.1913 (+0.0295) | 324.6s (+39.1%) | 22.708 (-0.880) | 0.7995 (-0.0245) | 0.1669 (+0.0281) | 324.6s (+39.0%) | 三项质量显著退化，接近 Dense 成本 |
| `updates1` | 22.834 (-0.050) | 0.7914 (-0.0001) | 0.1654 (+0.0036) | 319.5s (+37.2%) | 23.467 (-0.121) | 0.8219 (-0.0021) | 0.1449 (+0.0060) | 319.4s (+37.4%) | 更慢且更差 |
| `seed_endpoint_order` | 22.718 (-0.166) | 0.7863 (-0.0052) | 0.1673 (+0.0055) | 322.9s (+38.7%) | 23.427 (-0.161) | 0.8210 (-0.0030) | 0.1439 (+0.0050) | 322.8s (+38.9%) | 更慢且更差 |
| `q_reuses_k_layout` | 23.146 (+0.262) | 0.7927 (+0.0012) | 0.1574 (-0.0044) | 227.2s (-2.4%) | 23.881 (+0.294) | 0.8224 (-0.0016) | 0.1392 (+0.0003) | 227.1s (-2.3%) | 有 Pareto 潜力；确认集质量混合 |
| `m2_both_euclidean` | 21.313 (-1.571) | 0.7417 (-0.0498) | 0.2242 (+0.0624) | 234.3s (+0.4%) | 21.802 (-1.786) | 0.7744 (-0.0496) | 0.1992 (+0.0603) | 233.9s (+0.2%) | 三项质量均显著退化 |
| `m2_none_cosine` | 21.405 (-1.479) | 0.7418 (-0.0497) | 0.2253 (+0.0635) | 234.0s (+0.3%) | 21.796 (-1.791) | 0.7733 (-0.0507) | 0.2011 (+0.0622) | 233.6s (+0.1%) | 三项质量均显著退化 |
| `m2_none_euclidean` | 21.356 (-1.528) | 0.7398 (-0.0517) | 0.2274 (+0.0656) | 235.2s (+0.8%) | 21.806 (-1.782) | 0.7713 (-0.0527) | 0.2013 (+0.0625) | 235.2s (+0.7%) | 三项质量均显著退化 |
| `landmark_subsequence_mean`（raw-Q/K） | 22.433 (-0.451) | 0.7779 (-0.0136) | 0.1748 (+0.0130) | 353.7s (+51.6%) | 22.963 (-0.624) | 0.8101 (-0.0139) | 0.1530 (+0.0141) | 353.3s (+51.3%) | 三项质量均退化且显著更慢；不推进 |
| `landmark_subsequence_taylor01` | 22.309 (-0.574) | 0.7735 (-0.0180) | 0.1833 (+0.0216) | 390.4s (+67.3%) | 22.773 (-0.814) | 0.8021 (-0.0219) | 0.1620 (+0.0231) | 390.3s (+67.1%) | 9h 补全候选；全面退化且成本最高 |

raw-Q/K mean 行使用 roll13 自身的 paired baseline：all 为
`22.883437/0.791481/0.161784/233.404s`，confirmation 为
`23.587556/0.823997/0.138860/233.491s`。因此时间差没有混用 primary 的基线计时；mean 在
confirmation 的 PSNR、SSIM、LPIPS 三项均显著差于 paired baseline。

完整视频的 M2×distance 交互很明确：双方 M2 开启时，cosine 相对 Euclidean 的 all PSNR 优势为
`+1.571 dB`；关闭 M2 后仅为 `+0.049 dB`。difference-in-differences 为
`+1.521 dB [1.122,1.921]`。三个非 baseline 组合的 PSNR/SSIM/LPIPS paired 95% CI 均显示显著退化。

roll15 的四行也使用上述 roll13 同源码 paired baseline。三个非 baseline M2×distance 方案在 all
split 的 PSNR/SSIM/LPIPS 胜场分别为 `0/0/0`、`0/0/0`、`1/1/0`（共 25），三项 paired 95% CI
全部显著退化。`k_reuses_q_layout` 的 all 差值和 95% CI 为 PSNR
`+0.323 [-0.105,+0.752]`、SSIM `+0.0035 [-0.0101,+0.0171]`、LPIPS
`-0.0047 [-0.0149,+0.0055]`；质量 CI 跨 0，但耗时降低
`8.838s [7.895,9.781]`，25/25 prompts 均更快。

roll16 的两行同样使用 roll13 paired baseline。`update_weighted_medoid` 的 all PSNR/SSIM/LPIPS
差值及 95% CI 为 `-0.726 [-1.029,-0.424]`、`-0.0256 [-0.0336,-0.0177]`、
`+0.0295 [+0.0214,+0.0376]`，耗时增加 39.1%，25/25 均更慢。`updates0` 对应为
`-0.446 [-0.722,-0.169]`、`-0.0165 [-0.0253,-0.0076]`、
`+0.0187 [+0.0110,+0.0263]`，耗时增加 35.3%，同样 25/25 更慢。两项均被完整视频明确否定。

Taylor 行使用 roll13 中同源码版本、同 prompt/seed 的 baseline；其 all 绝对值为
`22.309063/0.773466/0.183347/390.431s`，相对 baseline 为
`-0.574374/-0.018016/+0.021563/+67.28%`；confirmation 为
`22.773309/0.802116/0.161982/390.260s`，相对 baseline 为
`-0.814248/-0.021881/+0.023121/+67.14%`。其完整视频结果在三项质量和成本上全面退化。
paired t 审计中，all 的 PSNR/SSIM/LPIPS 差值及
95% CI 分别为 `-0.574374 [-0.944245,-0.204503]`、
`-0.018016 [-0.027635,-0.008396]`、`+0.021563 [+0.013142,+0.029984]`；confirmation 三项也均
显著退化。去噪时间增加 `157.027s [156.300,157.754]`，25/25 prompts 均更慢。

逐 prompt 结果支持相同排序：`landmarks128` 的 PSNR/SSIM/LPIPS 胜场为 21/19/22（共25），
`landmarks64` 为 16/15/17。`q_reuses_k_layout` 为 13/14/17，且 25/25 prompts 的去噪时间低于
baseline；但其 confirmation 的 SSIM 和 LPIPS 均值略差，不能称为确认集全面获胜。
`k_reuses_q_layout` 为 13/13/15，25/25 更快；其 confirmation 三项均值均改善，但质量置信区间仍
跨 0。

Dense 的平均去噪时间为 325.99s；`updates4`、strict raw-mean 和 Taylor 已慢于 Dense，其余表中正式
sparse 候选的均值仍快于 Dense。表中时间差专门相对各自 paired sparse baseline 报告，以衡量
reblock 消融本身的成本变化。

### 4.3 三个优先候选的 VBench 复验

2026-10-03 对 Dense 和三个优先候选补充了 5s768p、同一 25-prompts 子集的 VBench Core Five
评价。每个视频只计算该 prompt 在 `samples.json` 中声明的维度，不把所有五个维度强加到每个
prompt。每个方法实际共有 41 个 dimension-video job：主体 7、背景 9、运动 7、成像 9、美学 9；
四个方法共 164/164 job 完成、0 失败。按用户指定，本表不包含 sparse baseline，也不存在用缺失的
5s Sol 结果凑成六组比较的情况。

| 方法 | 主体一致性 ↑ | 背景一致性 ↑ | 运动平滑度 ↑ | 成像质量 ↑ | 美学质量 ↑ |
|---|---:|---:|---:|---:|---:|
| Dense | 92.0038 | 95.9887 | 98.6234 | 70.8600 | 66.5978 |
| `landmarks64` | 91.7785 | 95.5352 | 98.5960 | 70.0222 | 66.8072 |
| `q_reuses_k_layout` | 91.9531 | 95.7813 | 98.6492 | 70.0579 | 66.9204 |
| `k_reuses_q_layout` | 91.7943 | 96.0201 | 98.6285 | 70.6427 | 66.2518 |

这些结果是小子集均值：Q 复用 K 的运动和美学点估计最高，K 复用 Q 的背景点估计最高且成像质量
最接近 Dense；不能仅凭这些小幅均值差异宣称统计显著或总体优于 Dense。VBench 补充了与 Dense
的绝对感知维度，但不替代 4.2 节逐 prompt、匹配 Dense 的 PSNR/SSIM/LPIPS。

### 4.4 六方法 5s768p/50-prompts 扩展复验

官方 20% 子集的 50 prompts 包含上述 25 prompts。前 25 个样本按 `sample_id + prompt SHA + seed +
config + latent SHA` 复用，新增 25 个样本重新生成；六方法均为 50/50，共 300/300 个生成结果。
配对质量共 250/250 条，无损 FFV1 视频 300/300 个，VBench 为 498/498 个 prompt 指定维度作业。
本节的 confirmation 专指不在原 25-prompts 子集中的新增 25 个 prompts。

质量表中的差值均相对同批 `baseline`；时间是 50 prompts 的充分预热去噪均值。

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Denoise ↓ | 相对 baseline |
|---|---:|---:|---:|---:|---|
| `sol` | 20.307 | 0.7163 | 0.2268 | 250.6s | `-2.544 dB/-0.0693/+0.0644/+7.9%` |
| `baseline` | 22.851 | 0.7856 | 0.1624 | 232.4s | reference |
| `landmarks64` | 22.908 | 0.7882 | 0.1590 | 241.0s | `+0.057 dB/+0.0026/-0.0034/+3.7%` |
| `q_reuses_k_layout` | 23.073 | 0.7859 | 0.1614 | 225.7s | `+0.222 dB/+0.0002/-0.0010/-2.9%` |
| `k_reuses_q_layout` | 22.967 | 0.7863 | 0.1616 | 224.5s | `+0.116 dB/+0.0007/-0.0008/-3.4%` |

50-prompts paired 95% CI 显示三候选的全体质量差均仍跨 0。新增 25 prompts 上，`landmarks64` 为
`-0.075 dB/+0.0004/-0.0002`；Q 复用 K 为 `+0.182 dB/-0.0007/+0.0023`；K 复用 Q 为
`-0.091 dB/-0.0021/+0.0030`。因此扩展集支持两个复用方向的稳定加速，但不支持宣称它们在 5s
新 confirmation 上全面提高质量；`landmarks64` 的 5s 收益也应表述为小幅、存在样本异质性。

| 方法 | 主体一致性 ↑ | 背景一致性 ↑ | 运动平滑度 ↑ | 成像质量 ↑ | 美学质量 ↑ |
|---|---:|---:|---:|---:|---:|
| Dense | 93.9798 | 95.0741 | 98.9201 | 72.6943 | 68.8192 |
| Sol | 94.0868 | 95.3404 | 98.9001 | 72.6411 | 68.5371 |
| `baseline` | 93.9688 | 94.8885 | 98.9033 | 71.6542 | 69.0508 |
| `landmarks64` | 93.8714 | 94.6230 | 98.8985 | 72.0861 | 68.8866 |
| `q_reuses_k_layout` | 93.8863 | 95.0568 | 98.9206 | 71.9432 | 68.9661 |
| `k_reuses_q_layout` | 93.9398 | 94.9235 | 98.9189 | 72.4462 | 68.7788 |

各方法的维度样本数为主体 14、背景 17、运动 14、成像 19、美学 19。VBench 各维度排序并不与
Dense-paired PSNR/SSIM/LPIPS 完全一致，这也是保留两类评价、而不以单一代理指标排序的原因。

### 4.5 六方法 10s768p/25-prompts 迁移复验

10s 使用相同的原始 25 prompts、seed 42、20 steps 和 768×1344 分辨率，仅将输出长度改为 240 帧。
六方法生成 150/150，配对质量 125/125，无损视频 150/150，VBench 246/246。推理采用严格
arm-major 调度：任一时刻 8 卡只运行同一 arm，统一 warmup、动态领取 prompts，并在 arm 间设置
硬屏障，以避免 arm 与 GPU 速度差异耦合。

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Denoise ↓ | 相对 baseline |
|---|---:|---:|---:|---:|---|
| `sol` | 20.377 | 0.7201 | 0.2034 | 687.4s | `-3.311 dB/-0.0938/+0.0735/+14.6%` |
| `baseline` | 23.688 | 0.8139 | 0.1299 | 599.6s | reference |
| `landmarks64` | 24.175 | 0.8247 | 0.1187 | 612.8s | `+0.487 dB/+0.0108/-0.0111/+2.2%` |
| `q_reuses_k_layout` | 23.978 | 0.8198 | 0.1263 | 570.9s | `+0.290 dB/+0.0059/-0.0036/-4.8%` |
| `k_reuses_q_layout` | 23.343 | 0.8023 | 0.1360 | 570.5s | `-0.345 dB/-0.0116/+0.0062/-4.9%` |

`landmarks64` 的 all paired 95% CI 为 PSNR `[+0.187,+0.786]`、SSIM
`[+0.0027,+0.0189]`、LPIPS `[-0.0164,-0.0059]`，三项均支持改善；15-prompt confirmation
也三项改善。Q 复用 K 的 all PSNR/SSIM/LPIPS CI 仍部分跨 0，但 25/25 prompts 均更快，且
confirmation 的 PSNR 与 LPIPS CI 支持改善。K 复用 Q 的 all SSIM 差为
`-0.0116 [-0.0220,-0.0011]`，显示明确退化；其余两项点估计也朝不利方向。因此布局复用具有方向性，
5s 的小幅均值不能代替 10s 迁移验证。

| 方法 | 主体一致性 ↑ | 背景一致性 ↑ | 运动平滑度 ↑ | 成像质量 ↑ | 美学质量 ↑ |
|---|---:|---:|---:|---:|---:|
| Dense | 86.0691 | 94.4445 | 99.0195 | 70.1396 | 66.2261 |
| Sol | 86.8619 | 94.4565 | 98.9482 | 70.5106 | 66.0340 |
| `baseline` | 85.9415 | 94.4659 | 98.9872 | 69.8281 | 66.8278 |
| `landmarks64` | 86.2832 | 94.2001 | 98.9975 | 69.7083 | 66.3863 |
| `q_reuses_k_layout` | 86.0468 | 94.3760 | 99.0003 | 69.9130 | 66.9875 |
| `k_reuses_q_layout` | 86.4706 | 94.3575 | 98.9967 | 70.0266 | 66.4603 |

每个方法的维度样本数为主体 7、背景 9、运动 7、成像 9、美学 9。这里仍是官方 prompt 归属下的
分维度均值，不是完整 VBench 总分；维度样本较少，不能把小幅差异表述为统计显著。

## 5. 质量—成本判断与下一步

1. `landmarks64` 是当前推荐候选：5s/50 prompts 全体收益较小，但 10s/25 prompts 的 all 与
   confirmation 三项质量均改善，且 10s 额外耗时仅 2.2%。
2. `landmarks128` 适合作为高质量档：质量提升最大，但需接受约 17% 的 sparse-baseline 成本。
3. `q_reuses_k_layout` 是当前布局复用首选：5s/50 prompts 快 2.9%，10s/25 prompts 快 4.8%，
   10s 三项质量点估计均改善。`k_reuses_q_layout` 虽同样稳定加速，但 10s SSIM 显著退化，优先级
   下调为速度导向备选，不能作为质量保持型默认方案。
4. M2×distance 完整视频 2×2 明确支持 `both M2 + cosine`；不推进另外三个组合。
5. 不推进 endpoint seed、medoid、0/1/4 更新、block mean、mean+diag、full M2、strict raw-mean 或
   Taylor 0+1。
6. landmark 压缩轴的 raw-Q/K mean 与 Taylor 已分别由 roll13/roll14 的 25-prompts 完整视频判定退化。
7. 下一步只优先测试 `landmarks64 + q_reuses_k_layout` 的少量组合；本报告不从单轴结果推断组合收益。

## 6. 限制

- 10s768p 已在原 25 prompts 上验证，但尚未扩展到 50 prompts；不能把 10s 结果外推为更大数据集结论。
- 主消融仍以匹配 Dense 的 PSNR/SSIM/LPIPS 为主；扩展复验的 VBench 覆盖六方法，但只按各 prompt
  的官方维度归属评分，不是完整 VBench 榜单或人类偏好评价。
- full-M2 完整生成只归档了配置、最终 attention summary 和总去噪时间，没有归档逐次 M2 构造的
  分块、输入/乘法/累加精度和单独耗时。因此可比较端到端质量和总成本，但不能从该运行拆解
  full-M2 内部构造成本；数值正确性依据专项单元测试。
- raw-Q/K mean 与 Taylor 的完整视频运行、跨 root 同源码配对、哈希、公式和 CI 审计均已完成。
  完整生成 records 没有保存逐层 Taylor calls；“所有层生效”的直接证据来自冻结源码和专项多层测试。
- LPIPS 使用 AlexNet v0.1。初始 23-arm 主消融以内存解码记录为主；5s/50p 与 10s/25p 复验另保存了
  逐像素校验的 FFV1 视频、latent/video hashes、decoder 和 metric fingerprint。

## 7. 复现入口与原始记录

### 7.1 脚本

- `scripts/reblock_ablation_5s768p.py`：完整生成、GPU 分片、协议和汇总；
- `scripts/reblock_ablation_quality_5s768p.py`：paired latent decode、PSNR/SSIM/LPIPS；
- `scripts/reblock_priority_5s768p_vbench.py`：优先候选的 lossless FFV1 视频归档与哈希验证；
- `scripts/reblock_priority_5s_vbench_score.py`：四方法、prompt 指定维度的 VBench 队列和汇总；
- `scripts/reblock_priority_5s768p_50p.py`：六方法 5s768p/50-prompts 扩展生成；
- `scripts/reblock_priority_10s768p.py`：六方法 10s768p、严格全卡同-arm动态调度；
- `scripts/reblock_priority_10s768p_quality.py`：10s paired decode、质量指标和 FFV1 归档；
- `scripts/smoke_reblock_arms_5s_shape.py`：arm 配置和 5s grid 冒烟验证；
- `tests/test_reblock_m2_estimators.py`、`tests/test_reblock_proxy_controls.py`：公式、边界和控制路径测试。

典型命令：

```bash
# 完整生成；默认运行冻结的完整 arm matrix，dense 自动加入
python scripts/reblock_ablation_5s768p.py prepare
python scripts/reblock_ablation_5s768p.py run --arms baseline landmarks64 landmarks128
python scripts/reblock_ablation_5s768p.py summarize --arms baseline landmarks64 landmarks128

# 完整视频质量
python scripts/reblock_ablation_quality_5s768p.py prepare \
  --experiment /mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_20261002 \
  --arms baseline landmarks64 landmarks128
python scripts/reblock_ablation_quality_5s768p.py run \
  --experiment /mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_20261002 \
  --arms baseline landmarks64 landmarks128
python scripts/reblock_ablation_quality_5s768p.py summarize \
  --experiment /mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_20261002 \
  --arms baseline landmarks64 landmarks128

```

### 7.2 结果目录

```text
/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_20261002/
  protocol.json
  results.json                       # dense + baseline + L64/L128/midpoint1
  records/
  latents/
  quality_latent_decode/results.json

/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_roll01_*_20261002/
...
/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_roll13_landmark_subsequence_mean_raw_strict_20261002/
  results.json                       # 25/25 generation records per arm
  quality_latent_decode/results.json # 50/50 paired quality records

/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_roll14_landmark_subsequence_taylor01_20261002/
  results.json                       # 25/25 Taylor generation records
  quality_latent_decode/results.json # 25/25 Taylor quality records；baseline 复用 roll13

/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_roll15_priority4_20261003/
  results.json                       # 四个优先级候选，各 25/25 generation records
  quality_latent_decode/results.json # 100/100 quality records；baseline 复用 roll13

/mnt/CFS/tangzecheng/experiments/reblock_ablation_5s768p_25p_roll16_remaining2_20261003/
  results.json                       # weighted-medoid/updates0，各 25/25 generation records
  quality_latent_decode/results.json # 50/50 quality records；baseline 复用 roll13

/mnt/CFS/tangzecheng/experiments/reblock_priority_5s768p_vbench_25p_20261003/
  quality_videos/                    # 5 方法各 25 个逐像素校验的 FFV1 归档；baseline 未进入正式评分
  vbench/protocol.json               # prompt—维度归属和输入哈希
  vbench/queue.sqlite                # 164/164 job 的逐任务状态与原始分数
  vbench/results.json                # Dense + 三个优先候选汇总
  VBENCH_REPORT.md

/mnt/CFS/tangzecheng/experiments/reblock_priority_5s768p_50p_20261003/
  results.json                       # 六方法，各 50/50 generation records
  quality_latent_decode/results.json # 250/250 candidate-vs-Dense quality records
  quality_videos/                    # 六方法×50 个 FFV1 归档
  vbench/queue.sqlite                # 498/498 prompt-assigned jobs
  vbench/results.json

/mnt/CFS/tangzecheng/experiments/reblock_priority_10s768p_25p_20261003/
  results.json                       # 六方法，各 25/25 generation records
  quality_latent_decode/results.json # 125/125 candidate-vs-Dense quality records
  quality_videos/                    # 六方法×25 个 FFV1 归档
  vbench/queue.sqlite                # 246/246 prompt-assigned jobs
  vbench/results.json
  evaluation_status.json             # complete
```

各 roll 的确切目录名、配置、latent SHA、prompt SHA、GPU UUID 和逐样本失败状态均保存在对应的
`protocol.json`、`results.json`、`records/` 与 `quality_latent_decode/results.json` 中。复现或追加实验时
应创建独立输出目录，不覆盖上述原始记录。
