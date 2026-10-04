# Threshold-compatible external route 可行性实验（2026-10-04）

## 结论

在不修改生产源码的前提下，备用原型证明了：

- **可以明显缩小视频轨迹差距，同时没有观察到速度损失。** 两个完整推理样本的视频
  latent 相对 threshold 的 L2 距离分别缩小 33.78%（5s）和 17.63%（10s）。
- **不能用当前外部 GEMM 免费实现 threshold 的逐位输出一致。** 外部构造的 mask 已与
  外部 radix score 的严格 `score > cutoff` 逐位一致，但 threshold 主核内部重新计算
  route-QK，其逐 token MMA/归约算术和“先求 BF16 query centroid 再 GEMM”的外部路径
  不逐位等价。因此单层输出仍有残余差异。
- 若要求完全消除差异，必须在外部复刻主核 route-QK 算术，或保留主核内部 route-QK；
  前者需要新 kernel 和性能验证，后者直接放弃 `no_route_qk` 的主要收益。当前证据不支持
  “完全消除且保证零速度代价”。
- 当前方案是一个有希望的折中：使用 legacy directions 消除 fused 轴差异，再使用
  strict-radix packed mask 缩小 external 轴差异，并继续调用
  `packed_external_no_route_qk`。

所有实现都位于独立备用脚本中；`h3_sparse_attention/` 下生产文件和默认值未修改。

## 备用实现

脚本：`scripts/prototype_threshold_compatible_external_20261004.py`

原型流程：

1. 复用 `_gemm_score_map_prefix` 构造当前外部 BF16 GEMM scores；
2. 复用 `_radix_cutoff_from_scores` 得到相同 FP32 cutoff；
3. 由本脚本内独立 Triton kernel 按严格 `score > cutoff` 直接打包 int32 route；
4. 在当前进程中临时替换 `gemm_topk_packed_route`；
5. 继续使用未修改的 `packed_external_no_route_qk` 执行内核。

原型不写入生产模块。进程结束后 monkey patch 自动消失。

首版原型错误地假定 `einsum` score 连续布局，产生错误 mask；失败结果保留在
`threshold_compatible_external_20261004` 供审计。v2 改为显式使用 batch/query/head/key
stride，之后所有 mask 检查通过；没有覆盖或删除失败记录。

## 固定 capture 结果

有效结果在 GPU 0 和 GPU 1 各独立复跑一次。两卡输出指标完全一致。

### Mask

| Capture | threshold tie rows | 原 topk 与 strict-threshold 不同 bits | 原型与 strict-threshold 不同 bits |
|---|---:|---:|---:|
| 5s case01, 8 heads | 2,401 | 4,805 | 0 |
| 10s case05, 3 heads | 2,138 | 4,595 | 0 |

原型每行 exact 数量也与外部 threshold mask 一致：5s 为 44–58，10s 为 102–113。

### 单层输出距离

| Capture | 原 external vs threshold rel-L2 | 原型 vs threshold rel-L2 | 缩小比例 |
|---|---:|---:|---:|
| 5s case01 | 0.01733595 | 0.01594051 | 8.05% |
| 10s case05 | 0.00400114 | 0.00381535 | 4.64% |

mask 对外部 score 已逐位一致，但输出没有逐位一致。这一结果定位出 residual：生产
threshold 主核按 64 个 BF16 Q token 分别与 key centroid 做 MMA，再对 route fragments
归约；外部路径先把 Q block FP32 求均值、舍入到 BF16，再做 GEMM。两者在实数代数上
等价，在实际舍入/归约上不等价。

### 单层 replay 速度

CUDA event 中位数；包含 permutation、virtual summaries、sparse attention、dense suffix
和 inverse permutation，不包含模型投影或完整 denoise。

| GPU / Capture | threshold | 兼容 external | 兼容方案变化 |
|---|---:|---:|---:|
| GPU 0 / 5s | 5.8074 ms | 5.6664 ms | -2.43% |
| GPU 1 / 5s | 5.8007 ms | 5.6695 ms | -2.26% |
| GPU 0 / 10s | 8.0450 ms | 7.8988 ms | -1.82% |
| GPU 1 / 10s | 8.0279 ms | 7.8888 ms | -1.73% |

route builder 本身在 5s 略快约 5–6%；10s 慢约 22%，但绝对差只有约 0.049 ms，且被
少选 exact blocks 后的主核时间抵消。两卡均未观察到完整单层 replay 变慢。

## 同会话完整推理探针

备用 runner：`scripts/run_threshold_compatible_external_probe_20261004.py`。

固定 legacy midpoint directions，三臂为：

- `legacy_threshold`
- 当前 `legacy_packed_external_no_route_qk`
- `legacy_compatible_external`（本原型 mask + 未修改 no-route-QK 内核）

每臂先做独立 3-step warmup，再在相同模型进程、seed 42、20 requested steps、19
transformer evaluations、4 次 schedule-dense evaluation、1 层始终 dense 下计时。
5s/10s 分别在 GPU 0/1 并行运行。只保存紧凑指标，不保存或覆盖历史 latent/video。

### 5s case08

| 方法 | denoise time | video latent rel-L2 vs threshold | audio latent rel-L2 vs threshold |
|---|---:|---:|---:|
| threshold | 140.470 s | 0 | 0 |
| 原 external | 140.209 s | 0.216884 | 0.053865 |
| 兼容 external | 140.145 s | 0.143621 | 0.079407 |

视频 latent 距离缩小 **33.78%**。兼容方案比 threshold 快 0.325 s（0.23%），比原
external 快 0.063 s；单样本计时只作描述性结果。audio latent 距离增加，不能声称多模态
轨迹均单调改善。

### 10s case12

| 方法 | denoise time | video latent rel-L2 vs threshold | audio latent rel-L2 vs threshold |
|---|---:|---:|---:|
| threshold | 338.928 s | 0 | 0 |
| 原 external | 338.363 s | 0.111241 | 0.045229 |
| 兼容 external | 336.577 s | 0.091626 | 0.057382 |

视频 latent 距离缩小 **17.63%**。兼容方案比 threshold 快 2.351 s（0.69%），比原
external 快 1.786 s；仍是单样本、固定运行顺序的描述性结果，不能外推为总体加速。
audio latent 同样没有改善。

## 判断与下一步

### 已证明可行

- legacy direction 可去除 fused 方向构造这一轴的差异，历史实验没有测得端到端速度损失。
- strict-radix packed route 可在两个真实 capture、两张 GPU 上稳定重现。
- 它在两个完整推理样本中明显缩小视频 latent 差距，没有观察到速度回退。
- no-route-QK 执行内核保持未修改。

### 尚未证明

- 尚未证明 PSNR/SSIM/LPIPS 或语义质量必然改善；本探针没有 decode/质量打分。
- 两个 prompt 不能估计总体均值或置信区间。
- audio latent 在两个样本上都离 threshold 更远，说明不能用视频结论替代多模态结论。
- 尚未实现 threshold 主核实际 route trace 的外部逐位复刻，因此不能宣称输出等价。

建议把当前原型作为候选优化，而不是直接替换生产路径。下一阶段若继续，应先做固定
5s/10s 的 5–10 prompt paired 筛选，联合比较 video/audio latent、PSNR/SSIM/LPIPS 和
同会话时间；只有趋势稳定后才值得做更大的 50-prompt 验证。若目标改为严格 bitwise
parity，则应单独开发复刻 SM120 主核 route-MMA 归约的实验 kernel，并接受可能失去
no-route-QK 加速的风险。

## 产物

- 有效 GPU 0 fixed-capture：
  `/autodl-fs/data/h3_experiments/threshold_compatible_external_v2_20261004/results.json`
- 有效 GPU 1 fixed-capture：
  `/autodl-fs/data/h3_experiments/threshold_compatible_external_v3_gpu1_20261004/results.json`
- 完整推理探针：
  `/autodl-fs/data/h3_experiments/threshold_compatible_external_full_probe_20261004/results.json`
- 首版失败审计：
  `/autodl-fs/data/h3_experiments/threshold_compatible_external_20261004/results.json`

