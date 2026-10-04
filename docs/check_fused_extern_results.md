# Fused × external 精度差异归因结果（2026-10-04）

## 结论

结论类别：**未发现 bug，但存在符合当前代码契约的数值与路由差异**。

- `legacy` 与 `fused` 使用相同 midpoint token 和相同整数容量。首个差异发生在
  midpoint proxy direction 的 FP32 算术/归约次序：10s 真实 capture 的 3 次方向构造
  有 112,517 个 FP32 元素不逐位相同，但最大绝对误差仅 `1.1920929e-7`。该微小误差
  在部分 near-tie 节点跨过排序边界，因而可离散放大为不同的 Q/K permutation；没有
  发现索引越界、非双射、容量不守恒、K/V 不同步或 Q 逆置换错误。
- `threshold` 和 external 使用相同的 BF16 GEMM route score。没有 Top-K 边界 tie 时，
  两者 exact mask 和注意力输出均逐位相同；存在 BF16 边界 tie 时，两者按设计允许不同：
  threshold 主核使用严格 `score > cutoff`，会排除整个边界并列组，external 的
  `torch.topk` 则从并列组中选够固定预算后打包位图。
- 固定同一 packed mask 后，`packed_external` 与
  `packed_external_no_route_qk` 在 5s/10s、legacy/fused permutation 四组真实 replay
  中全部逐位相同；独立同掩码夹具也逐位相同。因此没有发现 packed bit 解码、
  route-QK 跳过、K/V 或输出执行逻辑错误。
- 原 50-prompt 结果的符号变化可以由“微小方向误差偶尔改变离散分组 + BF16 边界 tie
  使 threshold/external 选择不同 exact 块 + denoise 轨迹反馈”解释，但目前证据不足以
  预测单个 prompt 或 5s/10s 均值的符号。非显著结果不能写成等价。

未修改生产代码或默认值，也没有把新结果混入原 full50 统计。

## 数据、环境与源码

运行设备为两张 NVIDIA RTX PRO 6000 Blackwell Server Edition（SM120）；主诊断使用
GPU 0，同掩码交叉夹具使用 GPU 1。运行环境为 Python 3.12.3、PyTorch
`2.12.1+cu130`、CUDA 13.0、Triton 3.7.1。Git HEAD 为
`82f56f04edba7338fc75c5c2791b09ffe5d0a204`。执行前后 GPU 0–1 均无其他计算进程；
`three_task_queue_4gpu_20261004/status.json` 所列 PID 已不存在，是重启前的陈旧状态。

真实固定输入：

| Capture | 内容 | SHA256 |
|---|---|---|
| 5s case01, eval04, layer01 | 8 heads；37,296 video tokens + 593 context K/V | `d80975043bb17665a09edc63c2d8cbb660d3ac84c0651263d136c99246720a68` |
| 10s case05, eval04, layer01 | heads 13/51/54；72,576 video + 1,008 context tokens | `75197d1138abb90732ddd24a1ff0a2f1b81fd21e0a8beaf2a13260688981f84c` |

5s capture 没有保存 context Q，因此 replay 对 context Q 使用零张量；所有 route、
permutation 和 sparse video-prefix 结论仍使用真实 Q/K/V，dense context-query 输出不作为
原生成复现。10s capture 包含完整 Q/K/V。两者均只读复用，没有改写 capture。

本次关键源码 SHA256：

| 文件 | SHA256 |
|---|---|
| `landmark_tree_v2.py` | `db4e70ca1b1ab5149711a28a5394a0798d66dfabd5a3ff30914be5a097585de6` |
| `landmark_v2_fused_node.py` | `1fae0bfdde0a9dcec5f021d737c7687f2c0cfaf09b1c6ca6f444037b7d666d81` |
| `spark_integration.py` | `b03cee4628a028a98672ed4f947442d7026783e07e254214541aefc41af41940` |
| `sol_topk_cutoff.py` | `23a52a0907cb89b4b6c05661c99ed9c094945e27e3daf72d90fbe1ae15a5a663` |
| `sol_numerator_virtual_q.py` | `704f8004402649f4851e0678de88ca0bbe9179a5c55905206a53d883431e01ca` |
| `spark_reweight_sm120.py` | `988838851aeaaf75c7a67c2fbc0ec94eb36dc96356b723b812cadba64a0eec7c` |

两个 full50 `protocol.json` 记录的 cutoff、numerator、SM120 mainloop、processor 和 runner
哈希与本次读取值一致。cases 1–25 来自更早的 route-mode 运行，其 protocol 记录了 runner
哈希 `3fd906...` 和环境但没有逐核心文件哈希；因此不能把这部分描述为已证明的全源码
逐位相同快照。cases 26–50 的 runner 哈希为 `77e548...`。这一 provenance 限制不影响
固定 capture 的当前源码 replay，但仍是解释 full50 绝对均值时的历史基线 caveat。

## 四臂配置核对

四臂只改变下表两轴；共同设置为 BF16 模型与 anchor、Top-K 10%、20 requested steps、
19 transformer evaluations、4 次 schedule-dense evaluation、1 层始终 dense、seed 42、
768p、相同 conditioning 与逐 prompt Dense 配对。

| 配置 | midpoint direction | route selection / execution |
|---|---|---|
| `legacy_threshold` | legacy | FP32 radix cutoff；主核严格 `>` |
| `fused_threshold` | fused | FP32 radix cutoff；主核严格 `>` |
| `legacy_packed_external_no_route_qk` | legacy | exact `topk` packed mask；跳过重复 route-QK |
| `fused_packed_external_no_route_qk` | fused | exact `topk` packed mask；跳过重复 route-QK |

代码路径契约：

1. legacy 先用 `indexed_interval_means(..., midpoint=True)` 产生 BF16 midpoint samples，
   再由 `build_cosine_directions` 构造 proxy tree；fused 在一个 CTA 中读取相同 BF16
   midpoint samples、转 FP32 并完成 proxy tree。之后的节点容量、partition、Q/K
   permutation、K/V 成对重排和 Q inverse 是共同路径。
2. 两条 route 路径共用 BF16 query/key centroid GEMM score。threshold 将 BF16 GEMM
   结果提升到 FP32 做 radix cutoff；主核以严格 `>` 重判。external 对同一 score 直接
   `topk` 并打包 int32 mask。因此无 tie 时要求选择与输出逐位相同；边界 tie 时本来就
   不要求选相同块。
3. 原四臂中 local block 保护关闭（landmark preprocess 时 auto=False）。sink/context
   不写入 external mask，而由主核按 `[video_tokens//64, ceil(total_tokens/64))` 强制
   exact。非 64 整除的 video 尾部不进入 candidate complete blocks，query-tail 走 dense。

## 固定输入：方向与 permutation

历史 10s stage capture 已保存同一输入的 legacy/fused 中间量：

| 阶段 | 不同元素 | 最大绝对差 | 相对 L2 | 解释 |
|---|---:|---:|---:|---|
| proxy directions（3 calls） | 112,517 | 1.19e-7 | — | 首个差异；FP32 归约/中间算术 |
| Q permutation | 1,711 / 217,728 (0.786%) | 索引差 48,094 | 0.00689 | near-tie 离散放大 |
| K permutation | 1,122 / 217,728 (0.515%) | 索引差 7,935 | 0.00200 | near-tie 离散放大 |
| global anchor | 0 | 0 | 0 | 本例 active root anchor 不受影响 |
| threshold | 172 / 3,450 | 0.8289 | 0.00213 | permutation 改变 block centroid |
| sparse output | 743,934 | 3.125 | 0.00997 | 单层传播 |
| inverse 后 final output | 589,645 | 2.15625 | 0.00220 | 单层传播 |

5s case01 的 9 个真实 capture（eval 04/11/18 × layer 01/25/49）显示差异不是随 step、
layer 或 token 数单调增加：8 个点 Q/K permutation 完全相同；仅 eval04/layer49 的 K
permutation 有 313/298,368（0.105%）个索引不同，Q 仍完全相同。所有 permutation 均为
双射，inverse 恢复原顺序；K/V 使用同一个 paired permutation。既有 route/tree 测试还
覆盖了整数容量和尾部不变量。

## 固定输入：threshold 与 external mask

| 输入 / 固定 permutation | 边界 tie rows | threshold 每行块数 | external 每行块数 | 不同 bits | mean Jaccard |
|---|---:|---:|---:|---:|---:|
| 合成正常尾部（video 1025、context 67） | 0/32 | 2 | 2 | 0 | 1.0000 |
| 合成全 tie 尾部 | 32/32 | 0 | 2 | 64 | 0.0000 |
| 5s real / legacy | 2,393/4,656 | 44–58 | 58 | 4,805 | 0.98221 |
| 5s real / fused | 2,393/4,656 | 44–58 | 58 | 4,805 | 0.98221 |
| 10s real / legacy | 2,138/3,402 | 102–113 | 113 | 4,595 | 0.98805 |
| 10s real / fused | 2,140/3,402 | 102–113 | 113 | 4,645 | 0.98792 |

正常夹具同时覆盖非 64 整除的视频和 context，预算、候选范围、sink 范围和有限输出均
通过。真实 capture 的 mask 差异全部出现在 `selected_min == excluded_max` 的 BF16
边界 tie rows：threshold 的严格 `>` 排除边界 tie，external 则补足 58/113 的固定预算。
这解释了为什么两者高 Jaccard 但不逐位相同；不能把随后的输出差异算作内核错误。

## 固定 mask：执行内核隔离

| 对照 | 5s legacy | 5s fused | 10s legacy | 10s fused |
|---|---:|---:|---:|---:|
| packed external vs no-route-QK | bitwise equal | bitwise equal | bitwise equal | bitwise equal |

另用 `[1,8256,1,128]` BF16 结构化输入构造无边界 tie 的同一 32/128 exact mask：

- threshold mask 与 packed mask：0 个不同 bit；
- threshold vs `packed_external` 输出：逐位相同，max abs=0；
- `packed_external` vs `packed_external_no_route_qk` 输出：逐位相同，max abs=0；
- 三个输出均 finite。

真实 capture 中 threshold 与 external 输出的相对 L2 分别约为 0.01734（5s fixture）和
0.00400（10s）；由于其 mask 已证明不同，这些数字只表示 route-selection 影响，不能
解释为执行内核误差。生产尺寸 Dense attention 无法在不物化不可接受的全矩阵情况下
作为这一内核隔离的参考；小夹具中同 mask 三路径逐位一致，真实轨迹仍以原 matched
Dense 视频的 PSNR/SSIM/LPIPS 为端到端参照。

## full50 与定位样本

> **抽样口径提醒（2026-10-04）：**本节提到的“前 25 条/后 25 条”仅用于检查
> full-50 内部异质性；前 25 条是 `20pct` manifest 的 cases 1--25，不是官方
> `10pct` 子集，二者只重合 13 条。任何相对优劣、均值符号或候选排序都必须依据
> 完整 50 prompts；不能把分半结果单独当作可信 benchmark 结论。

相对 `legacy_threshold` 的 full50 均值复核如下：

| 配置 | 5s PSNR delta | 10s PSNR delta |
|---|---:|---:|
| `fused_threshold` | -0.1796 dB | -0.0322 dB |
| `legacy_packed_external_no_route_qk` | +0.0943 dB | -0.1175 dB |
| `fused_packed_external_no_route_qk` | -0.0987 dB | +0.0390 dB |

二阶交互仍为 5s -0.0134 dB（95% CI [-0.3999,+0.3731]）和 10s +0.1886 dB
（95% CI [-0.0755,+0.4528]），均不显著且未做多重比较校正。

按 `abs(fused_threshold - legacy_threshold)` 从 full50 预选的定位样本为：

| 时长 | case | 选择 | fused-threshold − legacy-threshold | legacy-external − legacy-threshold | fused-external − fused-threshold |
|---|---:|---|---:|---:|---:|
| 5s | 08 | 极端 | -3.3376 | -3.6697 | +0.5699 |
| 5s | 20 | 极端 | -2.1462 | +0.3966 | +3.6465 |
| 5s | 25 | 近零 | +0.0743 | +0.7273 | -1.4306 |
| 10s | 12 | 极端 | -3.1362 | -1.6095 | +0.0247 |
| 10s | 11 | 极端 | +1.3127 | +1.3344 | -0.6512 |
| 10s | 41 | 近零 | -0.0063 | -0.3014 | +0.0867 |

这些个例清楚显示方向轴和 external 轴都没有固定符号，不能用极端样本均值估计总体
效果。10s fused-threshold 前 25 条为 -0.1913 dB、后 25 条为 +0.1269 dB，也支持
“轨迹敏感、样本异质”而不是“长度决定固定方向”的解释。PSNR/SSIM/LPIPS 只表示相对
Dense 的像素/特征保真度，不自动代表语义质量。

## 测试、产物与复现

已有测试在 GPU 0 上以 production opt-in kernel parity 运行：

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 H3_RUN_KERNEL_PARITY=1 \
  pytest -q tests/test_packed_external_no_route_qk.py \
  tests/test_topk_host_synchronization.py \
  tests/test_landmark_v2_small_proxy.py tests/test_landmark_v2_route.py
```

结果为 `86 passed, 2 warnings`。首次未设置 `PYTHONPATH` 的尝试只在 import collection
阶段失败，未执行测试或 GPU kernel；以上命令是有效复跑。

新增可复现脚本：

- `scripts/diagnose_fused_external_20261004.py`
- `scripts/diagnose_fused_direction_trajectory_20261004.py`
- `scripts/verify_fused_external_same_mask_20261004.py`

独立实验目录：

`/autodl-fs/data/h3_experiments/check_fused_extern_20261004`

其中 `results.json` 保存环境、源码/capture 哈希、逐 stage/mask/output 指标；
`direction_trajectory_5s_case01.json` 保存 9 个 step/layer permutation 对照；
`same_mask_fixture.json` 保存 GPU 1 的同 mask 三路径逐位一致结果。没有保存重复的大张量。

最终判定不支持对生产实现做修复：保留 `legacy` 默认值是合理的复现性选择；
`packed_external_no_route_qk` 没有显示执行正确性问题。若未来要求 threshold 与 external
具有完全相同的固定预算语义，应将 tie policy 作为显式规格变更并单独评估质量，而不是
把当前差异当作 no-route-QK bug 修补。
