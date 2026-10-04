# Fused × external 精度差异归因实验：给执行 agent 的 Prompt

请在本仓库开展一次**诊断实验**，回答：5s/10s、768p、50 prompts 的四组结果是否暴露了 `fused` 中点方向构造或 `packed_external_no_route_qk` 路由执行的逻辑错误？请用固定输入下可复现的中间量和输出对照定位差异，不要仅凭最终视频 PSNR 的正负判断实现正确性。

## 已有证据与实验边界

先阅读 `docs/route_mode_quality_25prompt_20261003.md`、`docs/spark_midpoint_5prompt_speed_bitwise_20260927.md`、`scripts/run_route_mode_quality_25prompt_20261003.py`、`scripts/run_route_2x2_full50_3gpu_20261004.py`，以及以下两个结果文件的 `full50` 字段：

- `/autodl-fs/data/h3_experiments/route_2x2_full50_4gpu_20261004_5s768p/results.json`
- `/autodl-fs/data/h3_experiments/route_2x2_full50_4gpu_20261004_10s768p/results.json`

四个配置分别是 `legacy_threshold`、`fused_threshold`、`legacy_packed_external_no_route_qk`、`fused_packed_external_no_route_qk`。其中 `fused` 仅指 `landmark_tree_v2_midpoint_direction_mode="fused"`；`external` 特指 `sol_route_topk_execution="packed_external_no_route_qk"`，**不是** `sol_route_topk_execution="fused"`。固定其余 Spark-H3-10pct 设置：BF16 模型与 anchor、Top-K 10%、20 requested steps、19 transformer evaluations、4 次 schedule-dense evaluation、1 层始终 dense、seed 42、相同 conditioning、Dense 参考和逐 prompt 配对。核对历史记录和新运行的源码哈希，说明任何快照差异。不得覆盖原有实验文件。

已有完整样本的 PSNR 相对 `legacy_threshold` 的均值为：

> **抽样口径提醒（2026-10-04）：**下文“前 25 条/后 25 条”只是 `20pct`
> 完整 50 条的分半诊断，不是官方 `10pct` 25-prompt 子集。前 25 条与官方子集仅重合
> 13 条，不能单独用于判断相对优劣；可信的总体判断必须以这里的 full-50 结果为准。

| 配置 | 5s | 10s |
|---|---:|---:|
| `fused_threshold` | -0.1796 dB | -0.0322 dB |
| `legacy_packed_external_no_route_qk` | +0.0943 dB | -0.1175 dB |
| `fused_packed_external_no_route_qk` | -0.0987 dB | +0.0390 dB |

逐 prompt 的二阶交互定义为 `(fused_external - fused_threshold) - (legacy_external - legacy_threshold)`：5s 为 -0.0134 dB，95% CI [-0.3999, +0.3731]；10s 为 +0.1886 dB，95% CI [-0.0755, +0.4528]。两者均不显著。10s `fused_threshold` 相对 legacy 的前 25 条为 -0.1913 dB，后 25 条为 +0.1269 dB。历史五 prompt replay 已证明 fused 方向路径会失去 legacy 的逐位一致性，且恢复历史方向构造可恢复那五条 latent 的逐位一致性；复用这项发现，进一步查明差异具体来自哪里以及 external 如何改变其影响。将上述数字作为待解释的观察，不作为 bug 或协同收益的先验结论；原实验的多次探索性比较未做多重检验校正。PSNR/SSIM/LPIPS 是对 Dense 的像素或特征保真度，不能自动代表语义质量。

## 目标与必须回答的问题

1. 在**同一份输入**上，legacy 与 fused 中点方向从哪一步开始不同？比较 midpoint landmarks、数据类型及舍入、proxy seed/方向、节点分数、每层容量约束划分、最终 Q/K permutation。区分预期的浮点舍入差异与索引、容量、尾部、Q/K/V 同步等不变量错误。
2. 在**同一份重排后 Q/K/V 与 key centroids**上，threshold 与 external 选择的 exact 块是否相同？比较每个 query-block/head 的预算、掩码交并比、差异块、Top-K 边界分数和 ties，并确认 local/sink/context 保护与尾块范围。不要预设两条路径应逐位相同：`gemm_topk_packed_route` 先生成掩码，threshold 路径在主内核中按阈值重新判定。
3. 固定同一份 exact 掩码后，`packed_external` 与 `packed_external_no_route_qk` 的注意力输出是否严格一致？再比较 threshold 与 external 在相同掩码下的输出，以隔离路由选择和执行内核。如果无法向某路径注入相同掩码，建立最小等价测试夹具并明确其与生产路径的关系；不能把掩码不同造成的输出差异算作内核错误。
4. 差异是否随 5s/10s token 数、prompt、层、denoise step 或轨迹反馈扩大？给出一个能解释符号变化的机制，或明确说明证据不足。最后给出“确认 bug / 未发现 bug 但存在预期数值或路由差异 / 尚无法判定”之一，并附可复现证据。

## 执行顺序

### 1. 预检和静态契约

- 阅读 `h3_sparse_attention/landmark_tree_v2.py`、`landmark_v2_fused_node.py`、`spark_integration.py`、`sol_topk_cutoff.py`、`sol_numerator_virtual_q.py`、`spark_reweight_sm120.py` 中对应分支。列出两轴实际改变的代码路径、共同路径及预期等价关系。重点核实 legacy 中间 BF16 rounding 与 fused 方向构造的 FP32 中间值，以及 `threshold` 与显式 Top-K 的分数精度、`>` 边界和 tie 规则。
- 检查 GPU 占用及现有 `three_task_queue_4gpu_20261004` 状态，使用空闲 GPU 和独立输出目录；不要干扰正在运行的任务。记录 GPU 型号、CUDA/Torch/Triton 版本、环境变量、源码 SHA256、配置和输入摘要。已有测试 `tests/test_packed_external_no_route_qk.py`、`tests/test_topk_host_synchronization.py` 可作为最小对照；其通过只证明覆盖的形状与路径。
- 预先确定诊断样本：从已有 `full50.paired_quality` 中按绝对差异选每个时长的 2 个极端 prompt，另选 1 个近零 prompt；记录选择规则和 case ID。这些是定位用样本，不能把它们的均值当成新的总体效果估计。

### 2. 固定输入的分层诊断

- 首先用小形状确定性输入做单元级实验，覆盖正常分数、Top-K 边界 ties、非 64 整除的视频尾部以及非 64 整除的 context。检查 permutation 双射、各节点容量、K/V 同步、Q 逆置换、选块预算、local/sink 强制 exact 和有限输出。为确认的错误添加能失败的最小回归测试。
- 再从真实 5s 与 10s 推理中抓取少量代表层与 step 的**同一份** Q/K/V、reblock 输入和所需元数据。优先通过不改变随机数消耗和生产输出的 hook 捕获；核对 hook 前后的原配置输出一致。保存张量哈希、shape、dtype、layout、case/step/layer，并控制捕获数量和磁盘占用。若已有兼容 capture 可复用，先校验来源与源码哈希。
- 对每份 capture 在相同输入上运行 2×2 组合。按“方向/置换 → route score/掩码 → 单层 attention 输出”顺序逐级比较；报告首个出现差异的阶段和每阶段的数值分布，不只报告最大误差。至少记录 mask disagreement、Jaccard、每行 exact 块数、方向 cosine/相对误差、输出相对 L2/最大绝对误差，以及与同输入 Dense attention 的误差。分别统计普通块、local 块、sink/context 和尾块。对于多头多层数据给出聚合和最坏位置，不把全部 token 当作独立统计样本。
- 做两个隔离干预：固定 legacy permutation 后只切换 threshold/external；固定同一 route mask 后只切换执行实现。必要时再固定 fused permutation 重复。若 `packed_external_no_route_qk` 与 `packed_external` 在相同输入和掩码下不一致，按优先级检查 packed bit 解码、QK 跳过条件、近似质量累积及编译缓存键；先形成最小复现，再认定 bug。若仅因舍入改变分组或边界选块，量化变化并说明其是否符合代码契约。

### 3. 轨迹与完整生成核对

- 在固定输入测试定位后，对上述少量真实 prompt 做四臂完整生成，保持原 2×2 设置和 matched Dense 参考；优先复用经 hash 验证的历史视频与结果，只对无法回答的轨迹问题新增生成。记录每次推理的 layer/step 差异如何演化，并与最终 PSNR/SSIM/LPIPS 对照。若中途源代码修复，修复前后必须分开命名，不混进原 50-prompt 统计。
- 只在发现明确、可复现的错误时提交最小修复和回归测试，并重跑受影响的固定输入诊断及必要的少量生成；不要借诊断调整 Top-K 比例、anchor dtype、reblock 参数或生产默认值。若没有错误，报告合理的数值或路由差异及仍未排除的风险即可，不以扩大 50-prompt 生成代替定位。

## 交付与验收

将可复现脚本、测试、配置、逐 capture 的指标和日志保存在新的独立实验目录；在 `docs/check_fused_extern_results.md` 写总结。报告至少包含：使用的数据与源码哈希、四臂配置核对表、固定输入逐阶段差异表、掩码与同掩码输出对照、5s/10s 轨迹案例、是否发现确定性错误、最小复现或修复验证，以及对原 50-prompt 结果的审慎解释。明确哪些比较要求 bitwise equality，哪些只要求数值接近，哪些路径本来允许选择不同的块；不要把“不显著”写成“等价”。
