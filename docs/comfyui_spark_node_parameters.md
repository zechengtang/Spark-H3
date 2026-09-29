# ComfyUI Spark 节点参数

节点名称：`MiniMax H3 Spark Attention (SM120)`。把它接在 MiniMax H3 模型加载节点与采样器之间。界面中的每个输入也有中文悬浮说明。

`warmup_mode` 显示在预热参数上方，`topk_mode` 显示在 Top-K 参数上方。每组只显示当前模式使用的数值框；切换模式不会清除另一模式原来填写的值。如果把模式接成外部输入，界面会显示该组的两个数值框，因为实际模式要到运行时才能确定。

## 常用设置

默认使用 `topk_mode=topk_ratio`、`topk_ratio=0.2`；切换到固定块数模式时，`topk_blocks` 默认是 `228`。10% 更偏速度，30% 更偏保真度，建议用 15% 或 20% 取得均衡表现。预热通常设 `warmup_percent=20–25`；界面单位是百分数，填 `0.2` 只代表 0.2%。`steps` 填实际采样调用次数。

在本地 10 秒 768p 样本中，视频有 72,576 个 token，按 64 token 一块分成 1,134 个候选视频块。因此 `topk_blocks=114/228/342` 分别约为 **10%/20%/30% 精确保留比例**。在这几个预算中，10% 更偏速度，30% 更偏保真度，建议从 20% 开始。这里的“10%”是精确选中的候选块比例，不是 10% 被跳过；未选中块仍可经过 Spark 的近似校正。改变时长、分辨率或视频 token 布局时，相同块数对应的百分比也会变化。

## 所有输入

| 参数 | 作用与建议 |
| --- | --- |
| `model` | 要打补丁的 ComfyUI 原生 MiniMax H3 `MODEL`。连接模型加载节点输出，Spark 节点输出再接采样器。 |
| `enabled` | 是否启用 Spark。关闭时直接传出原模型，便于在同一工作流里做 dense 对照。 |
| `steps` | 一次采样的模型调用次数，用于按步数计算预热。通常与采样器的 `steps` 相同；应按实际模型调用次数设置。 |
| `warmup_percent` | 开始时保持 dense 的采样调用百分比；建议 **20–25**。例如 20 步采样填 `20`，前 4 次调用为 dense。仅在 `warmup_mode=warmup_percent` 时生效。 |
| `topk_ratio` | 精确计算的候选视频块比例：`0.1≈10%` 速度最快，`0.3≈30%` 保真度最好，建议 `0.15` 或 `0.2` 取得均衡。仅在 `topk_mode=topk_ratio` 时生效。 |
| `dense_layers` | 前多少个 Transformer 层始终使用 dense attention。默认 `1` 表示第 0 层；`0` 表示没有固定 dense 层。 |
| `min_tokens` | 序列 token 数低于此阈值时保持 dense。默认 `12288`，避免短序列走稀疏路径的固定开销；它不是 Top-K 的块数。 |
| `strict` | 开启时，遇到 Spark 不支持的输入直接报错；关闭时回退到原 attention 并在日志中记录原因。建议调试时开启，以免误把回退结果当作 Spark。 |
| `topk_mode` | Top-K 预算方式：默认 `topk_ratio`，用随视频长度变化的比例；`topk_blocks` 用固定块数。 |
| `topk_blocks` | 每个 query 块精确计算的候选视频块数。仅在 `topk_mode=topk_blocks` 时生效；本地 10 秒 768p 样本可参考 `114≈10%`、`228≈20%`、`342≈30%`，建议先用 `228`。实际保留数会受候选块总数限制；额外的 dense sink 不计入这个预算。 |
| `warmup_mode` | 预热长度的表示方式：`warmup_percent` 使用百分比，`warmup_steps` 使用固定次数。只读取当前模式对应的参数。 |
| `warmup_steps` | 前多少次模型调用保持 dense。仅在 `warmup_mode=warmup_steps` 时生效；20 步采样设 `4` 相当于 20% 预热。 |
| `tail_granularity` | 近似分支的 Q 粒度。`query`（默认）对应 sol-engine 的 Sol 实现思路：只对 K/V 下采样，每条真实 Q 分别消费近似汇总。`block` 对应 ComfyUI 的 Sol 实现思路：Q 和 K/V 都按块下采样，同一 Q 块共享近似结果。Spark 的精确分支在两种模式下仍使用真实 Q；此参数不控制视频尾块的 dense 策略。 |

`warmup_percent`、`topk_ratio` 与 `topk_blocks` 采用不同单位：前者填 `20` 表示 20%，比例参数填 `0.2` 表示 20%，固定块数填 `228` 表示 228 个块。`warmup_mode` 和 `topk_mode` 分别决定哪一个输入真正生效。
