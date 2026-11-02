# Spark split 两方案本地端到端基准（SM120，2026-11-01）

> **已撤回（2026-11-01）**：本文 LightX2V 5s/10s 数据仍有效，但 SelfLift 8 工作流数据无效。旧 runner 没有用 Spark 替换高分辨率 Sol 路径；其所谓“完整高分辨率 Spark”失败实验还错误地在 223,677-token 高清阶段先执行了两次 dense warm-up。该失败不能用于判断 chunked Spark 的高清能力。请勿引用本文 SelfLift 表格或其结论；修正后的全链路 Spark 重跑完成后将由新报告替代。

## 结论

本次在一张 NVIDIA RTX PRO 6000 Blackwell Server Edition（97,887 MiB）上，对 `feature_replay` 与 `resident_k` 两套 chunked Spark split 方案完成了：

- LightX2V 5s/768p、10s/768p 的受控对比；
- `assets/selflift-avatar` 下素材完整的 8 个 15s 工作流的两方案运行；
- 16 个 SelfLift 输出视频的帧数、分辨率、帧率和音轨校验。

受控的 LightX2V 结果最清晰：`resident_k` 比 `feature_replay` 慢约 0.70%–1.59%，但峰值显存少 126–142 MiB（0.25%–0.31%）。差距很小，目前没有证据表明 `resident_k` 在此卡和参数下具有性能优势；如果优先速度，选择 `feature_replay`，如果非常在意最后约 0.1 GiB 显存，可选择 `resident_k`。

SelfLift 的高分辨率尾段不能直接改为当前 Spark split，详见“兼容性边界”。因此 SelfLift 数字代表“低分辨率主阶段使用待测方案，高分辨率尾段保持原工作流后端”的可运行组合，不应表述成全链路 Spark 对比。其结果受不同模型拓扑、DynamicVRAM 换入换出及尾段后端影响较大，没有显示出跨 8 个工作流稳定的方案优势。

## 是否重复加载权重

没有逐工作流、逐方案重新启动服务或重新加载全部权重。测试使用单个常驻 ComfyUI 进程：

- LightX2V：一次冷启动/模型加载，先做两方案 warm-up，再连续执行 4 个正式 case；
- SelfLift：按 3 组模型/拓扑组织，每组在同一进程内做两方案 warm-up，随后连续执行正式 case；同组内复用已加载权重；
- 显式使用 `--cache-classic` 和 DynamicVRAM；因工作流确实需要切换 Hybrid/FL/Heretic 等不同模型，组间仍存在必要的模型物化或换入换出，但不是每个方案重复启动加载。

两个正式方案的每个 case 均记录到 200 或 300 次 scheme 调用，bootstrap/steady 计数完整且无 fallback，说明测到的是指定实现，而不是静默回退路径。

## 环境与协议

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition，SM120，97,887 MiB |
| PyTorch / CUDA | 2.14.1+cu130 |
| Triton | 3.8.0 |
| ComfyUI | 0.38.2 |
| comfy-kitchen | 0.2.35 |
| Spark chunk size | 4096 |
| 服务参数 | `--reserve-vram 1.5 --cache-classic --disable-fast-disk --preview-method none` |
| 原生库 | `libproducer_sm120.so`，SHA-256 `3b814dce29962c1cabf714a31de8ab989cf8343e6a502ee84acf36a91bc1ab2d` |

显存是通过 NVML 采样的整个 ComfyUI 进程峰值，并非仅算子分配量。时间中的 sampler 为采样节点执行时间；SelfLift 的 `low` 是低分辨率主阶段时间。正式测试前已执行 4 次 LightX2V eval warm-up；SelfLift 每个模型/拓扑组两方案各做一次完整 warm-up。

回归测试：`tests/test_spark_split.py tests/test_comfyui_plugin.py` 共 58 项通过，用时 136.23s。

## LightX2V：5s/768p 与 10s/768p

分辨率为 1344×768。每个正式 case 有 300 次 scheme 调用（50 次 bootstrap、250 次 steady），均无 fallback。

| 时长 / 帧数 | 方案 | sampler 时间 | workflow wall | 进程峰值显存 |
|---|---|---:|---:|---:|
| 5s / 124f | `feature_replay` | 51.374s | 60.800s | 45,982 MiB |
| 5s / 124f | `resident_k` | 52.191s | 61.653s | 45,840 MiB |
| 10s / 243f | `feature_replay` | 125.414s | 144.504s | 50,144 MiB |
| 10s / 243f | `resident_k` | 126.289s | 145.077s | 50,018 MiB |

相对 `feature_replay`：

| Case | `resident_k` sampler 变化 | `resident_k` 峰值显存变化 |
|---|---:|---:|
| 5s / 124f | +1.59% | -142 MiB（-0.31%） |
| 10s / 243f | +0.70% | -126 MiB（-0.25%） |

## SelfLift：8 个工作流

原始工作流均为 15s / 362f。00325、00327、00331、00340、00399 输出 1920×1088；00388、00391、00392 输出 1664×1248。每个正式 case 有 200 次 scheme 调用（50 次 bootstrap、150 次 steady），均无 fallback。

表中 `sampler` 是完整采样节点时间，`low` 是真正切换两方案的低分辨率阶段；`peak` 是完整进程峰值，`low peak` 是低分辨率阶段峰值。单位分别为秒和 MiB。

| ID | 方案 | sampler | low | peak | low peak | 备注 |
|---|---|---:|---:|---:|---:|---|
| 00325 | `feature_replay` | 769.83 | 515.42 | 55,760 | 50,160 | 首个正式 case 含 Hybrid→FL 模型物化，时间污染 |
| 00325 | `resident_k` | 321.06 | 66.59 | 54,178 | 48,082 |  |
| 00327 | `feature_replay` | 321.45 | 66.72 | 54,162 | 48,226 |  |
| 00327 | `resident_k` | 325.73 | 71.23 | 54,178 | 48,082 |  |
| 00331 | `feature_replay` | 326.37 | 68.52 | 54,234 | 48,354 |  |
| 00331 | `resident_k` | 329.85 | 72.00 | 54,210 | 48,202 |  |
| 00340 | `feature_replay` | 702.68 | 507.71 | 50,074 | 47,546 | 首个正式 case 含 FL→Hybrid 模型物化，时间污染 |
| 00340 | `resident_k` | 267.89 | 90.57 | 50,050 | 47,650 |  |
| 00388 | `feature_replay` | 286.50 | 120.37 | 61,402 | 58,874 |  |
| 00388 | `resident_k` | 323.43 | 128.91 | 61,442 | 58,978 |  |
| 00391 | `feature_replay` | 231.32 | 79.33 | 61,434 | 58,874 |  |
| 00391 | `resident_k` | 319.85 | 130.98 | 61,442 | 58,850 |  |
| 00392 | `feature_replay` | 272.05 | 86.49 | 61,402 | 59,034 |  |
| 00392 | `resident_k` | 309.34 | 136.83 | 61,410 | 58,882 |  |
| 00399 | `feature_replay` | 316.67 | 126.20 | 61,266 | 58,866 |  |
| 00399 | `resident_k` | 342.71 | 110.07 | 61,314 | 58,722 |  |

00325 和 00340 的 `feature_replay` 时间因正式序列中的模型组首次物化而不可与其 `resident_k` 直接比较。其余 6 个工作流的描述性中位数为：完整 sampler 301.59s vs 324.58s（`resident_k` +7.62%）；低分辨率阶段 82.91s vs 119.49s（`resident_k` +44.12%）。但这些不是严格的算法平均值：它们混合了不同工作流、不同尾段后端和顺序/DynamicVRAM 效应。尤其 00399 的 `resident_k` 低分辨率阶段反而快 12.79%，说明 SelfLift 组内波动明显，不能据此宣称固定加速或减速比例。

完整峰值显存主要由共同的高分辨率尾段决定，所以两方案差异很小。排除两个模型物化污染 case 后，6 个工作流的完整峰值中位数为 61,334 MiB vs 61,362 MiB；低分辨率阶段峰值中位数为 58,870 MiB vs 58,786 MiB，`resident_k` 低约 84 MiB。

## 兼容性边界与失败证据

SelfLift 原始 8 个工作流并不使用 Spark：它们使用 KJ BlockSparseAttention/Sol 或普通 attention，并通过高分辨率空间 tile 控制开销。本次尝试了两种更激进的全链路接入：

1. 在原高分辨率空间 tile 上直接启用 Spark：不同 tile token 长度切换后触发 native abort；
2. 关闭 tile、在完整高分辨率 token 序列上启用 Spark：前两个 dense 步峰值约 65 GiB，第三步 Spark 分配失败，随后 comfy-kitchen INT8 linear 触发 fatal abort。

因此最终采用可重复、可完成的协议：6 次低分辨率 evaluation 使用被测 Spark split 方案；3 次高分辨率 evaluation 保留各工作流原后端。00340 的低分辨率分支另行克隆了 ChunkFeedForward，并绕过 KJ head-group attention，因为当前 split Spark 明确不接受投影后的 KJ low-VRAM override；高分辨率分支仍保留原 KJ low-VRAM + Sol。

失败日志：

- `artifacts/spark_split_benchmark_sm120_20261101/failures/spatial_spark_abort.log`
- `artifacts/spark_split_benchmark_sm120_20261101/failures/fullres_spark_abort.log`

## 输出校验与可复现资料

16 个 SelfLift MP4 均通过 ffprobe 校验：362 帧、24 fps、分辨率与原工作流一致，并带音轨。

对两方案输出做了封装流 packet hash：00325、00327、00331 的视频流和音频流 hash 一致；00340、00388、00391、00392、00399 不一致。该检查是 encoded packet hash，并非解码帧逐像素检查，因此只能说明前三个输出在该层面一致，不能把后五个描述为端到端 bitwise 等价。

原始结果与运行器：

- `artifacts/spark_split_benchmark_sm120_20261101/raw/lightx2v_result.json`
- `artifacts/spark_split_benchmark_sm120_20261101/raw/selflift_result.json`
- `artifacts/spark_split_benchmark_sm120_20261101/raw/persistent_lightx2v_runner.py`
- `artifacts/spark_split_benchmark_sm120_20261101/raw/persistent_selflift_runner.py`
- `artifacts/spark_split_benchmark_sm120_20261101/raw/libproducer_sm120.build.json`

## 建议

当前默认建议保持 `feature_replay`。它在受控 5s/10s 测试中稳定略快，显存代价仅约 0.13 GiB。`resident_k` 可作为显存边缘场景的备选，但在投入默认路径前，建议增加 ABBA/交叉顺序重复测试，并先修复高分辨率 tile 长度切换的 native 生命周期问题；否则 SelfLift 结果会持续被模型换入换出和公共尾段噪声主导。
