# Table 1/2 本地可复现性审计（2026-09-27）

本报告只审计 `docs/blogs/spark-attn/README.md` 中 Spark-Reblock 与 Spark-Reweight 的两张四条 prompt 消融表；没有修改展示 blog 或历史结果。必须区分两件事：**从保存视频和 Q/K capture 重算表格**，以及**从模型权重重新去噪生成 latent**。前者已经独立完成且逐数一致；后者在下文的独立目录中验证，不能由前者自动推定。

## 原始产物与协议

- 原始实验：`/autodl-fs/data/h3_experiments/blog_ablation_4prompts_rerun_20260922/`；视频和 latent：`/autodl-fs/data/h3_outputs/blog_ablation_4prompts_rerun_20260922/`。
- prompt 来自 Benchmark 的 `vbench_core5_percent_subsets/blog_ablation_4prompts/samples.json`，索引 1–4：`vbench_all_0326`（长颈鹿）、`0705`（泰迪熊）、`0729`（柯基）、`0763`（北极熊）。原始 `protocol.json` 固定 seed 42、20 个时间网格点／19 次实际 denoiser evaluation、240 请求帧（VAE 按内部约束圆整）、1344×768、24 fps。
- 四条路径：Dense 参考；TopK10、flat64、无 reblock/reweight 的 BSA；BSA + LMv2 reblock（fanout 16）；BSA + global reweight（不 reblock）。四组各有 4 个 latent、4 个无损 FFV1 视频与 generation manifest。另有 Dense 轨迹在 eval 9/18、layer 1/24/49 的 24 个 Q/K captures。
- 历史 `runner_source.py`、`protocol.json`、`source_manifest.json`、`snapshot/{h3_sparse_attention,sol_attn}` 均在原始实验目录；对 source manifest 的 85 个 Python 文件重新执行 SHA-256 检查，85/85 通过。评分协议是保存的 Dense 视频作为同 prompt/seed 参考，对候选视频计算 pooled RGB-MSE PSNR、Gaussian SSIM、AlexNet LPIPS；attention-mass recall 与 log-mass error 来自上述冻结 Q/K captures。

## 有效 `torch.compile` 状态

Table 1/2 的历史脚本虽然没有传 `--no-torch-compile`，历史 Benchmark 的 `load_denoiser` 也把 `torch_compile=True` 传给 `install_h3_acceleration`，**实际却没有生效**。原因是该实验保存的实现快照中，`h3_sparse_attention/__init__.py` 没有导出 `H3AccelerationConfig` 和 `install_h3_acceleration`，也没有 `acceleration.py`。当时 Benchmark 提交 `4562d25` 的 `scripts/_impl_bootstrap.py` 遇到此 `ImportError` 会返回 `_NoOpAccelerationPlugin`，它的 `__enter__` 什么都不做。用这份冻结快照和历史 Bootstrap 做独立检查，`H3AccelerationConfig(torch_compile=True)` 得到 `_NoOpAccelerationPlugin`，且无 `_forward_originals` 包装记录。因此这些产物应归类为**实际未启用 transformer block 编译**，不能与 Table 4 的有效编译结果直接比较绝对 PSNR。原始 denoise 记录没有逐条保存真实包装块数，这一判断基于原始调用路径与完整源码快照，而非误把配置布尔值当成运行状态。

## 从保存产物独立重算

评分重算写入 `/autodl-fs/data/h3_experiments/blog_table12_repro_audit_20260927/`，使用新的 `quality_*` 目录；旧评分文件没有被覆盖。全部 4 条 prompt、3 个候选路径的 PSNR/SSIM/LPIPS 与旧评分 JSON **逐数一致**：

| 路径 | PSNR dB | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| BSA | 18.610669 | 0.677198 | 0.263338 |
| BSA + Reblock | 22.407563 | 0.798495 | 0.138026 |
| BSA + Reweight | 19.169964 | 0.702211 | 0.238717 |

独立重算冻结 capture 的 TopK10 attention-mass recall 为 flat64 **68.657755%**、fanout16 reblock **81.536559%**，与原始 JSON 逐数一致，按 blog 精度分别为 68.66% 和 81.54%。Table 2 的 log-mass error 重算结果见同一独立目录的 `jensen_results_recomputed.json`；其源指标为 BSA 3.861228、global reweight 2.780082 nats，按 blog 精度为 3.86 和 2.78。

这证明 Table 1/2 从已保存的视频及 Q/K capture **可重算**，不是从头推理的 bitwise 证明。

## 从头 denoise 重放

独立环境使用历史 Benchmark 提交 `4562d25` 的 detached worktree（保留其原始 no-op Bootstrap），配合原始实验保存的实现 snapshot。新脚本 `replay_case1.py` 将历史 `runner_source.py` 的输出重定位到独立的 `denoise_replay_case1*` 和 `outputs_case1*` 目录；复用原始 conditioning 缓存的逐字节副本，不覆盖任何历史 latent。各方法在独立进程和目录运行。比较脚本 `compare_latents.py` 对视频和音频 latent 做逐元素 bitwise、MSE、最大绝对误差及文件 SHA-256 检查。

对 prompt 1（`vbench_all_0326`），**四条方法全部从头去噪复现**：

| 方法 | 视频 latent bitwise | 音频 latent bitwise | `.pt` SHA-256 相同 |
| --- | --- | --- | --- |
| Dense | 是 | 是 | 是 |
| BSA | 是 | 是 | 是 |
| BSA + Reblock | 是 | 是 | 是 |
| BSA + Global Reweight | 是 | 是 | 是 |

四条视频 latent 均为 float32 `[1,24,72,48,84]`，音频 latent 均为 float32 `[2,32,405]`；每项 MSE 和最大绝对误差均为 **0**。机器可读记录为独立目录中的 `latent_compare_{dense,bsa,reblock,global_reweight}_case1*.json`。这比仅重算保存视频的分数强得多，但仍是 **1/4 prompt 的全方法抽样**，不是对四条 prompt 全量 bitwise 重跑。

## 限制

历史 runner 保存了自身副本和注意力实现快照，但未把完整 Benchmark 与 Diffusers 依赖打包；此次重放以 Git 提交 `4562d25` 的 Benchmark 作为对应版本。模型权重和底层 Python/CUDA 环境仍在归档外。四条 prompt 中仅索引 1 从头 bitwise 重放，剩余三条保留了完整保存产物及评分重算证据。历史无有效编译这一事实也意味着 Table 1/2 与 Table 4 只能分别解释组内消融，不能跨表比较绝对 PSNR。
