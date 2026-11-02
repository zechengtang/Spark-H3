# Spark chunked：两种保留实现

当前源码只保留以下两种 Python 分块调度。两者都先完成当前输入的 reblock，再执行分块 producer 和 BSA/reweight。

`drop_k` / `keep_k` 指规划阶段是否保留完整原始 K：`drop_k` 随后重算 K；`keep_k` 将保留的 K 量化后也会释放，不跨层或跨采样步保留 K。

| 底层方案 | 含义 | 生命周期 |
|---|---|---|
| `keep_k`（方案 2，暂不在节点开放） | 原始 K + 分块 Q/V | 保留 norm/RoPE 后的 K 和小变换矩阵，聚类按需生成有界变换块；规划后生成量化 K，释放 BF16 K，再分块投影 Q/V |
| `drop_k`（方案 1） | 变换 K + 分块 QKV 重算 | 仅保留变换 K 特征表用于规划；释放该表后，重新分块投影 QKV |

Spark 节点统一使用 `chunk_path`，只有 `none`（默认，非分块 Spark）和 `drop_k` 两个选项；移除原来的 `use_chunked` 和 `chunked_scheme`。

自定义 Sol 节点的 `chunk_path` 提供 `chunked`（默认）和 `none`：前者沿用官方 H3 分块 producer；后者取消 H3 专用分块 block patch，使用模型原始 QKV 投影，再由官方 attention override 调用非分块 `sol_attn`。两者都沿用官方稀疏窗口、sink 和 dense 层策略，预热百分比映射保持 0.1.5 行为。`none` 的完整浮点 QKV 会同时驻留，可能增加峰值显存。也可以直接选择 ComfyUI 官方 `Model Sparse Attention` 节点中的 `sol-attn`。

节点调用未提供 `chunk_path` 时使用 `none`；传入 `keep_k` 会报错。内部状态也统一使用 `chunk_path`，删除旧布尔开关、旧方案参数及 `H3_SPARK_SPLIT_SCHEME` 环境变量入口，不保留旧分块参数或工作流迁移逻辑。`keep_k` 仅保留底层实现及实验测试。

## 构建与使用

两种方案均需要带有 Spark 原生 BSA/reweight 核的 comfy-kitchen，以及独立的 split producer 动态库。当前验证环境包括 Linux、SM86/SM120、ComfyUI 0.38.2；其他平台需要单独构建和验证。下例构建 SM86；SM120 使用 `--arch 120`。

```bash
python scripts/build_spark_split.py \
  --kitchen /path/to/comfy-kitchen \
  --output /path/to/libproducer.so --arch 86
export H3_SPARK_SPLIT_LIBRARY=/path/to/libproducer.so
# 之后在同一环境启动 ComfyUI。
```

在 Spark 节点中设置 `chunk_path=drop_k`。支持默认 `full`、`q_reuse_k`、`legacy` midpoint 配置；输入要求 BF16、head_dim=128、batch=1，`producer_chunk` 必须为 64 的正整数倍。`topk_only`、`reblock_only`、`reweight_only`、跨层排列复用、Spark 配置中的层次化 head 分组规划、独立 Q/K 规划或 fused midpoint 在启用 chunked 时会提前报错。这里的“层次化 head 分组规划”不包括 KJNodes 的 `MiniMaxLowVRAMAttention`：后者的 `head_chunks` 已支持，会在每个 head group 内继续按 token chunk 执行 drop_k。缺少动态库时会提示构建方法，不回到旧实现。ComfyUI Spark 的 fused midpoint 已由上游整体禁用，`chunk_path=none` 也不接受该设置。

KJNodes 的两个 MiniMax H3 低显存节点必须连接在 Spark 节点之前。`MiniMaxChunkFeedForward` 原样保留；`MiniMaxLowVRAMAttention` 的 head groups 与 token chunks 组合，每组独立维护 bootstrap/steady 统计及 reblock plan，最终恢复完整 head 布局。非 chunk Spark 沿用 KJ 的投影后 compose hook，chunked Spark 使用投影前的 split producer，因此两条路径的内部调度不同但节点连接顺序相同。更多 head groups 通常降低 Spark 临时工作区峰值，但会重复调度 reblock/producer，可能显著增加耗时。

SelfLift 的 dense 低分辨率与 sparse 高分辨率两阶段配置应只把 Spark 接到高分辨率分支，并设置零 warmup、零 dense layers。使用 spatial tiling 时，Spark 的 `steps` 计数上限必须覆盖全部 tile 模型前向，例如 3 个高分辨率采样步 × 4 个 tile 需要至少 12；SelfLift 的采样步数和 sigmas 不变。过小的计数上限会在同一轮采样中重置 Spark 状态。当前 SelfLift 适配会拒绝额外顶层 attention override；高分支的强制全 sparse 测试移除了已知的 dense backend override，低分支保留原 backend。

## 已删除的旧路径

- H3 的 `_native_spark_attention_chunked` 和 `comfy_kitchen_spark_attention_chunked` 调度。
- 旧 CPU K staging、token-stream 和 feature-stream 规划模块及专用测试/诊断脚本。
- 相邻 comfy-kitchen 源码中的 Python `spark_attn_chunked`、`_spark_attn_chunked_native` 及公开导出。

`_C.spark_attn_chunked_core` 仍是两种新方案复用的底层量化重排、路由、exact attention 和 reweight 核。保留这个原生符号不表示旧 Python 调度仍可用。历史测试归档、已安装环境与历史发布包不会随源码删除而改写；复现旧实验应使用其归档源码。

## 验证与回放

`tests/test_spark_split.py` 覆盖两方案的 bootstrap/steady、非整块输入、原生 producer 对照、普通与 INT8 ConvRot 投影、KJ 不均匀 head groups，以及真实 Comfy 投影到节点输出的完整调用。`tests/test_comfyui_plugin.py` 覆盖默认值、方案选择、KJ 组合安装、非法配置和状态清理。

```bash
# PYTHONPATH 中放置 H3、ComfyUI 和编译好的 comfy-kitchen；使用运行 ComfyUI 的 Python。
python -m pytest tests/test_spark_split.py tests/test_comfyui_plugin.py -q
python scripts/benchmark_spark_split.py \
  --capture /path/to/native_eval3_layer0.pt \
  --output /path/to/replay.json --chunk 4096
```

回放从 GPU 驻留原始 QKV 开始，排除线性投影、CPU 传输和 VAE；不能替代端到端显存及耗时。此前各一次预热、两轮正式 5s768p 测试见 [实验报告](spark_split_5s768p_20261031.md)。该报告及归档记录当时使用环境变量切换方案的代码，当前节点入口已按本文更新。
