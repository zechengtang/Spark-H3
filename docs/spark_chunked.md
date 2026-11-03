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

两种方案均需要带有 `_C.spark_attn_chunked_core` 的 comfy-kitchen，以及独立的 split producer 动态库。仓库中的 `comfy-kitchen-spark-v0.2.36.patch`（同时用于 0.2.37）和 `comfy-kitchen-spark-v0.2.39.patch` 已包含该入口、C++ 绑定、量化载体重排内核及载体 reweight 内核。它们由现有 `comfyui/kernel_builder.py` 构建流程自动应用。

**旧版 0.1.5 预编译 wheel 不包含这个分块入口；只编译 `libproducer.so` 不够。** 必须先按当前仓库补丁重建并安装 Kitchen，再构建 producer。`chunk_path=none` 仍可使用原来的非分块后端。以下是 Linux / SM120 / CUDA 13.0 / ComfyUI 0.38.2 的完整重建步骤，使用 ComfyUI 自己的 Python；构建目录须为新的空目录。SM86/SM89 对应将 `SPARK_ARCH`、`COMFY_CUDA_ARCHS`、`--arch` 分别改为 `sm86/86/86` 或 `sm89/89/89`。

### 1. 从干净的固定版本源码构建 Kitchen

`0.38.x` 使用 Kitchen `0.2.36`，`0.39.x/0.40.x` 使用 `0.2.37`，`0.42.x` 使用 `0.2.39`。准备 CUDA 13.0 Toolkit、Git 和 C++ 构建工具，并确保 GitHub/PyPI 可访问。

```bash
export SPARK_ROOT=/absolute/path/to/Spark-H3
export SPARK_PY=/absolute/path/to/comfyui/venv/bin/python
export SPARK_BUILD_ROOT=/absolute/path/to/new-spark-build
export SPARK_KITCHEN_BASE=0.2.36
export SPARK_ARCH=sm120
export COMFY_CUDA_ARCHS=120f
export COMFY_BUILD_JOBS=4
mkdir -p "$SPARK_BUILD_ROOT/wheels"

# 与 kernel_builder CLI 使用同一个源码准备函数；保留源码供 producer 编译。
"$SPARK_PY" - <<'PY'
import os, sys
from pathlib import Path
root = Path(os.environ['SPARK_ROOT'])
sys.path.insert(0, str(root / 'comfyui'))
from kernel_builder import prepare_source, KITCHEN_SOURCE
prepare_source(
    root, Path(os.environ['SPARK_BUILD_ROOT']) / 'comfy-kitchen',
    source_url=KITCHEN_SOURCE,
    kitchen_base=os.environ['SPARK_KITCHEN_BASE'],
    architecture=os.environ['SPARK_ARCH'], cuda_tag='cu130',
)
PY

"$SPARK_PY" -m pip wheel --no-deps \
  --wheel-dir "$SPARK_BUILD_ROOT/wheels" "$SPARK_BUILD_ROOT/comfy-kitchen"
"$SPARK_PY" -m pip install --no-deps --force-reinstall \
  "$SPARK_BUILD_ROOT"/wheels/comfy_kitchen-*.whl

# 确认当前 Python 加载的确实是新扩展，不以版本号代替接口检查。
"$SPARK_PY" - <<'PY'
from comfy_kitchen.backends import cuda
print(cuda.__file__)
print(cuda._C.__file__)
for name in ('spark_attn', 'spark_attn_chunked_core', 'sol_attn_plan',
             'sol_producer_begin', 'sol_producer_chunk'):
    assert callable(getattr(cuda._C, name, None)), name
print('Spark native APIs: OK')
PY
```

### 2. 使用同一份 Kitchen 头文件构建 producer

```bash
"$SPARK_PY" "$SPARK_ROOT/scripts/build_spark_split.py" \
  --kitchen "$SPARK_BUILD_ROOT/comfy-kitchen" \
  --output "$SPARK_BUILD_ROOT/libproducer.so" --arch 120
export H3_SPARK_SPLIT_LIBRARY="$SPARK_BUILD_ROOT/libproducer.so"
# 之后在同一环境启动 ComfyUI。
```

当前运行验证使用 Linux；Windows 和其他 GPU 架构的构建及运行需另行验证。独立 producer 的构建脚本目前使用 Linux 编译参数。

### 3. 验证分块执行

在完整 Spark-H3 源码目录下执行。测试会实际运行 bootstrap/steady、`drop_k`/内部 `keep_k`、KJ 分组和分配器检查；缺少 CUDA 或 `H3_SPARK_SPLIT_LIBRARY` 时会跳过，所以验收时必须检查测试结果而不能只看退出码。

```bash
cd "$SPARK_ROOT"
export COMFYUI_ROOT=/absolute/path/to/ComfyUI
PYTHONPATH="$SPARK_ROOT:$COMFYUI_ROOT" "$SPARK_PY" -m pytest -q -ra \
  tests/test_spark_split.py tests/test_spark_split_allocator.py \
  tests/test_comfyui_plugin.py
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
