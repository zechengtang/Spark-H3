# MiniMax H3 sparse scope conventions

> 👉 [Ref2VA 任务加速解决方案](#ref2va-任务加速解决方案)

本文统一描述 MiniMax H3 的 FL2VA 与 Ref2VA 稀疏注意力范围。这里的范围是加速方法的约定，不是 H3 dense attention 自带的 mask。

## 记号

箭头左侧是 query，右侧是 key/value：

```text
Q -> KV
```

本文使用以下 token 角色。为避免引入任务实现细节，FL2VA 的首尾帧与 Ref2VA 的参考素材统一称为 reference：

| 记号 | 含义 |
|---|---|
| `T` | text 与 VLM context |
| `V` | reference visual token，包括 FL2VA 首尾帧以及 Ref2VA 参考图和参考视频 |
| `A` | reference audio token，用于 Ref2VA 参考音频 |
| `V_t` | target-video token |
| `A_t` | target-audio token |

本文还区分两个容易混淆的概念：

- **dense query**：该 query 行计算所有有效 K/V。
- **exact K/V（精确保留的 K/V）**：在原有 attention mask 允许的范围内，这些 K/V 不经过 Top-K、路由或近似计算；这不表示对应 query 行本身一定 dense。

下文的 Query × K/V 矩阵使用统一标记：

- **dense**：该角色对在原有 attention mask 允许的范围内精确计算，不经过稀疏选择。
- **sparse**：该角色对经过 Top-K、路由或其他近似计算。

因此，一行全部为 **dense** 才表示 dense query；某个单元格为 **dense**，则表示该列的 K/V 对该行 query 精确保留。

## FL2VA：target-video-only

FL2VA 常用约定是只压缩目标视频内部的注意力：

```text
V_t -> V_t sparse
```

对应关系为：

| Query \ K/V | `T + V + A_t` | `V_t` |
|---|---:|---:|
| `T + V + A_t` | **dense** | **dense** |
| `V_t` | **dense** | **sparse** |

也就是说：

- 文本、reference visual 和目标音频保持 dense query。
- 它们同时作为 exact K/V，对所有目标视频 query 可见。
- 只有 `V_t -> V_t` 使用 Top-K、窗口或其他稀疏/近似分支。

FastH3、VDN 以及 Spark 的 `sparse_scope=target-video` 都属于这一范围约定，但具体稀疏算子不同。FastH3 和 VDN 都让 FL2VA conditioning rows 保持 dense；VDN 还会把 `V_t` 内部的第 0 帧和最后一帧 latent 当作 dense query 和 exact K/V。这是 `V_t` 内部的 anchor 例外，不表示输入首尾帧 `V` 进入稀疏域，也不改变 target-video-only 的总体定义。

### Veda FL2VA 特例：visual-to-visual dual-pool

Veda 支持 FL2VA，但其默认范围不同于上述 target-video-only 约定。Veda 将 FL2VA 首尾帧的 conditioning rows 视为 reference visual `V`，与目标视频一起放入稀疏视觉象限：

```text
(V + V_t) -> (V + V_t) sparse
```

因此 `V -> V`、`V -> V_t`、`V_t -> V` 和 `V_t -> V_t` 四个方向都稀疏；不仅 reference K/V 会被选择，reference query 也走稀疏路径。reference 与 target-video 使用两个独立的 K/V 池：

| Query \ K/V | `T + A_t` | `V` | `V_t` |
|---|---:|---:|---:|
| `T + A_t` | **dense** | **dense** | **dense** |
| `V` | **dense** | **sparse** | **sparse** |
| `V_t` | **dense** | **sparse** | **sparse** |

```text
(V + V_t) -> [V]K_V + [V_t]K_Vt
```

矩阵中的两个 **sparse** K/V 列对应两个独立池，而不是一个共享池。当前 T2VA/FL2VA predictor 的单一 `keep_ratio` 会分别应用于两个池。将 `reference_sparsity` 设为 `0%` 时，reference 不再 tile 化，而是恢复为双向 dense/global；此时 Veda 才退化为 target-video-only。

## Ref2VA

Ref2VA 没有唯一通用的稀疏范围。当前常见的约定至少有以下三类。

### 1. Target-video-only

可以直接沿用 FL2VA 约定：

```text
V_t -> V_t sparse
```

此时所有 reference visual `V` 都保持 dense query 和 exact K/V。优点是参考信息不会被压缩；缺点是长参考视频的计算量不会被稀疏化。

| Query \ K/V | `T + V + A + A_t` | `V_t` |
|---|---:|---:|
| `T + V + A + A_t` | **dense** | **dense** |
| `V_t` | **dense** | **sparse** |

Spark 的 `sparse_scope=target-video` 使用该约定。

### 2. Veda R2VA：visual-to-visual dual-pool

Veda R2VA 与上述 Veda FL2VA 特例采用相同的范围：将 reference visual 与目标视频共同放入稀疏视觉象限：

```text
(V + V_t) -> (V + V_t) sparse
```

这里 `V` 与 `V_t` 之间的四个方向都稀疏：

| Query \ K/V | `T + A + A_t` | `V` | `V_t` |
|---|---:|---:|---:|
| `T + A + A_t` | **dense** | **dense** | **dense** |
| `V` | **dense** | **sparse** | **sparse** |
| `V_t` | **dense** | **sparse** | **sparse** |

但 K/V 不是放在一个共享 Top-K 中竞争。每个视觉 query 分别从两个 K/V 池选择：

```text
(V + V_t) -> [V]K_V + [V_t]K_Vt
```

当前发布的 H3 R2VA Preview 使用固定 tile 预算：

```text
(V + V_t) -> [V]32 + [V_t]32
```

即每个视觉 query tile、每个 head 最多选择 32 个 reference visual key tiles 和 32 个 target-video key tiles。这里是 32 个 **tile**，不是 32 个 token；每个 tile 有 128 个 token。两个池的额度相互独立，不从一个总预算中按 token 数量瓜分，因此 `V` 与 `V_t` 的实际保留率可以不同。

`T + A + A_t` 在 Veda 中是 global：其 query 行 dense，其 K/V 对所有视觉 query 无条件可见。

### 3. comfy-kitchen Sol：sparse query + exact prefix K/V

当前 ComfyUI generic Ref2VA profile 的 Sol 范围可写成：

```text
Q_sparse = All - A_t
KV_exact = All - V_t
```

忽略跨角色边界块后，这表示：

- `A_t` 是 dense query。
- `T + V + A + V_t` 的 query 行走 Sol sparse/approximate 路径。
- `T + V + A + A_t` 全部作为 exact K/V 保留。
- `V_t` 是被路由或近似的 K/V 区域。

因此，“reference token 是 exact K/V”不等于“reference token 是 dense query”。Sol 的文本、参考视觉和参考音频 query 可以走稀疏路径，同时这些角色对应的 K/V 仍被所有 sparse query 精确保留。

| Query \ K/V | `T + V + A + A_t` | `V_t` |
|---|---:|---:|
| `A_t` | **dense** | **dense** |
| `T + V + A + V_t` | **dense** | **sparse** |

第二行虽然是 sparse query，但第一列仍为 **dense**：这正是 exact prefix K/V 与 sparse query 可以同时成立的直观表示。

实际 kernel 以 block 为单位；跨越角色边界的 block 以及实现强制保留的 block 会形成少量 exact 例外。

## Spark `sparse_scope` 与上述约定

当前 Spark 节点提供两个范围值：

| `sparse_scope` | 稀疏域 | 保持 dense 的 reference visual |
|---|---|---|
| `target-video` | 仅 `V_t` | 所有 `V` |
| `joint-video` | 多时间位置的参考视频视觉 token 与 `V_t` 组成一个共同稀疏域 | 静态参考图以及其他未纳入的 `V` |

用 `V_video` 表示纳入 `joint-video` 的多时间位置参考视频，用 `V_static` 表示保持 dense 的静态参考图和其他 reference visual。两种 scope 可直观写成：

**`target-video`**

| Query \ K/V | `T + V + A + A_t` | `V_t` |
|---|---:|---:|
| `T + V + A + A_t` | **dense** | **dense** |
| `V_t` | **dense** | **sparse** |

**`joint-video`**

| Query \ K/V | `T + V_static + A + A_t` | `V_video + V_t` |
|---|---:|---:|
| `T + V_static + A + A_t` | **dense** | **dense** |
| `V_video + V_t` | **dense** | **sparse** |

`joint-video` 表示这些 token 进入同一个 Spark 稀疏域，不表示采用 Veda 的双池预算。当前 Spark 会把被选中的参考视频与目标视频组成一个稀疏视觉前缀；Veda 则明确使用 `[V]K_V + [V_t]K_Vt` 两个独立 K/V 池。没有多时间位置参考视频时，Spark 的 `joint-video` 自动退化为与 `target-video` 相同的路径。

### Ref2VA 任务加速解决方案

Spark-H3 v0.1.3 默认使用 `sparse_scope=target-video`，因此 Ref2VA 的参考视频仍以 dense 方式参与计算，整体加速比通常低于参考帧较少的 FL2VA。这是默认稀疏范围覆盖较小所致，并非 Ref2VA 的稀疏算子变慢。

Ref2VA 用户可以安装 [Spark-H3 v0.1.4](https://github.com/zechengtang/Spark-H3/releases/tag/comfyui-v0.1.4)，并将 `sparse_scope` 设置为 `joint-video`。这样，多时间位置的参考视频视觉 token 会与目标视频共同进入 Spark 稀疏域，从而减少参考视频保持 dense 带来的额外开销。

独立插件 [VPack v0.1.0](https://github.com/zechengtang/VPack) 用于压缩 Ref2VA 任务中的参考视频，推荐默认档位为 `2/5`；可按需与 Spark-H3 一起使用。

## 对照摘要

| 约定 | Sparse query | Sparse/routed K/V | Dense query | Dense K/V |
|---|---|---|---|---|
| FL2VA / target-video-only | `V_t` | `V_t` | `T + V + A_t` | `T + V + A_t` |
| Veda FL2VA（默认） | `V + V_t` | `V`、`V_t` 两个独立池 | `T + A_t` | `T + A_t` |
| Veda R2VA | `V + V_t` | `V`、`V_t` 两个独立池 | `T + A + A_t` | `T + A + A_t` |
| Sol generic Ref2VA | `T + V + A + V_t` | `V_t` | `A_t` | `T + V + A + A_t` |
