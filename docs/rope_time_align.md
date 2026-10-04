# Ref2VA 视频重采样与 RoPE 时间对齐

## 背景

Ref2VA 的参考视频帧和对应的 RoPE 时间坐标必须描述同一组真实媒体时间。当前 Diffusers 的视频重采样采用输出槽位舍入：

```python
scale = target_fps / source_fps
slots = floor(arange(num_source_frames) * scale + 0.5)
```

每个源帧按相邻 `slots` 的差值保留、丢弃或重复。该规则在不同整数降采样比例下会产生不同的采样相位：

```text
24 -> 12 fps: 0, 2, 4, 6, ...
24 ->  8 fps: 1, 4, 7, 10, ...
```

因此，旧的 24→8 fps 路径虽然使用了正确的 `time_scale=3`，首帧实际时间却是 `1/24` 秒，而不是零时刻。如果 RoPE 仍从零开始，参考图像内容与时间坐标会相差 `1/24` 秒；按 MiniMax-H3 每秒 40 个 RoPE 时间单位计算，对应 `5/3` 个单位。

## 本地实现

仓库内实现位于 [`scripts/ref2va_zero_origin_resample.py`](../scripts/ref2va_zero_origin_resample.py)，测试位于 [`tests/test_ref2va_zero_origin_resample.py`](../tests/test_ref2va_zero_origin_resample.py)。本地实现保留 Diffusers 的以下行为：

- 输入视频格式转换为 `uint8 THWC`；
- 按流结束时间舍入目标帧数；
- 按请求帧数截断；
- 时间重采样后逐帧执行空间 LANCZOS resize；
- 不进行时间方向的像素插值或光流插帧。

仅采样时间网格的起点策略可配置。

### `zero`：默认模式

目标时间网格从媒体零时刻开始。目标帧 `k` 直接选择包含时间 `k / target_fps` 的源帧：

```python
count = floor(num_source_frames * target_fps / source_fps + 0.5)
indices = floor(arange(count) * source_fps / target_fps)
frames = frames[indices]
```

示例：

```text
24 -> 8 fps:  0, 3, 6, 9, ...
24 -> 12 fps: 0, 2, 4, 6, ...
8 -> 24 fps:  0, 0, 0, 1, 1, 1, ...
```

降采样本质上是 index select；升采样通过重复索引保持上一源帧。默认模式不引入额外的时间插值开销。

### `legacy_round`：兼容模式

该模式逐项复现 Diffusers 当前的槽位舍入行为，用于读取旧缓存、复现实验或进行兼容性比较。对于 24→8 fps，它产生 `1,4,7,...`；对于 24→12 fps，它仍产生 `0,2,4,...`。

不要根据目标 fps 猜测采样起点。需要兼容旧路径时，应从实际首个采样索引计算：

```python
time_origin_seconds = indices[0] / source_fps
```

## RoPE 时间参数

帧选择和 RoPE 坐标变换是两个独立步骤。调用方必须显式提供：

- `time_scale`：参考 conditioning 的时间相对模型 24 fps 时钟的伸缩比例，即 `model_fps / reference_fps`；例如 8 fps conditioning 使用 `24 / 8 = 3`。这里不能使用任意源文件 fps；30 fps源文件重采样为8fps后仍应使用 `24 / 8`；
- `time_origin_seconds`：首个采样帧在源媒体中的真实时间，零起点模式为 `0.0`，旧24→8 fps路径为 `1/24`；
- `rope_units_per_second`：默认 `40.0`。

设参考视频块的原始 RoPE 起点为 `origin`，原坐标为 `t`，变换为：

```text
t' = origin + (t - origin) * time_scale
            + time_origin_seconds * rope_units_per_second
```

空间坐标 H/W 不变。参考视频自带音频仍从媒体零时刻开始，因此音频坐标不随视频 phase 平移。

### 后续块起点

视频块变换后，后续参考块和目标生成块不能继续使用旧的视频结束位置。设：

```text
old_block_span = max(audio_span, video_span)
new_video_span = time_origin * rope_units_per_second
                 + video_span * time_scale
new_block_span = max(audio_span, new_video_span)
```

后续块的时间坐标统一增加：

```text
new_block_span - old_block_span
```

这样后续块始终从修正后的音频、视频结束位置中的较大值开始。若自带音频比修正后的视频更长，该偏移可以为零。

## 推荐配置

新生成的 Ref2VA conditioning 默认使用：

```python
start_mode = "zero"
time_scale = model_fps / reference_fps
time_origin_seconds = 0.0
```

对于本项目的 24→8 fps 条件视频，具体为：

```python
start_mode = "zero"
time_scale = 3.0
time_origin_seconds = 0.0
```

只有在复现旧结果时才使用：

```python
start_mode = "legacy_round"
time_scale = 3.0
time_origin_seconds = 1 / 24
```

采样索引、参考 fps、时间起点、视觉 LLM 时间标签和 RoPE 参数应一起写入运行配置，避免 conditioning 内容与时间元数据再次失配。

## 效率

时间重采样只生成一个整数索引数组并进行一次 NumPy index select。对于当前约39帧的Ref2VA条件视频，索引数组和帧副本开销很小。主要耗时仍来自空间LANCZOS缩放、VAE编码和模型推理，因此无需为时间重采样开发CUDA算子。

推荐执行顺序为：

```text
解码源视频 -> 时间index select -> 帧数截断/VAE合法长度裁剪
           -> 空间LANCZOS缩放 -> VAE编码 -> RoPE时间变换
```

## 当前验证

本地测试覆盖：

- 24→8 fps默认零起点；
- 24→12 fps零偏移回归；
- `legacy_round`与Diffusers现有索引一致；
- 低帧率到高帧率的整帧重复；
- 同帧率输入逐帧保持不变；
- `time_scale`与`time_origin_seconds`独立生效；
- H/W坐标和参考音频坐标保持不变；
- 后续块起点按修正后的音视频结束位置更新。

当前测试结果为 `7 passed`。该模块目前是仓库内独立实现，尚未替换运行中的 Diffusers 路径；正式接入时应显式调用本地 normalizer 和 RoPE 时间变换函数。
