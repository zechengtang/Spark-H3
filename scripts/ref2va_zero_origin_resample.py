#!/usr/bin/env python3
"""Repository-local MiniMax-H3 video resampling and RoPE time helpers.

The video normalizer is a drop-in alternative to
``MiniMaxH3Ref2VASetupStep._normalize_video_condition``. It retains the
upstream input conversion, output-length rounding, truncation and LANCZOS
resize policy. Its default ``zero`` mode anchors the target frame grid at
media time zero; ``legacy_round`` reproduces Diffusers' current slot-rounding
behavior for compatibility.

The upstream implementation is Apache-2.0 licensed and lives in
``diffusers/modular_pipelines/minimax_h3/before_encoder.py``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from PIL import Image

from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size


ResampleStartMode = Literal["zero", "legacy_round"]
ROPE_TIME_UNITS_PER_SECOND = 40.0


def resample_frame_indices(
    num_source_frames: int,
    source_fps: float,
    target_fps: float,
    start_mode: ResampleStartMode = "zero",
) -> np.ndarray:
    """Return source indices on either a zero-origin or legacy rounded grid."""
    if num_source_frames < 0:
        raise ValueError(f"num_source_frames must be non-negative, got {num_source_frames}.")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(
            f"source_fps and target_fps must be positive, got {source_fps} and {target_fps}."
        )
    if start_mode not in ("zero", "legacy_round"):
        raise ValueError(f"Unknown resampling start mode {start_mode!r}.")

    count = math.floor(num_source_frames * target_fps / source_fps + 0.5)
    if count == 0:
        return np.empty(0, dtype=np.int64)

    if start_mode == "legacy_round":
        # This is Diffusers' current implementation expressed as indices:
        # every source frame is repeated by the change in its rounded output
        # slot. For 24 -> 8 it retains 1,4,7,...; for 24 -> 12 it retains
        # 0,2,4,... due to the half-up boundary.
        scale = target_fps / source_fps
        slots = np.floor(np.arange(num_source_frames) * scale + 0.5).astype(np.int64)
        repeats = np.diff(slots, append=count)
        return np.repeat(np.arange(num_source_frames, dtype=np.int64), repeats)

    indices = np.floor(
        np.arange(count, dtype=np.float64) * source_fps / target_fps + 1e-9
    ).astype(np.int64)
    return np.minimum(indices, num_source_frames - 1)


def zero_origin_frame_indices(
    num_source_frames: int,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    """Return source indices for a target grid beginning at exactly ``t=0``.

    The output count deliberately matches Diffusers' current stream-end
    rounding.  Each target timestamp ``k / target_fps`` selects the source
    frame whose interval contains it, namely
    ``floor(k * source_fps / target_fps)``.
    """
    return resample_frame_indices(num_source_frames, source_fps, target_fps, "zero")


def sampling_time_origin_seconds(indices: np.ndarray, source_fps: float) -> float:
    """Return the media timestamp represented by the first sampled frame."""
    indices = np.asarray(indices)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("indices must be a non-empty one-dimensional array.")
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}.")
    return float(indices[0]) / float(source_fps)


def normalize_video_condition(
    frames,
    fps: float,
    num_frames: int,
    canvas_multiple: int,
    canvas_short_edge: int,
    canvas_max_pixels: int,
    target_fps: float,
    start_mode: ResampleStartMode = "zero",
) -> np.ndarray:
    """Normalize a MiniMax-H3 video reference on a selectable time grid.

    The signature intentionally matches Diffusers' private normalization
    helper for its first seven arguments. The repository-local eighth argument
    selects zero-origin behavior by default or exact upstream compatibility.
    """
    if isinstance(frames, list):
        frames = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    if isinstance(frames, torch.Tensor):
        frames = frames.movedim(-3, -1).cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            "A reference video must be `(num_frames, height, width, 3)` RGB frames, "
            f"got {tuple(frames.shape)}."
        )
    if fps <= 0 or target_fps <= 0:
        raise ValueError(f"Video frame rates must be positive, got {fps} and {target_fps}.")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")

    if fps != target_fps:
        indices = resample_frame_indices(frames.shape[0], fps, target_fps, start_mode)
        frames = frames[indices]

    frames = frames[:num_frames]
    if len(frames) == 0:
        raise ValueError("Video resampling produced no frames.")

    height, width = resolve_canvas_size(
        frames.shape[2],
        frames.shape[1],
        canvas_multiple,
        canvas_short_edge,
        canvas_max_pixels,
    )
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
    )


def normalize_video_condition_zero_origin(
    frames,
    fps: float,
    num_frames: int,
    canvas_multiple: int,
    canvas_short_edge: int,
    canvas_max_pixels: int,
    target_fps: float,
) -> np.ndarray:
    """Backward-compatible convenience wrapper fixed to zero-origin mode."""
    return normalize_video_condition(
        frames,
        fps,
        num_frames,
        canvas_multiple,
        canvas_short_edge,
        canvas_max_pixels,
        target_fps,
        start_mode="zero",
    )


def transform_reference_video_rope_time(
    position_ids: torch.Tensor,
    condition_video_indices: torch.Tensor,
    *,
    video_span_rope_units: float,
    time_scale: float = 1.0,
    time_origin_seconds: float = 0.0,
    audio_span_rope_units: float = 0.0,
    following_start_index: int | None = None,
    rope_units_per_second: float = ROPE_TIME_UNITS_PER_SECOND,
) -> torch.Tensor:
    """Apply explicit scale and media-time origin to one reference video.

    Audio coordinates are intentionally untouched. If rows follow the
    reference block, ``following_start_index`` shifts their time coordinates
    by the change in ``max(audio_end, video_end)`` so subsequent blocks begin
    after the corrected reference end.
    """
    if position_ids.ndim != 2 or position_ids.shape[1] != 3:
        raise ValueError(f"position_ids must have shape (rows, 3), got {tuple(position_ids.shape)}.")
    if condition_video_indices.ndim != 1 or condition_video_indices.numel() == 0:
        raise ValueError("condition_video_indices must be a non-empty one-dimensional tensor.")
    values = (video_span_rope_units, time_scale, time_origin_seconds, audio_span_rope_units, rope_units_per_second)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("RoPE time parameters must be finite.")
    if video_span_rope_units < 0 or audio_span_rope_units < 0:
        raise ValueError("Video and audio spans must be non-negative.")
    if time_scale <= 0 or rope_units_per_second <= 0:
        raise ValueError("time_scale and rope_units_per_second must be positive.")
    if following_start_index is not None and not 0 <= following_start_index <= len(position_ids):
        raise ValueError(f"following_start_index is out of range: {following_start_index}.")

    result = position_ids.clone()
    origin = result[condition_video_indices[0], 0].clone()
    phase = float(time_origin_seconds) * float(rope_units_per_second)
    old_time = result[condition_video_indices, 0]
    result[condition_video_indices, 0] = origin + (old_time - origin) * time_scale + phase

    old_block_span = max(float(audio_span_rope_units), float(video_span_rope_units))
    new_video_span = phase + float(video_span_rope_units) * float(time_scale)
    new_block_span = max(float(audio_span_rope_units), new_video_span)
    if following_start_index is not None:
        result[following_start_index:, 0] += new_block_span - old_block_span
    return result


__all__ = [
    "ROPE_TIME_UNITS_PER_SECOND",
    "ResampleStartMode",
    "normalize_video_condition",
    "normalize_video_condition_zero_origin",
    "resample_frame_indices",
    "sampling_time_origin_seconds",
    "transform_reference_video_rope_time",
    "zero_origin_frame_indices",
]
