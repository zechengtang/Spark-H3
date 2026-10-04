"""Regression tests for the local zero-origin Ref2VA video resampler."""

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ref2va_zero_origin_resample import (  # noqa: E402
    normalize_video_condition,
    normalize_video_condition_zero_origin,
    resample_frame_indices,
    sampling_time_origin_seconds,
    transform_reference_video_rope_time,
    zero_origin_frame_indices,
)
import torch


def marker_video(count=145):
    frames = np.zeros((count, 1, 1, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(count, dtype=np.uint8)
    return frames


def test_integer_downsampling_is_anchored_at_zero():
    assert np.array_equal(zero_origin_frame_indices(145, 24, 8), np.arange(0, 144, 3))
    assert np.array_equal(zero_origin_frame_indices(145, 24, 12), np.arange(0, 145, 2))


def test_legacy_round_mode_reproduces_diffusers_phase():
    eight = resample_frame_indices(145, 24, 8, "legacy_round")
    twelve = resample_frame_indices(145, 24, 12, "legacy_round")
    assert np.array_equal(eight, np.arange(1, 145, 3))
    assert np.array_equal(twelve, np.arange(0, 145, 2))
    assert sampling_time_origin_seconds(eight, 24) == 1 / 24
    assert sampling_time_origin_seconds(twelve, 24) == 0


def test_normalizer_has_drop_in_signature_and_zero_origin_output():
    result = normalize_video_condition_zero_origin(marker_video(), 24, 39, 1, 1, 1, 8)
    assert result.shape == (39, 1, 1, 3)
    assert np.array_equal(result[:, 0, 0, 0], np.arange(0, 117, 3))


def test_upsampling_also_starts_at_zero():
    assert np.array_equal(zero_origin_frame_indices(3, 8, 24), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]))


def test_equal_rate_preserves_frames_exactly():
    source = marker_video(5)
    result = normalize_video_condition_zero_origin(source, 24, 5, 1, 1, 1, 24)
    assert np.array_equal(result, source)


def test_normalizer_can_select_legacy_round_mode():
    result = normalize_video_condition(
        marker_video(), 24, 39, 1, 1, 1, 8, start_mode="legacy_round"
    )
    assert np.array_equal(result[:, 0, 0, 0], np.arange(1, 118, 3))


def test_rope_scale_and_time_origin_are_independent_and_shift_following_rows():
    positions = torch.tensor(
        [[10.0, 1.0, 2.0], [10.0, 3.0, 4.0], [12.0, 5.0, 6.0], [12.0, 7.0, 8.0],
         [15.0, 9.0, 10.0], [20.0, 11.0, 12.0]],
        dtype=torch.float64,
    )
    condition = torch.tensor([0, 1, 2, 3])
    actual = transform_reference_video_rope_time(
        positions,
        condition,
        video_span_rope_units=3.0,
        time_scale=3.0,
        time_origin_seconds=1 / 24,
        audio_span_rope_units=5.0,
        following_start_index=5,
    )
    phase = 5 / 3
    expected = positions.new_tensor([10 + phase, 10 + phase, 16 + phase, 16 + phase])
    assert torch.allclose(actual[condition, 0], expected)
    # Old block span=max(5,3)=5; corrected span=max(5, 3*3+5/3)=10+2/3.
    assert actual[5, 0] == positions[5, 0] + (10 + 2 / 3 - 5)
    assert torch.equal(actual[:, 1:], positions[:, 1:])
    # The attached-audio row before following_start_index is not shifted.
    assert actual[4, 0] == positions[4, 0]
