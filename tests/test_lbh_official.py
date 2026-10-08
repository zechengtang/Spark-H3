import torch
import torch.nn.functional as F
from argparse import Namespace

from diffusers import MiniMaxH3Blocks
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

from h3_lbh import H3LBHOfficialConfig, patch_lbh_official_into_pipeline
from h3_lbh.resolution import megapixel_canvas, official_canvas_pair
from h3_lbh.lora import _comfy_fc1_up_to_diffusers
from h3_lbh.models import LBHMiniMaxH3LatentUpscaler
from h3_lbh.workflow import (
    MiniMaxH3LBHOfficialDenoiseStep,
    comfy_simple_sigmas,
    map_shifted_sigmas,
)
from scripts.run_lbh_official_diffusers import resolve_canvases


def test_official_schedule_is_four_low_three_high():
    config = H3LBHOfficialConfig()
    assert config.low_grid_points == 8
    assert config.low_evaluations == 4
    assert config.high_sigmas == (0.9035, 0.6316, 0.3158, 0.0)
    assert len(config.high_sigmas) - 1 == 3
    assert config.highres_group_offload_blocks is None


def test_published_resolution_selector_canvases_are_independently_aligned():
    assert megapixel_canvas(0.2, 16 / 9, 32) == (352, 608)
    assert megapixel_canvas(1.0, 16 / 9, 32) == (768, 1376)
    low, high = official_canvas_pair()
    assert high[0] / low[0] != high[1] / low[1]


def test_config_accepts_axis_specific_grid_aligned_scale():
    config = H3LBHOfficialConfig(lowres_scale=(352 / 768, 608 / 1376))
    assert isinstance(config.lowres_scale, tuple)


def test_runner_defaults_to_h3_official_768p_with_lbh_point_two_mp_prefix():
    args = Namespace(
        height=None, width=None, target_megapixels=0.98,
        low_height=None, low_width=None, lowres_scale=None,
        low_megapixels=0.2, aspect_ratio=16 / 9, align=32,
    )
    assert resolve_canvases(args) == ((352, 608), (768, 1344))


def test_runner_can_still_reproduce_lbh_one_mp_target():
    args = Namespace(
        height=None, width=None, target_megapixels=1.0,
        low_height=None, low_width=None, lowres_scale=None,
        low_megapixels=0.2, aspect_ratio=16 / 9, align=32,
    )
    assert resolve_canvases(args) == ((352, 608), (768, 1376))


def test_audio_sigma_mapping_preserves_base_clock():
    video = [0.9035, 0.6316, 0.3158, 0.0]
    audio = map_shifted_sigmas(video, 12.0, 3.0)
    assert audio[-1] == 0.0
    assert all(right < left for left, right in zip(audio, audio[1:]))
    video_tensor = torch.tensor(video, dtype=torch.float64)
    base = video_tensor / (12.0 + video_tensor * (1.0 - 12.0))
    expected = 3.0 * base / (1.0 + 2.0 * base)
    assert torch.allclose(torch.tensor(audio, dtype=torch.float64), expected)


def test_manual_sigmas_are_verbatim_in_minimax_scheduler():
    from diffusers import MiniMaxH3Scheduler

    expected = [0.9035, 0.6316, 0.3158, 0.0]
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(sigmas=expected)
    assert torch.allclose(scheduler.sigmas, torch.tensor(expected), atol=1e-6)
    assert scheduler.timesteps.shape[0] == 3


def test_low_schedule_matches_comfy_simple_eight_step_clock():
    from diffusers import MiniMaxH3Scheduler

    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(sigmas=comfy_simple_sigmas(8, 12.0))
    expected = torch.tensor([1.0, 0.9882353, 0.9729730, 0.9523810])
    assert torch.allclose(scheduler.sigmas[:4], expected, atol=1e-6)


def test_comfy_fc1_swiglu_permutation_is_reversed():
    # Diffusers [value; gate] -> converter's Comfy [gate; value] -> loader.
    diffusers_up = torch.arange(24).reshape(6, 4)
    value, gate = diffusers_up.chunk(2, dim=0)
    comfy_up = torch.cat((gate, value), dim=0)
    assert torch.equal(_comfy_fc1_up_to_diffusers(comfy_up), diffusers_up)


def test_lbh_wrapper_applies_official_channel_normalization(monkeypatch):
    model = LBHMiniMaxH3LatentUpscaler(channels=32, in_blocks=0, out_blocks=0)

    def doubled_normalized_input(hidden_states, scale, target_size):
        return 2 * F.interpolate(hidden_states, size=target_size, mode="trilinear", align_corners=False)

    monkeypatch.setattr(model, "_forward_segment", doubled_normalized_input)
    latents = torch.randn(1, 24, 2, 4, 6)
    actual = model(latents, (8, 12))
    mean = model.latents_mean
    expected = 2 * F.interpolate(latents, size=(2, 8, 12), mode="trilinear", align_corners=False) - mean
    assert torch.allclose(actual, expected, atol=1e-5)


def test_lbh_wrapper_uses_mean_effective_scale_for_misaligned_target(monkeypatch):
    model = LBHMiniMaxH3LatentUpscaler(channels=32, in_blocks=0, out_blocks=0)
    seen = {}

    def capture(hidden_states, scale, target_size):
        seen["scale"] = scale
        return F.interpolate(hidden_states, size=target_size, mode="trilinear", align_corners=False)

    monkeypatch.setattr(model, "_forward_segment", capture)
    latents = torch.randn(1, 24, 2, 22, 38)
    actual = model(latents, (48, 86))
    assert actual.shape[-2:] == (48, 86)
    assert seen["scale"] == ((48 / 22) + (86 / 38)) / 2


def test_pipeline_patch_installs_standalone_lbh_loop():
    lifter = object()
    pipe = MiniMaxH3ModularPipeline(blocks=MiniMaxH3Blocks().get_workflow("t2va"))
    patch_lbh_official_into_pipeline(pipe, lifter)
    block = pipe._blocks.sub_blocks["denoise.denoise"]
    assert isinstance(block, MiniMaxH3LBHOfficialDenoiseStep)
    assert block.latent_lifter is lifter
