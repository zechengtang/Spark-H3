"""LBH's official two-pass MiniMax-H3 denoising schedule for Diffusers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import logging
from typing import TypeAlias

import torch
import torch.nn.functional as F

from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseStep
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam
from diffusers.utils.torch_utils import randn_tensor

log = logging.getLogger(__name__)

SpatialScale: TypeAlias = float | tuple[float, float]


def _unpatchify_video_rows(rows, channels, frames, height, width, patch_size):
    patch_t, patch_h, patch_w = patch_size
    expected = frames // patch_t * (height // patch_h) * (width // patch_w)
    if rows.ndim != 2 or rows.shape[0] != expected:
        raise ValueError(
            f"cannot unpack {tuple(rows.shape)} as a ({frames}, {height}, {width}) H3 grid; "
            f"expected {expected} rows"
        )
    values = rows.reshape(
        1, frames // patch_t, height // patch_h, width // patch_w,
        channels, patch_t, patch_h, patch_w,
    )
    return values.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        1, channels, frames, height, width
    ).contiguous()


def _resize_keyframe_rows(rows, count, channels, full_hw, low_hw, patch_size):
    if count == 0:
        return rows[:0]
    full_h, full_w = full_hw
    low_h, low_w = low_hw
    rows_per_keyframe = (full_h // patch_size[1]) * (full_w // patch_size[2])
    if rows.shape[0] != count * rows_per_keyframe:
        raise ValueError("FL2VA conditioning rows do not match the target-resolution keyframe grid")
    resized = []
    for chunk in rows.split(rows_per_keyframe):
        latent = _unpatchify_video_rows(chunk, channels, 1, full_h, full_w, patch_size)
        low = F.interpolate(
            latent.float(), size=(1, low_h, low_w), mode="trilinear", align_corners=False
        )
        low += latent.float().mean((-2, -1), keepdim=True) - low.mean((-2, -1), keepdim=True)
        resized.append(patchify_video_latents(low, patch_size))
    return torch.cat(resized)


@dataclass(frozen=True)
class H3LBHOfficialConfig:
    """The active settings in LBH's published I2V example workflow."""

    # A scalar retains the old proportional mode.  A (height, width) pair is
    # useful when the two independently grid-aligned ResolutionSelector
    # canvases do not have exactly the same ratio (the published 0.2 MP ->
    # 1.0 MP workflow is one such case).
    lowres_scale: SpatialScale = 0.5
    low_grid_points: int = 8  # Comfy BasicScheduler(simple, steps=8): 8 NFE + terminal sigma.
    low_evaluations: int = 4
    high_sigmas: tuple[float, ...] = (0.9035, 0.6316, 0.3158, 0.0)
    highres_group_offload_blocks: int | None = None
    low_only: bool = False
    lift_only: bool = False
    use_spark: bool = False
    spark_overrides: dict = field(default_factory=lambda: {
        "warmup_percent": 0.0,
        "sol_route_topk_ratio": 0.1,
        "sol_dense_layers": 1,
        "landmark_tree_v2_layout_reuse": "q_from_k",
    })

    def __post_init__(self):
        scales = (self.lowres_scale, self.lowres_scale) if isinstance(self.lowres_scale, (int, float)) else self.lowres_scale
        if len(scales) != 2 or any(not 0.25 <= float(value) < 1.0 for value in scales):
            raise ValueError("lowres_scale must be a scalar or (height, width) pair in [0.25, 1)")
        if self.low_grid_points < 2 or not 1 <= self.low_evaluations < self.low_grid_points:
            raise ValueError("invalid low-resolution schedule")
        sigmas = tuple(float(value) for value in self.high_sigmas)
        if len(sigmas) < 2 or sigmas[-1] != 0.0 or any(b >= a for a, b in zip(sigmas, sigmas[1:])):
            raise ValueError("high_sigmas must be strictly decreasing and terminate at zero")
        if self.highres_group_offload_blocks is not None and self.highres_group_offload_blocks < 1:
            raise ValueError("highres_group_offload_blocks must be positive or None")
        if self.low_only and self.lift_only:
            raise ValueError("low_only and lift_only are mutually exclusive")


def map_shifted_sigmas(sigmas, source_shift: float, target_shift: float) -> list[float]:
    """Map already-shifted flow sigmas from one H3 modality clock to another."""
    values = torch.as_tensor(sigmas, dtype=torch.float64)
    base = values / (source_shift + values * (1.0 - source_shift))
    mapped = target_shift * base / (1.0 + (target_shift - 1.0) * base)
    mapped[-1] = 0.0
    return mapped.tolist()


def comfy_simple_sigmas(steps: int, shift: float) -> list[float]:
    """Shifted sigma grid selected by Comfy's ``simple`` scheduler.

    MiniMax-H3's Comfy model has a 10,000-entry FlowMatch clock.  Eight
    evenly indexed samples are therefore exactly 1, 7/8, ..., 1/8, followed
    by zero. MiniMaxH3Scheduler accepts this fully formed grid verbatim.
    """
    if steps < 1 or 10_000 % steps:
        raise ValueError("official Comfy simple emulation requires a positive divisor of 10000")
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return (shift * base / (1.0 + (shift - 1.0) * base)).tolist()


class MiniMaxH3LBHOfficialDenoiseStep(MiniMaxH3DenoiseStep):
    """Four low-resolution evaluations, LBH lift, then three fresh high-resolution evaluations."""

    def __init__(self, config: H3LBHOfficialConfig, latent_lifter):
        self.lbh_config = config
        self.latent_lifter = latent_lifter
        self._highres_group_offload_enabled = False
        super().__init__()

    @property
    def loop_inputs(self):
        return super().loop_inputs + [
            InputParam.template("generator"),
            InputParam("latent_height", type_hint=int, required=True),
            InputParam("latent_width", type_hint=int, required=True),
            InputParam("num_latent_frames", type_hint=int, required=True),
            InputParam("num_audio_latents", type_hint=int, required=True),
            InputParam("text_token_tags", type_hint=torch.Tensor, required=True),
            InputParam("keyframe_anchors", type_hint=tuple, default=()),
        ]

    @staticmethod
    def _set_layout(block_state, layout):
        names = (
            "position_ids", "token_tags", "video_indices", "audio_indices", "text_indices",
            "num_condition_video_rows", "num_condition_audio_rows",
        )
        for name, value in zip(names, layout):
            setattr(block_state, name, value)
        for name, value in zip(names[:5], layout[:5]):
            block_state.denoiser_input_fields[name] = value

    @staticmethod
    def _row_plan(components, block_state):
        return [
            tuple(
                tensor.to(components._execution_device)
                for tensor in MiniMaxH3SetTimestepsStep.build_row_timesteps(
                    block_state.video_indices,
                    block_state.audio_indices,
                    block_state.num_condition_video_rows,
                    block_state.num_condition_audio_rows,
                    block_state.text_indices.numel(),
                    float(video_t),
                    float(audio_t),
                    max(float(video_t), components.keyframe_noise_aug),
                    1.0,
                )
            )
            for video_t, audio_t in zip(block_state.timesteps, block_state.audio_timesteps)
        ]

    @staticmethod
    def _stage_generator(generator):
        if isinstance(generator, list):
            if len(generator) != 1:
                raise ValueError("the LBH workflow currently supports batch size one")
            generator = generator[0]
        stage = torch.Generator(device="cpu")
        return stage.manual_seed(generator.initial_seed())

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState):
        block_state = self.get_block_state(state)
        cfg = self.lbh_config
        device = components._execution_device
        patch_size = components.patch_size
        channels = components.vae_latent_channels
        full_h, full_w = block_state.latent_height, block_state.latent_width
        patch_h, patch_w = patch_size[1:]
        if isinstance(cfg.lowres_scale, (int, float)):
            scale_h = scale_w = float(cfg.lowres_scale)
        else:
            scale_h, scale_w = map(float, cfg.lowres_scale)
        low_h = max(patch_h, round(full_h * scale_h / patch_h) * patch_h)
        low_w = max(patch_w, round(full_w * scale_w / patch_w) * patch_w)

        full_layout = tuple(
            getattr(block_state, name)
            for name in (
                "position_ids", "token_tags", "video_indices", "audio_indices", "text_indices",
                "num_condition_video_rows", "num_condition_audio_rows",
            )
        )
        full_condition_count = block_state.num_condition_video_rows
        full_condition_rows = block_state.latents[:full_condition_count]
        low_layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            block_state.text_token_tags,
            block_state.num_latent_frames,
            low_h,
            low_w,
            block_state.num_audio_latents,
            patch_size,
            components.audio_channels,
            components.audio_tag,
            components.video_tag,
            block_state.keyframe_anchors,
        )
        low_layout = tuple(value.to(device) if torch.is_tensor(value) else value for value in low_layout)
        low_condition_count = low_layout[-2]
        low_condition_rows = _resize_keyframe_rows(
            full_condition_rows,
            len(block_state.keyframe_anchors),
            channels,
            (full_h, full_w),
            (low_h, low_w),
            patch_size,
        )

        # RandomNoise(seed=fixed) is evaluated independently by each official
        # sampler. Re-seeding each stage reproduces that behavior and avoids the
        # unused full-resolution preparation draw changing the public seed.
        low_generator = self._stage_generator(block_state.generator)
        low_noise = randn_tensor(
            (1, channels, block_state.num_latent_frames, low_h, low_w),
            generator=low_generator, device=device, dtype=torch.float32,
        )
        audio_noise = randn_tensor(
            (block_state.num_audio_latents * components.audio_channels, components.audio_latent_channels),
            generator=low_generator, device=device, dtype=torch.float32,
        )
        block_state.latents = torch.cat((low_condition_rows, patchify_video_latents(low_noise, patch_size)))
        block_state.audio_latents = audio_noise
        self._set_layout(block_state, low_layout)

        components.scheduler.set_timesteps(
            sigmas=comfy_simple_sigmas(cfg.low_grid_points, components.scheduler.shift), device=device
        )
        components.audio_scheduler.set_timesteps(
            sigmas=comfy_simple_sigmas(cfg.low_grid_points, components.audio_scheduler.shift), device=device
        )
        block_state.timesteps = components.scheduler.timesteps
        block_state.audio_timesteps = components.audio_scheduler.timesteps
        block_state.row_timestep_plan = self._row_plan(components, block_state)

        denoiser = self.sub_blocks["denoiser"]
        updater = self.sub_blocks["update"]
        high_evaluations = len(cfg.high_sigmas) - 1
        total_evaluations = (
            cfg.low_evaluations if (cfg.low_only or cfg.lift_only) else cfg.low_evaluations + high_evaluations
        )
        with self.progress_bar(total=total_evaluations) as progress_bar:
            for index in range(cfg.low_evaluations):
                t = block_state.timesteps[index]
                components, block_state = denoiser(components, block_state, i=index, t=t)
                if index + 1 < cfg.low_evaluations:
                    components, block_state = updater(components, block_state, i=index, t=t)
                progress_bar.update()

            # SamplerCustomAdvanced.denoised_output: the last model prediction
            # x0, not the still-noisy sampler output at the split boundary.
            video_rows = block_state.latents[low_condition_count:]
            video_velocity = block_state.noise_pred[0, low_condition_count:].to(video_rows)
            # MiniMax-H3 predicts data-ward velocity: x0 = x + sigma*v.
            video_sigma = components.scheduler.sigmas[cfg.low_evaluations - 1].to(video_rows)
            x0_video_rows = video_rows + video_sigma * video_velocity
            x0_video_low = _unpatchify_video_rows(
                x0_video_rows, channels, block_state.num_latent_frames, low_h, low_w, patch_size
            )

            audio_count = block_state.num_condition_audio_rows
            audio_rows = block_state.audio_latents[audio_count:]
            audio_velocity = block_state.audio_noise_pred[0, audio_count:].to(audio_rows)
            audio_sigma = components.audio_scheduler.sigmas[cfg.low_evaluations - 1].to(audio_rows)
            x0_audio_rows = audio_rows + audio_sigma * audio_velocity

            if cfg.low_only:
                # Preserve the packed representation expected by the standard
                # after-denoise block, but replace the current noisy rows with
                # the sampler's clean prediction and expose the low canvas.
                block_state.latents = torch.cat((low_condition_rows, x0_video_rows))
                if audio_count:
                    block_state.audio_latents = torch.cat(
                        (block_state.audio_latents[:audio_count], x0_audio_rows)
                    )
                else:
                    block_state.audio_latents = x0_audio_rows
                block_state.latent_height = low_h
                block_state.latent_width = low_w
                self.set_block_state(state, block_state)
                log.info("LBH diagnostic low-only: %dx%d/%d NFE", low_h, low_w, cfg.low_evaluations)
                return components, state

            self.latent_lifter.to(device=device)
            x0_video_high = self.latent_lifter(x0_video_low, (full_h, full_w)).to(video_rows)
            self.latent_lifter.to(device="cpu")
            torch.cuda.empty_cache()

            if cfg.lift_only:
                block_state.latents = torch.cat(
                    (full_condition_rows, patchify_video_latents(x0_video_high, patch_size))
                )
                if audio_count:
                    block_state.audio_latents = torch.cat(
                        (block_state.audio_latents[:audio_count], x0_audio_rows)
                    )
                else:
                    block_state.audio_latents = x0_audio_rows
                self._set_layout(block_state, full_layout)
                # This diagnostic branch skips the high-resolution phase where
                # group offload is normally installed. Free the transformer so
                # the full-resolution VAE decode has enough device memory.
                components.transformer.to("cpu")
                torch.cuda.empty_cache()
                self.set_block_state(state, block_state)
                log.info("LBH diagnostic lift-only: %dx%d", full_h, full_w)
                return components, state

            high_video_sigmas = list(cfg.high_sigmas)
            high_audio_sigmas = map_shifted_sigmas(
                high_video_sigmas, components.scheduler.shift, components.audio_scheduler.shift
            )
            components.scheduler.set_timesteps(sigmas=high_video_sigmas, device=device)
            components.audio_scheduler.set_timesteps(sigmas=high_audio_sigmas, device=device)
            block_state.timesteps = components.scheduler.timesteps
            block_state.audio_timesteps = components.audio_scheduler.timesteps

            high_generator = self._stage_generator(block_state.generator)
            high_video_noise = randn_tensor(
                x0_video_high.shape, generator=high_generator, device=device, dtype=x0_video_high.dtype
            )
            high_audio_noise = randn_tensor(
                x0_audio_rows.shape, generator=high_generator, device=device, dtype=x0_audio_rows.dtype
            )
            high_video = components.scheduler.scale_noise(
                x0_video_high, block_state.timesteps[0], high_video_noise
            )
            high_audio = components.audio_scheduler.scale_noise(
                x0_audio_rows, block_state.audio_timesteps[0], high_audio_noise
            )
            block_state.latents = torch.cat((full_condition_rows, patchify_video_latents(high_video, patch_size)))
            if audio_count:
                block_state.audio_latents = torch.cat((block_state.audio_latents[:audio_count], high_audio))
            else:
                block_state.audio_latents = high_audio
            self._set_layout(block_state, full_layout)
            block_state.row_timestep_plan = self._row_plan(components, block_state)

            transformer = components.transformer
            if cfg.highres_group_offload_blocks is not None and not self._highres_group_offload_enabled:
                transformer.enable_group_offload(
                    onload_device=device,
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    num_blocks_per_group=cfg.highres_group_offload_blocks,
                    use_stream=False,
                )
                self._highres_group_offload_enabled = True
                torch.cuda.empty_cache()

            context = nullcontext()
            if cfg.use_spark:
                from h3_sparse_attention import install_h3_spark_attn

                context = install_h3_spark_attn(
                    transformer, num_inference_steps=high_evaluations + 1, **dict(cfg.spark_overrides)
                )
            with context:
                for index, t in enumerate(block_state.timesteps):
                    components, block_state = denoiser(components, block_state, i=index, t=t)
                    components, block_state = updater(components, block_state, i=index, t=t)
                    progress_bar.update()

        self.set_block_state(state, block_state)
        log.info(
            "LBH official: low=%dx%d/%d NFE, high=%dx%d/%d NFE, Spark=%s",
            low_h, low_w, cfg.low_evaluations, full_h, full_w, high_evaluations, cfg.use_spark,
        )
        return components, state


def patch_lbh_official_into_pipeline(pipe, latent_lifter, config: H3LBHOfficialConfig | None = None):
    """Install the standalone LBH schedule into a T2VA/FL2VA modular pipeline."""
    if not isinstance(pipe, MiniMaxH3ModularPipeline):
        raise TypeError("expected a MiniMaxH3ModularPipeline")
    if latent_lifter is None:
        raise ValueError("the official LBH workflow requires a learned latent lifter")
    config = config or H3LBHOfficialConfig()
    replacements = 0

    def visit(blocks):
        nonlocal replacements
        for name, block in list(blocks.sub_blocks.items()):
            if type(block) is MiniMaxH3DenoiseStep:
                blocks.sub_blocks[name] = MiniMaxH3LBHOfficialDenoiseStep(config, latent_lifter)
                replacements += 1
            elif getattr(block, "sub_blocks", None):
                visit(block)

    visit(pipe._blocks)
    if replacements != 1:
        raise RuntimeError(f"expected one T2VA/FL2VA denoise loop, replaced {replacements}")
    return pipe


__all__ = [
    "H3LBHOfficialConfig",
    "MiniMaxH3LBHOfficialDenoiseStep",
    "comfy_simple_sigmas",
    "map_shifted_sigmas",
    "patch_lbh_official_into_pipeline",
]
