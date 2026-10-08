"""Learned latent upscaler used by LBH's MiniMax-H3 workflow."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


LBH_REPO_ID = "LBH-123-AI/Minimax_h3_latent_Upscaler"
LBH_FILENAMES = {
    "bf16": "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
    "fp16": "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
}
LBH_LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LBH_LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)


def _resolve_checkpoint(source, filename, *, cache_dir=None, local_files_only=False, revision=None):
    path = Path(source).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        candidate = path / filename
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"checkpoint not found: {candidate}")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=str(source), filename=filename, cache_dir=cache_dir,
        local_files_only=local_files_only, revision=revision,
    )


def _group_norm(channels):
    return nn.GroupNorm(32, channels)


class _LBHResBlock3D(nn.Module):
    def __init__(self, channels, embed_dim=64, dropout=0.1):
        super().__init__()
        self.in_layers = nn.Sequential(
            _group_norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1)
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * channels))
        self.out_norm = _group_norm(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(dropout), nn.Conv3d(channels, channels, 3, padding=1)
        )

    def forward(self, hidden_states, embedding):
        hidden_states = self.in_layers(hidden_states)
        scale, shift = self.emb_layers(embedding).to(hidden_states.dtype).chunk(2, dim=1)
        hidden_states = self.out_norm(hidden_states) * (1 + scale[..., None, None, None])
        hidden_states = hidden_states + shift[..., None, None, None]
        return self.out_layers(hidden_states)


class _LBHTemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.norm = _group_norm(channels)
        self.dwconv = nn.Conv3d(
            channels, channels, (kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0), groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, 1)

    def forward(self, hidden_states):
        return hidden_states + self.pwconv(self.dwconv(F.silu(self.norm(hidden_states))))


class LBHMiniMaxH3LatentUpscaler(nn.Module):
    """Exact v1 LBH 3D-conv architecture, independent of ComfyUI."""

    def __init__(
        self, in_channels=24, channels=512, in_blocks=12, out_blocks=12,
        temporal_every=2, temporal_kernel=5, dropout=0.1, embed_dim=64,
    ):
        super().__init__()
        self.scale_range = (1.0, 4.0)
        self.temporal_kernel = temporal_kernel
        self.register_buffer(
            "latents_mean", torch.tensor(LBH_LATENTS_MEAN).view(1, -1, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "latents_std", torch.tensor(LBH_LATENTS_STD).view(1, -1, 1, 1, 1), persistent=False
        )
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.in_blocks = self._make_blocks(
            in_blocks, channels, embed_dim, dropout, temporal_every, temporal_kernel
        )
        self.out_blocks = self._make_blocks(
            out_blocks, channels, embed_dim, dropout, temporal_every, temporal_kernel
        )
        self.norm_out = _group_norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    @staticmethod
    def _make_blocks(count, channels, embed_dim, dropout, temporal_every, temporal_kernel):
        blocks = nn.ModuleList()
        for index in range(count):
            blocks.append(_LBHResBlock3D(channels, embed_dim, dropout))
            if temporal_every > 0 and index % temporal_every == 0:
                blocks.append(_LBHTemporalConv(channels, temporal_kernel))
        return blocks

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path=LBH_REPO_ID, *, variant="fp16",
        filename=None, torch_dtype=None, device="cpu", cache_dir=None,
        local_files_only=False, revision=None,
    ):
        """Load an LBH v1 safetensors checkpoint from a path or the Hub."""
        if filename is None:
            try:
                filename = LBH_FILENAMES[variant]
            except KeyError as error:
                raise ValueError(f"variant must be one of {tuple(LBH_FILENAMES)}") from error
        path = _resolve_checkpoint(
            pretrained_model_name_or_path, filename, cache_dir=cache_dir,
            local_files_only=local_files_only, revision=revision,
        )
        from safetensors.torch import load_file

        state_dict = load_file(path, device="cpu")
        if any(key.startswith("upscaler.") for key in state_dict):
            state_dict = {
                key.removeprefix("upscaler."): value
                for key, value in state_dict.items() if key.startswith("upscaler.")
            }
        weight = state_dict.get("conv_in.weight")
        if weight is None or tuple(weight.shape[1:]) != (24, 3, 3, 3):
            raise ValueError("not an LBH MiniMax-H3 24-channel 3D upscaler checkpoint")
        if torch_dtype is not None:
            state_dict = {
                key: value.to(torch_dtype) if value.is_floating_point() else value
                for key, value in state_dict.items()
            }
        channels = weight.shape[0]
        in_count = sum(
            key.endswith("in_layers.2.weight")
            for key in state_dict if key.startswith("in_blocks.")
        )
        out_count = sum(
            key.endswith("in_layers.2.weight")
            for key in state_dict if key.startswith("out_blocks.")
        )
        temporal = [value for key, value in state_dict.items() if key.endswith("dwconv.weight")]
        with torch.device("meta"):
            model = cls(
                channels=channels, in_blocks=in_count, out_blocks=out_count,
                temporal_every=2 if temporal else 0,
                temporal_kernel=temporal[0].shape[2] if temporal else 5,
            )
        model.load_state_dict(state_dict, strict=True, assign=True)
        buffer_dtype = torch_dtype or weight.dtype
        model.latents_mean = torch.tensor(LBH_LATENTS_MEAN, dtype=buffer_dtype).view(1, -1, 1, 1, 1)
        model.latents_std = torch.tensor(LBH_LATENTS_STD, dtype=buffer_dtype).view(1, -1, 1, 1, 1)
        return model.to(device=device).eval().requires_grad_(False)

    def _forward_segment(self, hidden_states, scale, target_size):
        embedding = self.embed(hidden_states.new_tensor([[scale - 1.0]])).expand(
            hidden_states.shape[0], -1
        )
        hidden_states = self.conv_in(hidden_states)
        for block in self.in_blocks:
            hidden_states = (
                hidden_states + block(hidden_states, embedding)
                if isinstance(block, _LBHResBlock3D) else block(hidden_states)
            )
        hidden_states = F.interpolate(
            hidden_states, size=target_size, mode="trilinear", align_corners=False
        )
        for block in self.out_blocks:
            hidden_states = (
                hidden_states + block(hidden_states, embedding)
                if isinstance(block, _LBHResBlock3D) else block(hidden_states)
            )
        return self.conv_out(F.silu(self.norm_out(hidden_states)))

    def forward(self, latents, target_hw):
        if latents.ndim != 5 or latents.shape[1] != 24:
            raise ValueError(f"expected H3 latents [B,24,T,H,W], got {tuple(latents.shape)}")
        target_h, target_w = map(int, target_hw)
        height, width = latents.shape[-2:]
        scale_h, scale_w = target_h / height, target_w / width
        if target_h < height or target_w < width:
            raise ValueError("LBH target dimensions must not downscale either spatial axis")
        scale = (scale_h + scale_w) / 2.0
        if not self.scale_range[0] <= scale <= self.scale_range[1]:
            raise ValueError(
                f"LBH effective spatial scale must be in {self.scale_range}, got {scale:.4f}"
            )
        parameter = self.conv_in.weight
        input_device, input_dtype = latents.device, latents.dtype
        work = latents.to(device=parameter.device, dtype=parameter.dtype)
        mean = self.latents_mean.to(device=work.device, dtype=work.dtype)
        std = self.latents_std.to(device=work.device, dtype=work.dtype)
        work = (work - mean) / std
        frames = work.shape[2]
        if frames <= 32:
            output = self._forward_segment(work, scale, (frames, target_h, target_w))
        else:
            temporal_layers = sum(
                isinstance(block, _LBHTemporalConv)
                for block in (*self.in_blocks, *self.out_blocks)
            )
            radius = temporal_layers * (self.temporal_kernel // 2)
            pieces = []
            for start in range(0, frames, 32):
                end = min(start + 32, frames)
                lo, hi = max(0, start - radius), min(frames, end + radius)
                segment = self._forward_segment(
                    work[:, :, lo:hi], scale, (hi - lo, target_h, target_w)
                )
                pieces.append(segment[:, :, start - lo:end - lo])
            output = torch.cat(pieces, dim=2)
        output = output * std + mean
        return output.to(device=input_device, dtype=input_dtype)


__all__ = ["LBHMiniMaxH3LatentUpscaler"]
