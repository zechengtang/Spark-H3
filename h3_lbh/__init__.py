"""Diffusers implementation of LBH's two-pass MiniMax-H3 workflow."""

from .lora import apply_comfy_h3_lora
from .models import LBHMiniMaxH3LatentUpscaler
from .workflow import H3LBHOfficialConfig, patch_lbh_official_into_pipeline
from .resolution import megapixel_canvas, official_canvas_pair

__all__ = [
    "H3LBHOfficialConfig",
    "LBHMiniMaxH3LatentUpscaler",
    "apply_comfy_h3_lora",
    "patch_lbh_official_into_pipeline",
    "megapixel_canvas",
    "official_canvas_pair",
]
