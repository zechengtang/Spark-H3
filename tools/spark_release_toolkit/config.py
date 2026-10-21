"""Shared architecture and runtime identity for a generated release kit."""

from __future__ import annotations

import json
import os
from pathlib import Path

KIT = Path(__file__).resolve().parent
IDENTITY = json.loads((KIT / "release.json").read_text(encoding="utf-8"))
ARCH = IDENTITY["architecture"]
PLATFORM = IDENTITY["platform"]
if PLATFORM not in ("linux-x86_64", "windows-x86_64"):
    raise ValueError(f"unsupported release platform: {PLATFORM}")
WHEEL_PLATFORM = "win_amd64" if PLATFORM == "windows-x86_64" else "linux_x86_64"
EXTENSION_SUFFIX = ".pyd" if PLATFORM == "windows-x86_64" else ".so"
CUDA_TAG = IDENTITY["cuda_tag"]
CAPABILITY = tuple(IDENTITY["capability"])
ELF_ARCH = "sm_" + ARCH[2:]
NODE = "MiniMaxH3SparkAttention" + ARCH.upper()
ACTIVATION = f"[Spark-H3] active on {ARCH.upper()}"
LOCAL_VERSION = f"spark.h3.{ARCH}.{CUDA_TAG}.1"
RESULTS = Path(os.environ.get("SPARK_RESULTS", str(KIT / "results"))).resolve()
WHEEL = Path(os.environ["SPARK_WHEEL"]).resolve() if "SPARK_WHEEL" in os.environ else None
PORT = int(os.environ.get("SPARK_TEST_PORT", "8628"))
STEPS = int(os.environ.get("SPARK_STEPS", "8"))
if STEPS < 3:
    raise ValueError("SPARK_STEPS must be at least 3 for the Fidelity warmup and sparse validation")
LORA_NAME = os.environ.get(
    "SPARK_LORA_NAME", "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
)
if (not LORA_NAME or Path(LORA_NAME).name != LORA_NAME
        or not LORA_NAME.lower().endswith(".safetensors")):
    raise ValueError("SPARK_LORA_NAME must be a .safetensors filename under ComfyUI models/loras")


def wheel_base(wheel_name: str) -> str | None:
    """Return a supported kitchen base for a wheel matching this kit."""
    import re

    match = re.fullmatch(
        rf"comfy_kitchen-(0\.2\.(?:36|37|39))\+{re.escape(LOCAL_VERSION)}-.*-"
        rf"(?P<platform>[^-]+)\.whl", wheel_name,
    )
    if match is None:
        return None
    tag = match.group("platform")
    if PLATFORM == "windows-x86_64":
        return match.group(1) if tag == "win_amd64" else None
    if tag == "linux_x86_64" or (tag.startswith("manylinux") and tag.endswith("_x86_64")):
        return match.group(1)
    return None
