"""Audit wheels and run Spark-H3 ComfyUI validation on Linux or Windows."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from config import KIT, PLATFORM, wheel_base


SUPPORTED = {"0.38": "0.2.36", "0.39": "0.2.37", "0.40": "0.2.37", "0.42": "0.2.39"}


def comfy_identity(root: Path) -> tuple[str, str]:
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    base = re.search(r"comfy-kitchen==([0-9.]+)", requirements)
    source = (root / "comfyui_version.py").read_text(encoding="utf-8")
    version = re.search(r"__version__\s*=\s*['\"]([0-9]+\.[0-9]+\.[0-9]+)['\"]", source)
    if base is None or version is None:
        raise RuntimeError("cannot identify ComfyUI or comfy-kitchen version")
    series = ".".join(version.group(1).split(".")[:2])
    if series not in SUPPORTED or SUPPORTED[series] != base.group(1):
        raise RuntimeError(f"unsupported ComfyUI {version.group(1)} / comfy-kitchen {base.group(1)}")
    return version.group(1), base.group(1)


def validate(root: Path | None, python: Path, mode: str, results: Path,
             port: int, steps: int, lora_name: str) -> Path:
    host = "windows-x86_64" if sys.platform == "win32" else "linux-x86_64" if sys.platform.startswith("linux") else None
    if host != PLATFORM:
        raise RuntimeError(f"{PLATFORM} kit must be validated on {PLATFORM}; current host is {sys.platform}")
    if not python.is_file():
        raise FileNotFoundError(python)
    if Path(lora_name).name != lora_name or not lora_name.lower().endswith(".safetensors"):
        raise ValueError("--lora-name must be a .safetensors filename")
    results.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(python), str(KIT / "preflight.py"), "--report", str(results / "preflight.json")], check=True)
    subprocess.run([
        str(python), str(KIT / "validate_wheels.py"), "--wheelhouse", str(KIT / "wheelhouse"),
        "--report", str(results / "wheel-validation.json"),
    ], check=True)
    if mode == "audit":
        return results
    if root is None:
        raise ValueError("--comfyui-root is required for full mode")
    _, base = comfy_identity(root)
    wheels = [wheel for wheel in (KIT / "wheelhouse").glob("comfy_kitchen-*.whl") if wheel_base(wheel.name) == base]
    if len(wheels) != 1:
        raise RuntimeError(f"expected one {PLATFORM} wheel for comfy-kitchen {base}; found {wheels}")
    subprocess.run([str(python), "-c", "import requests, websocket"], check=True)
    for executable in ("ffprobe", "ffmpeg"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"full validation requires {executable}")
    env = {**os.environ, "COMFYUI_ROOT": str(root), "SPARK_WHEEL": str(wheels[0]),
           "SPARK_RESULTS": str(results), "SPARK_TEST_PORT": str(port),
           "SPARK_STEPS": str(steps), "SPARK_LORA_NAME": lora_name}
    subprocess.run([str(python), str(KIT / "run_lightx2v_three_variants.py")], env=env, check=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mode", choices=("audit", "full"), default="full")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--port", type=int, default=8628)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lora-name", default="minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors")
    args = parser.parse_args()
    if args.steps < 3:
        parser.error("--steps must be at least 3")
    result = validate(
        args.comfyui_root.resolve() if args.comfyui_root else None,
        args.python.absolute(), args.mode,
        (args.results or KIT / f"results-{args.mode}").resolve(),
        args.port, args.steps, args.lora_name,
    )
    print(json.dumps({"valid": True, "platform": PLATFORM, "results": str(result)}))


if __name__ == "__main__":
    main()
