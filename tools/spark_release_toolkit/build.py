"""Build architecture-specific spark-h3-kitchen wheels on the target OS."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from config import ARCH, CUDA_TAG, KIT, PLATFORM, wheel_implementation


def require_host() -> None:
    host = "windows-x86_64" if sys.platform == "win32" else "linux-x86_64" if sys.platform.startswith("linux") else None
    if host != PLATFORM:
        raise RuntimeError(f"{PLATFORM} kit must be built on {PLATFORM}; current host is {sys.platform}")


def build(python: Path, implementations: tuple[str, ...]) -> list[Path]:
    require_host()
    required = ("git", "nvcc", "cl" if PLATFORM == "windows-x86_64" else "g++")
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"missing build tools: {', '.join(missing)}")
    if not python.is_file():
        raise FileNotFoundError(python)
    report_path = KIT / "build-preflight.json"
    subprocess.run([str(python), str(KIT / "preflight.py"), "--report", str(report_path),
                    "--require-build-toolkit"], check=True)
    toolkit = json.loads(report_path.read_text(encoding="utf-8"))["build_toolkit"]
    env = {**os.environ, "CUDA_HOME": toolkit["toolkit_root"],
           "CUDA_PATH": toolkit["toolkit_root"], "CUDAToolkit_ROOT": toolkit["toolkit_root"],
           "CUDA_TOOLKIT_ROOT_DIR": toolkit["toolkit_root"],
           "CUDACXX": toolkit["compiler"], "CMAKE_CUDA_COMPILER": toolkit["compiler"],
           "CUDA_NVCC_EXECUTABLE": toolkit["compiler"]}
    wheelhouse = KIT / "wheelhouse"
    wheelhouse.mkdir(exist_ok=True)
    results = []
    for implementation in implementations:
        subprocess.run([
            str(python), str(KIT / "ComfyUI-Spark-H3/kernel_builder.py"),
            "--python", str(python), "--output-dir", str(wheelhouse),
            "--kitchen-implementation", implementation,
            "--architecture", ARCH, "--cuda-tag", CUDA_TAG,
        ], env=env, check=True)
        matches = [
            path for path in wheelhouse.glob("spark_h3_kitchen-*.whl")
            if wheel_implementation(path.name) == implementation
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {PLATFORM} spark-h3-kitchen wheel for "
                f"implementation {implementation}; found {matches}"
            )
        results.append(matches[0])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--kitchen-implementation",
        choices=("all", "0.2.37", "0.2.39"),
        default="all",
    )
    args = parser.parse_args()
    implementations = (
        ("0.2.37", "0.2.39")
        if args.kitchen_implementation == "all"
        else (args.kitchen_implementation,)
    )
    for wheel in build(args.python.absolute(), implementations):
        print(json.dumps({"wheel": str(wheel), "platform": PLATFORM}))


if __name__ == "__main__":
    main()
