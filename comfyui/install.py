"""Cross-platform installer for the Spark-H3 ComfyUI CUDA backend."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

DEFAULT_RELEASE_API = (
    "https://api.github.com/repos/zechengtang/Spark-H3/releases/tags/"
    "comfy-kitchen-spark-v0.2.36-1"
)


def _node_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if (here.parent / "comfyui_backend.py").is_file() else here


def _requirements(root: Path) -> Path | None:
    candidates = (
        root / "requirements.txt",
        root / "comfyui" / "standalone" / "requirements.txt",
    )
    return next((path for path in candidates if path.is_file()), None)


def _backend_available() -> bool:
    try:
        module = importlib.import_module("comfy_kitchen.backends.cuda")
        return callable(getattr(module, "spark_attn", None))
    except (ImportError, OSError):
        return False


def _validate_runtime() -> None:
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Spark-H3 requires the torch-compatible Triton build supplied by "
            "your ComfyUI environment. Install that build before this node; do "
            "not replace ComfyUI's torch package with a generic dependency."
        ) from error


def _pip_install(specification: str, *, force: bool = False) -> None:
    command = [sys.executable, "-m", "pip", "install"]
    if force:
        command.extend(("--force-reinstall", "--no-deps"))
    command.append(specification)
    print(f"[Spark-H3] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
    importlib.invalidate_caches()


def _wheel_tags(filename: str):
    try:
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
    except ImportError:
        from pip._vendor.packaging.tags import sys_tags
        from pip._vendor.packaging.utils import parse_wheel_filename
    try:
        _, _, _, tags = parse_wheel_filename(filename)
    except ValueError:
        return set(), set()
    return set(tags), set(sys_tags())


def _matching_local_wheel(root: Path) -> Path | None:
    candidates = sorted((root / "wheelhouse").glob("*.whl"))
    for wheel in candidates:
        wheel_tags, supported = _wheel_tags(wheel.name)
        if wheel_tags & supported:
            return wheel
    return None


def _matching_release_wheel(api_url: str) -> str | None:
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "ComfyUI-Spark-H3"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        release = json.load(response)
    for asset in release.get("assets", ()):
        name = asset.get("name", "")
        wheel_tags, supported = _wheel_tags(name)
        if wheel_tags & supported:
            return asset.get("browser_download_url")
    return None


def _install_requirements(root: Path) -> None:
    requirements = _requirements(root)
    if requirements is not None:
        _pip_install(f"-r{requirements}")


def _validate_backend() -> None:
    if not _backend_available():
        raise RuntimeError(
            "installation finished, but comfy_kitchen.backends.cuda.spark_attn "
            "cannot be imported"
        )
    from comfy_kitchen.backends.cuda import spark_attn

    assert callable(spark_attn)
    print("[Spark-H3] CUDA backend is ready.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", help="local path or URL of a Spark comfy-kitchen wheel")
    parser.add_argument("--source", action="store_true", help="skip wheel discovery and build source")
    parser.add_argument(
        "--no-source-fallback",
        action="store_true",
        help="fail if a compatible wheel is unavailable",
    )
    parser.add_argument("--skip-dependencies", action="store_true")
    parser.add_argument(
        "--release-api",
        default=os.environ.get("SPARK_H3_KERNEL_RELEASE_API", DEFAULT_RELEASE_API),
    )
    args = parser.parse_args(argv)
    root = _node_root()
    if not args.skip_dependencies:
        _install_requirements(root)
    _validate_runtime()
    if _backend_available() and not args.wheel and not args.source:
        print("[Spark-H3] compatible comfy-kitchen backend is already installed.")
        return 0

    wheel = args.wheel or os.environ.get("SPARK_H3_KERNEL_WHEEL")
    if wheel is None and not args.source:
        local = _matching_local_wheel(root)
        wheel = str(local) if local is not None else None
    if wheel is None and not args.source:
        try:
            wheel = _matching_release_wheel(args.release_api)
        except (OSError, ValueError) as error:
            print(f"[Spark-H3] release-wheel lookup failed: {error}", file=sys.stderr)
    if wheel is not None:
        _pip_install(wheel, force=True)
        _validate_backend()
        return 0
    if args.no_source_fallback:
        raise RuntimeError("no compatible Spark-H3 kernel wheel is available")

    print("[Spark-H3] no compatible wheel found; compiling the pinned source backend.")
    try:
        from .kernel_builder import install_from_source
    except ImportError:
        from kernel_builder import install_from_source
    install_from_source(
        root,
        source_url=os.environ.get(
            "SPARK_COMFY_KITCHEN_SOURCE",
            "https://github.com/Comfy-Org/comfy-kitchen.git",
        ),
    )
    _validate_backend()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
