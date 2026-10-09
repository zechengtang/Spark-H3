"""Cross-platform installer for the Spark-H3 ComfyUI CUDA backend."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
import subprocess
import sys
import urllib.request
import urllib.parse
from pathlib import Path

SUPPORTED_KITCHEN_BASES = ("0.2.36", "0.2.37")
DEFAULT_RELEASE_APIS = {
    base: (
        "https://api.github.com/repos/zechengtang/Spark-H3/releases/tags/"
        f"comfy-kitchen-spark-v{base}-1"
    )
    for base in SUPPORTED_KITCHEN_BASES
}

BACKEND_PROBE = (
    "from importlib.metadata import version\n"
    "import sys\n"
    "from comfy_kitchen.backends.cuda import spark_attn\n"
    "base = version('comfy-kitchen').split('+', 1)[0]\n"
    "raise SystemExit(0 if callable(spark_attn) and base == sys.argv[1] else 1)\n"
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


def _installed_kitchen_base() -> str | None:
    try:
        installed = metadata.version("comfy-kitchen")
    except metadata.PackageNotFoundError:
        return None
    return installed.split("+", 1)[0]


def _resolve_kitchen_base(requested: str | None = None) -> str:
    base = requested or _installed_kitchen_base()
    if base not in SUPPORTED_KITCHEN_BASES:
        found = "not installed" if base is None else base
        raise RuntimeError(
            f"unsupported comfy-kitchen base {found}; this package supports "
            "ComfyUI 0.38.x (comfy-kitchen 0.2.36) and ComfyUI 0.39.x "
            "(comfy-kitchen 0.2.37). Use --kitchen-base only when the target "
            "ComfyUI version is known."
        )
    return base


def _backend_available(kitchen_base: str) -> bool:
    """Probe the backend in a fresh interpreter.

    The installer may replace an older comfy-kitchen wheel in place.  Probing
    in this process would leave that old package in ``sys.modules``, causing
    the post-install check to inspect stale Python and native-extension module
    objects even though pip has already updated the files on disk.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", BACKEND_PROBE, kitchen_base],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _cuda_capability() -> tuple[int, int]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Spark-H3 installation requires an available NVIDIA CUDA GPU")
    return tuple(torch.cuda.get_device_capability())


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


def _wheel_metadata(filename: str):
    try:
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
    except ImportError:
        from pip._vendor.packaging.tags import sys_tags
        from pip._vendor.packaging.utils import parse_wheel_filename
    try:
        name = Path(urllib.parse.unquote(urllib.parse.urlparse(filename).path)).name
        distribution, version, _, tags = parse_wheel_filename(name)
    except ValueError:
        return None, None, set(), set()
    return distribution, str(version), set(tags), set(sys_tags())


def _wheel_matches(filename: str, kitchen_base: str) -> bool:
    distribution, version, wheel_tags, supported = _wheel_metadata(filename)
    return (
        distribution == "comfy-kitchen"
        and version is not None
        and version.split("+", 1)[0] == kitchen_base
        and bool(wheel_tags & supported)
    )


def _matching_local_wheel(root: Path, kitchen_base: str) -> Path | None:
    candidates = sorted((root / "wheelhouse").glob("*.whl"))
    for wheel in candidates:
        if _wheel_matches(wheel.name, kitchen_base):
            return wheel
    return None


def _matching_release_wheel(api_url: str, kitchen_base: str) -> str | None:
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "ComfyUI-Spark-H3"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        release = json.load(response)
    for asset in release.get("assets", ()):
        name = asset.get("name", "")
        if _wheel_matches(name, kitchen_base):
            return asset.get("browser_download_url")
    return None


def _install_requirements(root: Path) -> None:
    requirements = _requirements(root)
    if requirements is not None:
        _pip_install(f"-r{requirements}")


def _validate_backend(kitchen_base: str) -> None:
    if not _backend_available(kitchen_base):
        raise RuntimeError(
            f"installation finished, but comfy-kitchen {kitchen_base}+spark.h3.1 "
            "cannot be imported with a callable CUDA spark_attn backend"
        )
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
        "--kitchen-base",
        choices=SUPPORTED_KITCHEN_BASES,
        help="override automatic selection for ComfyUI 0.38.x or 0.39.x",
    )
    parser.add_argument(
        "--release-api",
        default=os.environ.get("SPARK_H3_KERNEL_RELEASE_API"),
    )
    args = parser.parse_args(argv)
    root = _node_root()
    if not args.skip_dependencies:
        _install_requirements(root)
    _validate_runtime()
    kitchen_base = _resolve_kitchen_base(args.kitchen_base)
    print(f"[Spark-H3] target comfy-kitchen base: {kitchen_base}", flush=True)
    capability = _cuda_capability()
    print(f"[Spark-H3] detected GPU architecture: SM{capability[0]}{capability[1]}", flush=True)
    if capability != (12, 0):
        raise RuntimeError(
            f"this ComfyUI package supports SM120, found "
            f"SM{capability[0]}{capability[1]}"
        )
    if _backend_available(kitchen_base) and not args.wheel and not args.source:
        print("[Spark-H3] compatible Spark comfy-kitchen backend is already installed.")
        return 0

    wheel = args.wheel or os.environ.get("SPARK_H3_KERNEL_WHEEL")
    if wheel is not None and not _wheel_matches(wheel, kitchen_base):
        raise RuntimeError(
            f"wheel {wheel!r} does not match comfy-kitchen {kitchen_base} "
            "or this Python platform"
        )
    if wheel is None and not args.source:
        local = _matching_local_wheel(root, kitchen_base)
        wheel = str(local) if local is not None else None
    if wheel is None and not args.source:
        release_api = args.release_api or DEFAULT_RELEASE_APIS[kitchen_base]
        try:
            wheel = _matching_release_wheel(release_api, kitchen_base)
        except (OSError, ValueError) as error:
            print(f"[Spark-H3] release-wheel lookup failed: {error}", file=sys.stderr)
    if wheel is not None:
        _pip_install(wheel, force=True)
        _validate_backend(kitchen_base)
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
        kitchen_base=kitchen_base,
        source_url=os.environ.get(
            "SPARK_COMFY_KITCHEN_SOURCE",
            "https://github.com/Comfy-Org/comfy-kitchen.git",
        ),
    )
    _validate_backend(kitchen_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
