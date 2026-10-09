"""Cross-platform installer for the Spark-H3 ComfyUI CUDA backend."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
import platform
import subprocess
import sys
import urllib.request
import urllib.parse
from pathlib import Path

SUPPORTED_KITCHEN_BASES = ("0.2.36", "0.2.37")
SUPPORTED_ARCHITECTURES = {
    (8, 9): "sm89",
    (12, 0): "sm120",
}
BACKEND_LOCAL_VERSION_PREFIXES = {
    "sm89": "spark.h3.sm89",
    "sm120": "spark.h3.sm120",
}
MINIMUM_RELEASE_CUDA = (13, 0)
DEFAULT_RELEASE_CUDA_TAG = "cu130"
PACKAGE_BUILD_IDENTITY = "spark_h3_build.json"
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
    "installed = version('comfy-kitchen')\n"
    "base, _, local = installed.partition('+')\n"
    "ok = callable(spark_attn) and base == sys.argv[1] and local == sys.argv[2]\n"
    "raise SystemExit(0 if ok else 1)\n"
)


def _cuda_tag(cuda_version: tuple[int, int]) -> str:
    return f"cu{cuda_version[0]}{cuda_version[1]}"


def _backend_local_version(
    architecture: str,
    cuda_tag: str = DEFAULT_RELEASE_CUDA_TAG,
) -> str:
    return f"{BACKEND_LOCAL_VERSION_PREFIXES[architecture]}.{cuda_tag}.1"


def _runtime_platform_tag() -> str:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}:
        return "linux-x86_64"
    if sys.platform == "win32" and machine in {"x86_64", "amd64"}:
        return "windows-x86_64"
    raise RuntimeError(
        f"unsupported Spark-H3 platform: sys.platform={sys.platform!r}, "
        f"machine={platform.machine()!r}"
    )


def _node_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if (here.parent / "comfyui_backend.py").is_file() else here


def _validate_package_identity(
    root: Path,
    *,
    architecture: str,
    cuda_tag: str,
    platform_tag: str,
) -> None:
    identity_path = root / PACKAGE_BUILD_IDENTITY
    if not identity_path.is_file():
        # The repository source tree intentionally has no generated package
        # identity. A standalone package always includes one.
        if root == Path(__file__).resolve().parent:
            raise RuntimeError(
                f"standalone package build identity is missing: {identity_path}"
            )
        return
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read package build identity {identity_path}") from error
    expected = {
        "architecture": architecture,
        "cuda_tag": cuda_tag,
        "platform": platform_tag,
    }
    mismatches = {
        key: (identity.get(key), value)
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={found!r} (expected {wanted!r})"
            for key, (found, wanted) in mismatches.items()
        )
        raise RuntimeError(f"Spark-H3 package identity mismatch: {details}")
    listed_wheels = identity.get("kernel_wheels")
    if not isinstance(listed_wheels, list) or not all(
        isinstance(name, str) for name in listed_wheels
    ):
        raise RuntimeError("Spark-H3 package identity has an invalid kernel_wheels list")
    actual_wheels = sorted(
        wheel.name for wheel in (root / "wheelhouse").glob("*.whl")
    )
    if sorted(listed_wheels) != actual_wheels:
        raise RuntimeError(
            "Spark-H3 package wheelhouse does not match its build identity: "
            f"listed={sorted(listed_wheels)!r}, actual={actual_wheels!r}"
        )


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


def _backend_available(
    kitchen_base: str,
    architecture: str = "sm120",
    cuda_tag: str = DEFAULT_RELEASE_CUDA_TAG,
) -> bool:
    """Probe the backend in a fresh interpreter.

    The installer may replace an older comfy-kitchen wheel in place.  Probing
    in this process would leave that old package in ``sys.modules``, causing
    the post-install check to inspect stale Python and native-extension module
    objects even though pip has already updated the files on disk.
    """
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                BACKEND_PROBE,
                kitchen_base,
                _backend_local_version(architecture, cuda_tag),
            ],
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


def _validate_runtime(*, experimental_cuda: bool = False) -> tuple[int, int]:
    try:
        import torch
        import triton  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Spark-H3 requires the torch-compatible Triton build supplied by "
            "your ComfyUI environment. Install that build before this node; do "
            "not replace ComfyUI's torch package with a generic dependency."
        ) from error
    cuda_text = torch.version.cuda
    if cuda_text is None:
        raise RuntimeError("Spark-H3 requires a CUDA-enabled PyTorch build")
    try:
        cuda_version = tuple(int(part) for part in cuda_text.split(".")[:2])
    except ValueError as error:
        raise RuntimeError(f"cannot parse PyTorch CUDA version {cuda_text!r}") from error
    if cuda_version < MINIMUM_RELEASE_CUDA:
        message = (
            f"PyTorch CUDA {cuda_text} is excluded from the Spark-H3 release plan "
            "because ComfyUI disables its optimized comfy-kitchen CUDA backend, "
            "causing a severe end-to-end performance regression"
        )
        if not experimental_cuda:
            raise RuntimeError(
                message + ". Use CUDA 13.0+ for a release installation, or pass "
                "--experimental-cuda together with --wheel or --source for local "
                "correctness and adaptation work."
            )
        print(f"[Spark-H3] WARNING: {message}.", file=sys.stderr, flush=True)
        print(
            "[Spark-H3] WARNING: experimental CUDA mode is not release-supported "
            "and its performance must not be compared with the CU130 release.",
            file=sys.stderr,
            flush=True,
        )
    return cuda_version


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


def _wheel_matches(
    filename: str,
    kitchen_base: str,
    architecture: str = "sm120",
    cuda_tag: str = DEFAULT_RELEASE_CUDA_TAG,
) -> bool:
    distribution, version, wheel_tags, supported = _wheel_metadata(filename)
    expected = f"{kitchen_base}+{_backend_local_version(architecture, cuda_tag)}"
    return (
        distribution == "comfy-kitchen"
        and version is not None
        and version == expected
        and bool(wheel_tags & supported)
    )


def _matching_local_wheel(
    root: Path,
    kitchen_base: str,
    architecture: str = "sm120",
    cuda_tag: str = DEFAULT_RELEASE_CUDA_TAG,
) -> Path | None:
    candidates = sorted((root / "wheelhouse").glob("*.whl"))
    for wheel in candidates:
        if _wheel_matches(wheel.name, kitchen_base, architecture, cuda_tag):
            return wheel
    return None


def _matching_release_wheel(
    api_url: str,
    kitchen_base: str,
    architecture: str = "sm120",
    cuda_tag: str = DEFAULT_RELEASE_CUDA_TAG,
) -> str | None:
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "ComfyUI-Spark-H3"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        release = json.load(response)
    for asset in release.get("assets", ()):
        name = asset.get("name", "")
        if _wheel_matches(name, kitchen_base, architecture, cuda_tag):
            return asset.get("browser_download_url")
    return None


def _install_requirements(root: Path) -> None:
    requirements = _requirements(root)
    if requirements is not None:
        _pip_install(f"-r{requirements}")


def _validate_backend(
    kitchen_base: str,
    architecture: str = "sm120",
    cuda_tag: str = DEFAULT_RELEASE_CUDA_TAG,
) -> None:
    if not _backend_available(kitchen_base, architecture, cuda_tag):
        raise RuntimeError(
            f"installation finished, but comfy-kitchen "
            f"{kitchen_base}+{_backend_local_version(architecture, cuda_tag)} "
            "cannot be imported with a callable CUDA spark_attn backend"
        )
    print(f"[Spark-H3] {architecture.upper()} {cuda_tag.upper()} backend is ready.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", help="local path or URL of a Spark comfy-kitchen wheel")
    parser.add_argument("--source", action="store_true", help="skip wheel discovery and build source")
    parser.add_argument(
        "--experimental-cuda",
        action="store_true",
        help="opt in to a CUDA < 13 diagnostic install; requires --wheel or --source",
    )
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
    cuda_version = _validate_runtime(experimental_cuda=args.experimental_cuda)
    cuda_tag = _cuda_tag(cuda_version)
    platform_tag = _runtime_platform_tag()
    experimental_runtime = cuda_version < MINIMUM_RELEASE_CUDA
    requested_wheel = args.wheel or os.environ.get("SPARK_H3_KERNEL_WHEEL")
    if experimental_runtime and not (args.source or requested_wheel):
        raise RuntimeError(
            "experimental CUDA installs require an explicit --wheel or --source; "
            "automatic release-wheel discovery is disabled"
        )
    capability = _cuda_capability()
    print(f"[Spark-H3] detected GPU architecture: SM{capability[0]}{capability[1]}", flush=True)
    architecture = SUPPORTED_ARCHITECTURES.get(capability)
    if architecture is None:
        raise RuntimeError(
            f"this ComfyUI package supports SM89 and SM120, found "
            f"SM{capability[0]}{capability[1]}"
        )
    _validate_package_identity(
        root,
        architecture=architecture,
        cuda_tag=cuda_tag,
        platform_tag=platform_tag,
    )
    kitchen_base = _resolve_kitchen_base(args.kitchen_base)
    print(
        f"[Spark-H3] target comfy-kitchen base: {kitchen_base}; "
        f"kernel architecture: {architecture.upper()}; CUDA tag: {cuda_tag}; "
        f"platform: {platform_tag}",
        flush=True,
    )
    if (
        _backend_available(kitchen_base, architecture, cuda_tag)
        and not args.wheel
        and not args.source
    ):
        print("[Spark-H3] compatible Spark comfy-kitchen backend is already installed.")
        return 0

    wheel = requested_wheel
    if wheel is not None and not _wheel_matches(
        wheel, kitchen_base, architecture, cuda_tag
    ):
        raise RuntimeError(
            f"wheel {wheel!r} does not match comfy-kitchen {kitchen_base} "
            f"for {architecture.upper()} or this Python platform"
        )
    if wheel is None and not args.source:
        local = _matching_local_wheel(root, kitchen_base, architecture, cuda_tag)
        wheel = str(local) if local is not None else None
    if wheel is None and not args.source:
        release_api = args.release_api or DEFAULT_RELEASE_APIS[kitchen_base]
        try:
            wheel = _matching_release_wheel(
                release_api, kitchen_base, architecture, cuda_tag
            )
        except (OSError, ValueError) as error:
            print(f"[Spark-H3] release-wheel lookup failed: {error}", file=sys.stderr)
    if wheel is not None:
        _pip_install(wheel, force=True)
        _validate_backend(kitchen_base, architecture, cuda_tag)
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
        architecture=architecture,
        cuda_tag=cuda_tag,
        cuda_archs=(
            "120a"
            if experimental_runtime and architecture == "sm120" and cuda_version < (13, 0)
            else None
        ),
        source_url=os.environ.get(
            "SPARK_COMFY_KITCHEN_SOURCE",
            "https://github.com/Comfy-Org/comfy-kitchen.git",
        ),
    )
    _validate_backend(kitchen_base, architecture, cuda_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
