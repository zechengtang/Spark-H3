"""Build the pinned Spark-enabled comfy-kitchen backend.

This module is intentionally usable on Linux and Windows.  The custom-node
installer imports it for its source fallback, while maintainers use the CLI to
produce release wheels.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

KITCHEN_SOURCE = "https://github.com/Comfy-Org/comfy-kitchen.git"
KITCHEN_TARGETS = {
    "0.2.36": {
        "tag": "v0.2.36",
        "revision": "888b13e2c0e721f6576fe351a2ad79894b1c451f",
    },
    "0.2.37": {
        "tag": "v0.2.37",
        "revision": "be003b7c23c5b01328657955b8bc5d3f073d868e",
    },
}
DEFAULT_KITCHEN_BASE = "0.2.37"
DEFAULT_ARCHITECTURE = "sm120"
DEFAULT_CUDA_TAG = "cu130"
ARCHITECTURES = {
    "sm89": {"capability": (8, 9), "cuda_archs": "89"},
    "sm120": {"capability": (12, 0), "cuda_archs": "120f"},
}
CUDA_ARCH_OVERRIDES = {
    ("sm120", "cu128"): "120a",
    ("sm120", "cu129"): "120a",
}
# Retain the public constant for existing SM120 release automation.
SPARK_LOCAL_VERSION = "spark.h3.sm120.cu130.1"
CUDA_ARCHS = ARCHITECTURES[DEFAULT_ARCHITECTURE]["cuda_archs"]


def _run(command: list[str], *, cwd: Path | None = None, env=None) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"[Spark-H3] {printable}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _patch_path(node_root: Path) -> Path:
    candidates = (
        node_root / "patches" / "comfy-kitchen-spark-v0.2.36.patch",
        node_root / "comfyui" / "patches" / "comfy-kitchen-spark-v0.2.36.patch",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("the comfy-kitchen Spark patch is missing from this package")


def _architecture(name: str) -> dict:
    try:
        return ARCHITECTURES[name]
    except KeyError as error:
        raise ValueError(
            f"unsupported Spark architecture {name!r}; expected one of "
            f"{', '.join(ARCHITECTURES)}"
        ) from error


def _validate_cuda_tag(cuda_tag: str) -> str:
    if re.fullmatch(r"cu\d+", cuda_tag) is None:
        raise ValueError(
            f"invalid CUDA tag {cuda_tag!r}; expected a tag such as 'cu128' or 'cu130'"
        )
    return cuda_tag


def _cuda_architecture(architecture: str, cuda_tag: str) -> str:
    _validate_cuda_tag(cuda_tag)
    return CUDA_ARCH_OVERRIDES.get(
        (architecture, cuda_tag), str(_architecture(architecture)["cuda_archs"])
    )


def _spark_local_version(
    architecture: str = DEFAULT_ARCHITECTURE,
    cuda_tag: str = DEFAULT_CUDA_TAG,
) -> str:
    _architecture(architecture)
    _validate_cuda_tag(cuda_tag)
    return f"spark.h3.{architecture}.{cuda_tag}.1"


def architecture_for_capability(capability: tuple[int, int]) -> str:
    for name, target in ARCHITECTURES.items():
        if tuple(target["capability"]) == tuple(capability):
            return name
    raise ValueError(
        f"unsupported Spark GPU architecture SM{capability[0]}{capability[1]}"
    )


def _spark_version(
    kitchen_base: str,
    architecture: str = DEFAULT_ARCHITECTURE,
    cuda_tag: str = DEFAULT_CUDA_TAG,
) -> str:
    if kitchen_base not in KITCHEN_TARGETS:
        raise ValueError(
            f"unsupported comfy-kitchen base {kitchen_base!r}; expected one of "
            f"{', '.join(KITCHEN_TARGETS)}"
        )
    return f"{kitchen_base}+{_spark_local_version(architecture, cuda_tag)}"


def _set_local_version(
    source: Path,
    kitchen_base: str,
    architecture: str = DEFAULT_ARCHITECTURE,
    cuda_tag: str = DEFAULT_CUDA_TAG,
) -> None:
    pyproject = source / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    old = f'version = "{kitchen_base}"'
    new = f'version = "{_spark_version(kitchen_base, architecture, cuda_tag)}"'
    if old not in text:
        raise RuntimeError(f"cannot find {old!r} in the pinned comfy-kitchen source")
    pyproject.write_text(text.replace(old, new, 1), encoding="utf-8")


def prepare_source(
    node_root: Path,
    destination: Path,
    *,
    source_url: str,
    kitchen_base: str = DEFAULT_KITCHEN_BASE,
    architecture: str = DEFAULT_ARCHITECTURE,
    cuda_tag: str = DEFAULT_CUDA_TAG,
) -> Path:
    """Clone, pin, patch, and version the backend in ``destination``."""

    if destination.exists():
        raise FileExistsError(f"build destination already exists: {destination}")
    _spark_version(kitchen_base, architecture, cuda_tag)
    target = KITCHEN_TARGETS[kitchen_base]
    _run([
        "git", "clone", "--quiet", "--depth", "1", "--branch", target["tag"],
        "--recursive", source_url, str(destination),
    ])
    _run(["git", "checkout", "--quiet", target["revision"]], cwd=destination)
    _run(
        ["git", "submodule", "update", "--init", "--recursive", "--quiet"],
        cwd=destination,
    )
    patch = _patch_path(node_root.resolve())
    _run(["git", "apply", "--check", str(patch)], cwd=destination)
    _run(["git", "apply", str(patch)], cwd=destination)
    _set_local_version(destination, kitchen_base, architecture, cuda_tag)
    return destination


def build_wheel(
    node_root: Path,
    output_dir: Path,
    *,
    python: str = sys.executable,
    source_url: str = KITCHEN_SOURCE,
    kitchen_base: str = DEFAULT_KITCHEN_BASE,
    architecture: str = DEFAULT_ARCHITECTURE,
    cuda_tag: str = DEFAULT_CUDA_TAG,
    cuda_archs: str | None = None,
) -> list[Path]:
    """Build one architecture-specific wheel and return the created paths."""

    expected_cuda_archs = _cuda_architecture(architecture, cuda_tag)
    if cuda_archs is not None and cuda_archs != expected_cuda_archs:
        raise ValueError(
            f"CUDA architecture {cuda_archs!r} does not match "
            f"{architecture}/{cuda_tag}; expected {expected_cuda_archs!r}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spark-comfy-kitchen-") as temporary:
        source = prepare_source(
            node_root,
            Path(temporary) / "comfy-kitchen",
            source_url=source_url,
            kitchen_base=kitchen_base,
            architecture=architecture,
            cuda_tag=cuda_tag,
        )
        env = os.environ.copy()
        env["COMFY_CUDA_ARCHS"] = expected_cuda_archs
        _run(
            [python, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(output_dir), "."],
            cwd=source,
            env=env,
        )
    wheels = sorted(
        output_dir.glob(
            f"comfy_kitchen-{_spark_version(kitchen_base, architecture, cuda_tag)}-*.whl"
        )
    )
    if not wheels:
        raise RuntimeError("comfy-kitchen build completed without producing a wheel")
    return wheels


def install_from_source(
    node_root: Path,
    *,
    python: str = sys.executable,
    source_url: str = KITCHEN_SOURCE,
    kitchen_base: str = DEFAULT_KITCHEN_BASE,
    architecture: str = DEFAULT_ARCHITECTURE,
    cuda_tag: str = DEFAULT_CUDA_TAG,
    cuda_archs: str | None = None,
) -> None:
    """Compile and install the pinned backend into ``python``."""

    expected_cuda_archs = _cuda_architecture(architecture, cuda_tag)
    if cuda_archs is not None and cuda_archs != expected_cuda_archs:
        raise ValueError(
            f"CUDA architecture {cuda_archs!r} does not match "
            f"{architecture}/{cuda_tag}; expected {expected_cuda_archs!r}"
        )
    with tempfile.TemporaryDirectory(prefix="spark-comfy-kitchen-") as temporary:
        source = prepare_source(
            node_root,
            Path(temporary) / "comfy-kitchen",
            source_url=source_url,
            kitchen_base=kitchen_base,
            architecture=architecture,
            cuda_tag=cuda_tag,
        )
        env = os.environ.copy()
        env["COMFY_CUDA_ARCHS"] = expected_cuda_archs
        _run(
            [python, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(source)],
            env=env,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("wheelhouse"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--source-url",
        default=os.environ.get("SPARK_COMFY_KITCHEN_SOURCE", KITCHEN_SOURCE),
    )
    parser.add_argument(
        "--kitchen-base",
        choices=tuple(KITCHEN_TARGETS),
        default=DEFAULT_KITCHEN_BASE,
        help="upstream comfy-kitchen version required by the target ComfyUI release",
    )
    parser.add_argument(
        "--architecture",
        choices=tuple(ARCHITECTURES),
        default=DEFAULT_ARCHITECTURE,
        help="GPU architecture encoded into the comfy-kitchen Spark wheel",
    )
    parser.add_argument(
        "--cuda-tag",
        default=DEFAULT_CUDA_TAG,
        help="CUDA toolchain identity encoded into the wheel version (for example cu130)",
    )
    parser.add_argument(
        "--cuda-archs",
        help="advanced CMake CUDA architecture override (for example 120a on CUDA 12.8)",
    )
    parser.add_argument("--install", action="store_true", help="install instead of building a wheel")
    args = parser.parse_args(argv)
    node_root = Path(__file__).resolve().parent
    if (node_root / "patches").is_dir():
        pass
    else:
        node_root = node_root.parent
    if args.install:
        install_from_source(
            node_root,
            python=args.python,
            source_url=args.source_url,
            kitchen_base=args.kitchen_base,
            architecture=args.architecture,
            cuda_tag=args.cuda_tag,
            cuda_archs=args.cuda_archs,
        )
    else:
        for wheel in build_wheel(
            node_root,
            args.output_dir.resolve(),
            python=args.python,
            source_url=args.source_url,
            kitchen_base=args.kitchen_base,
            architecture=args.architecture,
            cuda_tag=args.cuda_tag,
            cuda_archs=args.cuda_archs,
        ):
            print(wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
