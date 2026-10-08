"""Build the pinned Spark-enabled comfy-kitchen backend.

This module is intentionally usable on Linux and Windows.  The custom-node
installer imports it for its source fallback, while maintainers use the CLI to
produce release wheels.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

KITCHEN_SOURCE = "https://github.com/Comfy-Org/comfy-kitchen.git"
KITCHEN_TAG = "v0.2.36"
KITCHEN_REVISION = "888b13e2c0e721f6576fe351a2ad79894b1c451f"
KITCHEN_BASE_VERSION = "0.2.36"
KITCHEN_SPARK_VERSION = "0.2.36+spark.h3.1"
CUDA_ARCHS = "120f"


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


def _set_local_version(source: Path) -> None:
    pyproject = source / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    old = f'version = "{KITCHEN_BASE_VERSION}"'
    new = f'version = "{KITCHEN_SPARK_VERSION}"'
    if old not in text:
        raise RuntimeError(f"cannot find {old!r} in the pinned comfy-kitchen source")
    pyproject.write_text(text.replace(old, new, 1), encoding="utf-8")


def prepare_source(node_root: Path, destination: Path, *, source_url: str) -> Path:
    """Clone, pin, patch, and version the backend in ``destination``."""

    if destination.exists():
        raise FileExistsError(f"build destination already exists: {destination}")
    _run([
        "git", "clone", "--quiet", "--depth", "1", "--branch", KITCHEN_TAG,
        "--recursive", source_url, str(destination),
    ])
    _run(["git", "checkout", "--quiet", KITCHEN_REVISION], cwd=destination)
    _run(
        ["git", "submodule", "update", "--init", "--recursive", "--quiet"],
        cwd=destination,
    )
    patch = _patch_path(node_root.resolve())
    _run(["git", "apply", "--check", str(patch)], cwd=destination)
    _run(["git", "apply", str(patch)], cwd=destination)
    _set_local_version(destination)
    return destination


def build_wheel(
    node_root: Path,
    output_dir: Path,
    *,
    python: str = sys.executable,
    source_url: str = KITCHEN_SOURCE,
) -> list[Path]:
    """Build an SM120 wheel and return the newly-created wheel paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spark-comfy-kitchen-") as temporary:
        source = prepare_source(node_root, Path(temporary) / "comfy-kitchen", source_url=source_url)
        env = os.environ.copy()
        env.setdefault("COMFY_CUDA_ARCHS", CUDA_ARCHS)
        _run(
            [python, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(output_dir), "."],
            cwd=source,
            env=env,
        )
    wheels = sorted(
        output_dir.glob(f"comfy_kitchen-{KITCHEN_SPARK_VERSION}-*.whl")
    )
    if not wheels:
        raise RuntimeError("comfy-kitchen build completed without producing a wheel")
    return wheels


def install_from_source(
    node_root: Path,
    *,
    python: str = sys.executable,
    source_url: str = KITCHEN_SOURCE,
) -> None:
    """Compile and install the pinned backend into ``python``."""

    with tempfile.TemporaryDirectory(prefix="spark-comfy-kitchen-") as temporary:
        source = prepare_source(node_root, Path(temporary) / "comfy-kitchen", source_url=source_url)
        env = os.environ.copy()
        env.setdefault("COMFY_CUDA_ARCHS", CUDA_ARCHS)
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
    parser.add_argument("--install", action="store_true", help="install instead of building a wheel")
    args = parser.parse_args(argv)
    node_root = Path(__file__).resolve().parent
    if (node_root / "patches").is_dir():
        pass
    else:
        node_root = node_root.parent
    if args.install:
        install_from_source(node_root, python=args.python, source_url=args.source_url)
    else:
        for wheel in build_wheel(
            node_root, args.output_dir.resolve(), python=args.python, source_url=args.source_url
        ):
            print(wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
