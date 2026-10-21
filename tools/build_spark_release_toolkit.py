"""Generate a Spark-H3 build and validation toolkit for Linux or Windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from build_comfyui_package import DEFAULT_VERSION, assemble, _validate_kernel_wheel

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).resolve().parent / "spark_release_toolkit"
CAPABILITIES = {"sm86": (8, 6), "sm89": (8, 9), "sm120": (12, 0)}
PLATFORMS = {"linux": "linux-x86_64", "windows": "windows-x86_64"}
REQUIRED_TRITON_MODULES = (
    "landmark_projection.py", "landmark_v2_cosine_fast.py",
    "landmark_v2_cosine_triton.py", "landmark_v2_euclidean.py",
    "landmark_v2_fused_node.py", "landmark_v2_max.py", "landmark_v2_route.py",
)
TEMPLATE_FILES = (
    "README.md", "build.py", "validate.py",
    "config.py", "preflight.py",
    "validate_wheels.py", "run_lightx2v_three_variants.py",
    "lightx2v_workflow.json", "bundle.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(
    output: Path, *, architecture: str, publisher_id: str,
    wheels: tuple[Path, ...] = (), force: bool = False,
    platform_tag: str = "windows-x86_64",
) -> tuple[Path, Path]:
    if architecture not in CAPABILITIES:
        raise ValueError(f"unsupported architecture: {architecture}")
    if platform_tag not in PLATFORMS.values():
        raise ValueError(f"unsupported platform: {platform_tag}")
    if output.exists():
        if not force:
            raise FileExistsError(f"output exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    package = assemble(
        output, version=DEFAULT_VERSION, publisher_id=publisher_id,
        architecture=architecture, cuda_tag="cu130", platform_tag=platform_tag,
    )
    for name in REQUIRED_TRITON_MODULES:
        source = (package / "h3_sparse_attention" / name).read_text(encoding="utf-8")
        if "import triton" not in source:
            raise RuntimeError(f"missing Triton planner source: {name}")
    for name in TEMPLATE_FILES:
        shutil.copy2(TEMPLATES / name, output / name)
    for name in (("build.ps1", "validate.ps1") if platform_tag == "windows-x86_64"
                 else ("build.sh", "validate.sh")):
        shutil.copy2(TEMPLATES / name, output / name)
    # lightx2v_workflow.json is the maintained eight-step validation graph.
    wheelhouse = output / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "PUT_PLATFORM_WHEELS_HERE.txt").write_text(
        f"Copy architecture-matched comfy-kitchen {platform_tag} wheels here before validation.\n",
        encoding="utf-8",
    )
    for wheel in wheels:
        _validate_kernel_wheel(
            wheel, architecture=architecture, cuda_tag="cu130", platform_tag=platform_tag
        )
        shutil.copy2(wheel, wheelhouse / wheel.name)
    identity = {
        "architecture": architecture, "capability": CAPABILITIES[architecture],
        "cuda_tag": "cu130", "platform": platform_tag,
        "version": DEFAULT_VERSION, "publisher_id": publisher_id,
        "included_wheels": [wheel.name for wheel in wheels],
    }
    (output / "release.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    hashes = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    (output / "source-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, (Path(output.name) / path.relative_to(output)).as_posix())
    return output, archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(CAPABILITIES), required=True)
    parser.add_argument("--publisher-id", required=True)
    parser.add_argument("--platform", choices=tuple(PLATFORMS), default="windows")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wheel", type=Path, action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    system = args.platform
    output = args.output or ROOT / "dist" / f"spark-release-toolkit-{system}-{args.architecture}-cu130"
    directory, archive = generate(
        output.resolve(), architecture=args.architecture, publisher_id=args.publisher_id,
        wheels=tuple(path.resolve() for path in args.wheel), force=args.force,
        platform_tag=PLATFORMS[args.platform],
    )
    print(directory)
    print(archive)
    print(f"SHA256 {sha256(archive)}")


if __name__ == "__main__":
    main()
