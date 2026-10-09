"""Assemble the independent ComfyUI-Spark-H3 custom-node package."""
from __future__ import annotations

import argparse
import json
import shutil
import stat
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ComfyUI-Spark-H3"
PACKAGE_BUILD_IDENTITY = "spark_h3_build.json"
DEFAULT_VERSION = "0.1.1"
RELEASE_CUDA_TAGS = ("cu130",)
EXPERIMENTAL_CUDA_TAGS = ("cu128", "cu129")
KNOWN_CUDA_TAGS = EXPERIMENTAL_CUDA_TAGS + RELEASE_CUDA_TAGS
PACKAGE_PLATFORMS = {
    "linux-x86_64": ("linux_x86_64", "manylinux"),
    "windows-x86_64": ("win_amd64",),
}
PACKAGE_CNR_ID = "comfyui-spark-h3"
PACKAGE_SPARK_NODE = "MiniMaxH3SparkAttentionSM120"
PACKAGE_SPARK_NODES = {
    "sm89": "MiniMaxH3SparkAttentionSM89",
    "sm120": PACKAGE_SPARK_NODE,
}
BACKEND_LOCAL_VERSION_PREFIXES = {
    "sm89": "spark.h3.sm89",
    "sm120": "spark.h3.sm120",
}

ROOT_FILES = (
    "__init__.py",
    "comfyui_nodes.py",
    "comfyui_backend.py",
    "comfyui_reblock_plan.py",
    "comfyui_dmad.py",
    "LICENSE",
)
SOURCE_DIRECTORIES = ("web",)
REBLOCK_RUNTIME_FILES = (
    "landmark_direction.py",
    "landmark_initial_order.py",
    "landmark_projection.py",
    "landmark_tree_clustering.py",
    "landmark_tree_triton.py",
    "landmark_tree_v2.py",
    "landmark_tree_v2_triton.py",
    "landmark_v2_cosine.py",
    "landmark_v2_cosine_fast.py",
    "landmark_v2_cosine_triton.py",
    "landmark_v2_euclidean.py",
    "landmark_v2_fused_node.py",
    "landmark_v2_max.py",
    "landmark_v2_order.py",
    "landmark_v2_route.py",
    "landmark_v2_terminal.py",
    "mahalanobis_kmeans.py",
    "reblock_hierarchy.py",
)
WORKFLOWS = (
    "spark_h3_vdn8_14p4s_t2va.json",
    "spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json",
    "spark_h3_larryvrh_8step_lora_14p4s_t2va.json",
    "spark_h3_dmad_4step_lora_5p2s_t2va.json",
    "spark_h3_lbh_official_lightx2v_4step_5p2s_i2va.json",
)
GENERATED_FILES = {
    "comfyui/install.py": "install.py",
    "comfyui/kernel_builder.py": "kernel_builder.py",
    "comfyui/INSTALL.zh-CN.md": "INSTALL.zh-CN.md",
    "comfyui/standalone/requirements.txt": "requirements.txt",
    "comfyui/standalone/README.md": "README.md",
    "tools/convert_larryvrh_lora_comfyui.py": "convert_larryvrh_lora_comfyui.py",
    "comfyui/assets/adaln_curve_projection.safetensors": (
        "assets/adaln_curve_projection.safetensors"
    ),
    "comfyui/patches/comfy-kitchen-spark-v0.2.36.patch": (
        "patches/comfy-kitchen-spark-v0.2.36.patch"
    ),
}


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name == "__pycache__"
        or name.startswith(".")
        or "sm89" in name.lower()
        or name.endswith((".pyc", ".pyo", ".so"))
    }


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, ignore=_ignore)


def assemble(
    output_dir: Path,
    *,
    version: str,
    publisher_id: str,
    kernel_wheels: tuple[Path, ...] = (),
    architecture: str = "sm120",
    cuda_tag: str = "cu130",
    platform_tag: str,
) -> Path:
    """Create and return an unpacked, registry-compatible custom-node tree."""

    _validate_platform_tag(platform_tag)
    try:
        package_spark_node = PACKAGE_SPARK_NODES[architecture]
    except KeyError as error:
        raise ValueError(
            f"unsupported package architecture {architecture!r}; expected one of "
            f"{', '.join(PACKAGE_SPARK_NODES)}"
        ) from error

    target = output_dir / PACKAGE_NAME
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for relative in ROOT_FILES:
        shutil.copy2(REPOSITORY_ROOT / relative, target / relative)
    for relative in SOURCE_DIRECTORIES:
        _copy_tree(REPOSITORY_ROOT / relative, target / relative)
    reblock_runtime = target / "h3_sparse_attention"
    reblock_runtime.mkdir()
    (reblock_runtime / "__init__.py").write_text(
        '"""Low-level reblock planner primitives vendored for ComfyUI."""\n',
        encoding="utf-8",
    )
    for name in REBLOCK_RUNTIME_FILES:
        shutil.copy2(REPOSITORY_ROOT / "h3_sparse_attention" / name, reblock_runtime / name)
    for source_name, target_name in GENERATED_FILES.items():
        destination = target / target_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / source_name, destination)
    workflows = target / "workflows"
    workflows.mkdir()
    for name in WORKFLOWS:
        source = REPOSITORY_ROOT / "workflows" / name
        workflow = json.loads(source.read_text(encoding="utf-8"))
        for node in workflow.get("nodes", ()):
            if node.get("type") != "MiniMaxH3SparkAttentionSM120":
                continue
            node["type"] = package_spark_node
            properties = node.setdefault("properties", {})
            properties["cnr_id"] = PACKAGE_CNR_ID
            properties["ver"] = version
        (workflows / name).write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if kernel_wheels:
        wheelhouse = target / "wheelhouse"
        wheelhouse.mkdir()
        for wheel in kernel_wheels:
            if wheel.suffix != ".whl" or not wheel.is_file():
                raise ValueError(f"kernel wheel does not exist or is not a .whl: {wheel}")
            _validate_kernel_wheel(
                wheel,
                architecture=architecture,
                cuda_tag=cuda_tag,
                platform_tag=platform_tag,
            )
            shutil.copy2(wheel, wheelhouse / wheel.name)

    (target / PACKAGE_BUILD_IDENTITY).write_text(
        json.dumps(
            {
                "architecture": architecture,
                "cuda_tag": cuda_tag,
                "platform": platform_tag,
                "kernel_wheels": sorted(wheel.name for wheel in kernel_wheels),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    template = (REPOSITORY_ROOT / "comfyui/standalone/pyproject.toml.in").read_text(
        encoding="utf-8"
    )
    (target / "pyproject.toml").write_text(
        template.replace("@VERSION@", version).replace("@PUBLISHER_ID@", publisher_id),
        encoding="utf-8",
    )
    return target


def make_zip(
    package_dir: Path,
    output_dir: Path,
    *,
    version: str,
    architecture: str = "sm120",
    cuda_tag: str | None = None,
    experimental: bool = False,
    platform_tag: str,
) -> Path:
    validate_cuda_tag(cuda_tag, experimental=experimental)
    _validate_platform_tag(platform_tag)
    identity_path = package_dir / PACKAGE_BUILD_IDENTITY
    if not identity_path.is_file():
        raise ValueError(f"package build identity is missing: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("architecture") != architecture:
        raise ValueError(
            f"package architecture {identity.get('architecture')!r} does not match "
            f"archive architecture {architecture!r}"
        )
    if identity.get("platform") != platform_tag:
        raise ValueError(
            f"package platform {identity.get('platform')!r} does not match "
            f"archive platform {platform_tag!r}"
        )
    if cuda_tag is not None and identity.get("cuda_tag") != cuda_tag:
        raise ValueError(
            f"package CUDA tag {identity.get('cuda_tag')!r} does not match "
            f"archive CUDA tag {cuda_tag!r}"
        )
    suffix = "-experimental" if experimental else ""
    suffix += f"-{platform_tag}"
    suffix += f"-{architecture}"
    if cuda_tag is not None:
        suffix += f"-{cuda_tag}"
    archive = output_dir / f"{PACKAGE_NAME}-{version}{suffix}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(package_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(PACKAGE_NAME) / path.relative_to(package_dir)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = stat.S_IFREG | (0o755 if path.name in {
                "install.py", "kernel_builder.py", "convert_larryvrh_lora_comfyui.py"
            } else 0o644)
            info.external_attr = mode << 16
            bundle.writestr(info, path.read_bytes())
    return archive


def validate_cuda_tag(cuda_tag: str | None, *, experimental: bool = False) -> None:
    """Keep experimental CUDA archives out of the default release path."""

    if cuda_tag in EXPERIMENTAL_CUDA_TAGS and not experimental:
        raise ValueError(
            f"{cuda_tag} is an experimental build target, not a release target; "
            "pass --experimental-cuda to build a local diagnostic archive"
        )


def _validate_platform_tag(platform_tag: str) -> None:
    if platform_tag not in PACKAGE_PLATFORMS:
        raise ValueError(
            f"unsupported package platform {platform_tag!r}; expected one of "
            f"{', '.join(PACKAGE_PLATFORMS)}"
        )


def _wheel_platform_matches(wheel_platform: str, platform_tag: str) -> bool:
    if platform_tag == "linux-x86_64":
        return wheel_platform == "linux_x86_64" or (
            wheel_platform.startswith("manylinux")
            and wheel_platform.endswith("_x86_64")
        )
    if platform_tag == "windows-x86_64":
        return wheel_platform == "win_amd64"
    _validate_platform_tag(platform_tag)
    return False


def _validate_kernel_wheel(
    wheel: Path,
    *,
    architecture: str,
    cuda_tag: str,
    platform_tag: str,
) -> None:
    try:
        from packaging.utils import parse_wheel_filename
    except ImportError:
        from pip._vendor.packaging.utils import parse_wheel_filename
    try:
        distribution, version, _, tags = parse_wheel_filename(wheel.name)
    except ValueError as error:
        raise ValueError(f"invalid kernel wheel filename: {wheel.name}") from error
    expected_local = f"{BACKEND_LOCAL_VERSION_PREFIXES[architecture]}.{cuda_tag}.1"
    if distribution != "comfy-kitchen" or version.local != expected_local:
        raise ValueError(
            f"kernel wheel {wheel.name!r} does not match {architecture}/{cuda_tag}; "
            f"expected local version +{expected_local}"
        )
    wheel_platforms = {tag.platform for tag in tags}
    if not any(
        _wheel_platform_matches(wheel_platform, platform_tag)
        for wheel_platform in wheel_platforms
    ):
        raise ValueError(
            f"kernel wheel {wheel.name!r} does not match platform {platform_tag}; "
            f"found wheel platforms {', '.join(sorted(wheel_platforms))}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/comfyui"))
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--architecture",
        choices=tuple(PACKAGE_SPARK_NODES),
        default="sm120",
        help="architecture used by bundled workflows and backend wheels",
    )
    parser.add_argument(
        "--cuda-tag",
        choices=KNOWN_CUDA_TAGS,
        help="append the CUDA toolchain tag to the release archive name",
    )
    parser.add_argument(
        "--platform",
        dest="platform_tag",
        choices=tuple(PACKAGE_PLATFORMS),
        required=True,
        help="target OS/CPU identity encoded into the archive and build manifest",
    )
    parser.add_argument(
        "--experimental-cuda",
        action="store_true",
        help="allow a diagnostic CUDA target excluded from the release plan",
    )
    parser.add_argument(
        "--publisher-id",
        required=True,
        help="existing Comfy Registry publisher id; never inferred from a GitHub username",
    )
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--kernel-wheel",
        type=Path,
        action="append",
        default=[],
        help="bundle a prebuilt backend wheel; may be repeated",
    )
    args = parser.parse_args(argv)
    try:
        validate_cuda_tag(args.cuda_tag, experimental=args.experimental_cuda)
    except ValueError as error:
        parser.error(str(error))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package = assemble(
        output,
        version=args.version,
        publisher_id=args.publisher_id,
        kernel_wheels=tuple(path.resolve() for path in args.kernel_wheel),
        architecture=args.architecture,
        cuda_tag=args.cuda_tag or "cu130",
        platform_tag=args.platform_tag,
    )
    print(package)
    if not args.no_zip:
        print(
            make_zip(
                package,
                output,
                version=args.version,
                architecture=args.architecture,
                cuda_tag=args.cuda_tag,
                experimental=args.experimental_cuda,
                platform_tag=args.platform_tag,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
