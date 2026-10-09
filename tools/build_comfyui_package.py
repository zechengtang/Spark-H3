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
DEFAULT_VERSION = "0.1.1"
PACKAGE_CNR_ID = "comfyui-spark-h3"
PACKAGE_SPARK_NODE = "MiniMaxH3SparkAttentionSM120"
PACKAGE_SPARK_NODES = {
    "sm89": "MiniMaxH3SparkAttentionSM89",
    "sm120": PACKAGE_SPARK_NODE,
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
) -> Path:
    """Create and return an unpacked, registry-compatible custom-node tree."""

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
            shutil.copy2(wheel, wheelhouse / wheel.name)

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
) -> Path:
    suffix = "" if architecture == "sm120" else f"-{architecture}"
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
        choices=("cu128", "cu129", "cu130"),
        help="append the CUDA toolchain tag to the release archive name",
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
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package = assemble(
        output,
        version=args.version,
        publisher_id=args.publisher_id,
        kernel_wheels=tuple(path.resolve() for path in args.kernel_wheel),
        architecture=args.architecture,
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
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
