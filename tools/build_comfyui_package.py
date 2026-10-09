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
DEFAULT_VERSION = "0.1.0"
PACKAGE_CNR_ID = "comfyui-spark-h3"
PACKAGE_SPARK_NODE = "MiniMaxH3SparkAttentionSM120"

ROOT_FILES = (
    "__init__.py",
    "comfyui_nodes.py",
    "comfyui_backend.py",
    "comfyui_reblock_plan.py",
    "comfyui_dmad.py",
    "LICENSE",
)
SOURCE_DIRECTORIES = ("h3_sparse_attention", "web")
WORKFLOWS = (
    "spark_h3_vdn8_14p4s_t2va.json",
    "spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json",
    "spark_h3_larryvrh_8step_lora_14p4s_t2va.json",
)
GENERATED_FILES = {
    "comfyui/install.py": "install.py",
    "comfyui/kernel_builder.py": "kernel_builder.py",
    "comfyui/INSTALL.zh-CN.md": "INSTALL.zh-CN.md",
    "comfyui/standalone/requirements.txt": "requirements.txt",
    "comfyui/standalone/README.md": "README.md",
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
) -> Path:
    """Create and return an unpacked, registry-compatible custom-node tree."""

    target = output_dir / PACKAGE_NAME
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for relative in ROOT_FILES:
        shutil.copy2(REPOSITORY_ROOT / relative, target / relative)
    for relative in SOURCE_DIRECTORIES:
        _copy_tree(REPOSITORY_ROOT / relative, target / relative)
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
            node["type"] = PACKAGE_SPARK_NODE
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


def make_zip(package_dir: Path, output_dir: Path, *, version: str) -> Path:
    archive = output_dir / f"{PACKAGE_NAME}-{version}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(package_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(PACKAGE_NAME) / path.relative_to(package_dir)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = stat.S_IFREG | (0o755 if path.name in {"install.py", "kernel_builder.py"} else 0o644)
            info.external_attr = mode << 16
            bundle.writestr(info, path.read_bytes())
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/comfyui"))
    parser.add_argument("--version", default=DEFAULT_VERSION)
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
    )
    print(package)
    if not args.no_zip:
        print(make_zip(package, output, version=args.version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
