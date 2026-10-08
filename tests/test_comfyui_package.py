from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_package_is_small_complete_and_importable(tmp_path):
    builder = _load("build_comfyui_package", ROOT / "tools/build_comfyui_package.py")
    package = builder.assemble(
        tmp_path,
        version="0.1.7",
        publisher_id="test-publisher",
    )
    assert (package / "__init__.py").is_file()
    assert (package / "comfyui_backend.py").is_file()
    assert (package / "h3_sparse_attention/landmark_tree_v2.py").is_file()
    assert (package / "patches/comfy-kitchen-spark-v0.2.36.patch").is_file()
    assert not list(package.rglob("__pycache__"))
    assert not (package / "assets").exists()
    assert sum(path.stat().st_size for path in package.rglob("*") if path.is_file()) < 3_000_000

    metadata = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == "0.1.7"
    assert metadata["tool"]["comfy"]["PublisherId"] == "test-publisher"
    assert metadata["tool"]["comfy"]["requires-comfyui"] == ">=0.30.0"

    probe = """
import importlib.util
from pathlib import Path
import sys
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    'comfyui_spark_h3_package', root / '__init__.py',
    submodule_search_locations=[str(root)],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'MiniMaxH3SparkAttentionSM120' in module.NODE_CLASS_MAPPINGS
reblock = __import__(
    'comfyui_spark_h3_package.comfyui_reblock_plan', fromlist=['unused']
)
assert reblock.__package__ == 'comfyui_spark_h3_package'
"""
    subprocess.run([sys.executable, "-I", "-c", probe, str(package)], check=True)


def test_standalone_zip_has_one_installable_root(tmp_path):
    builder = _load("build_comfyui_package_zip", ROOT / "tools/build_comfyui_package.py")
    package = builder.assemble(tmp_path, version="0.1.0", publisher_id="test")
    archive = builder.make_zip(package, tmp_path, version="0.1.0")
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
    assert names
    assert all(name.startswith("ComfyUI-Spark-H3/") for name in names)
    assert "ComfyUI-Spark-H3/install.py" in names
    assert "ComfyUI-Spark-H3/pyproject.toml" in names


def test_kernel_builder_applies_distinct_local_version(tmp_path):
    kernel_builder = _load("spark_kernel_builder", ROOT / "comfyui/kernel_builder.py")
    source = tmp_path / "kitchen"
    source.mkdir()
    pyproject = source / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.2.36"\n', encoding="utf-8")
    kernel_builder._set_local_version(source)
    assert 'version = "0.2.36+spark.h3.1"' in pyproject.read_text(encoding="utf-8")


def test_kernel_builder_recognizes_an_existing_release_wheel(tmp_path, monkeypatch):
    kernel_builder = _load("spark_kernel_builder_repeat", ROOT / "comfyui/kernel_builder.py")
    expected = tmp_path / "comfy_kitchen-0.2.36+spark.h3.1-cp312-abi3-linux_x86_64.whl"
    expected.touch()

    class TemporarySource:
        def __enter__(self):
            return str(tmp_path / "temporary")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(kernel_builder.tempfile, "TemporaryDirectory", lambda **_kwargs: TemporarySource())
    monkeypatch.setattr(kernel_builder, "prepare_source", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(kernel_builder, "_run", lambda *_args, **_kwargs: None)
    assert kernel_builder.build_wheel(ROOT, tmp_path) == [expected]


def test_installer_finds_compatible_local_wheel(tmp_path):
    installer = _load("spark_comfyui_installer", ROOT / "comfyui/install.py")
    try:
        from packaging.tags import sys_tags
    except ImportError:
        from pip._vendor.packaging.tags import sys_tags
    tag = next(iter(sys_tags()))
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    expected = wheelhouse / f"comfy_kitchen-0.2.36+spark.h3.1-{tag}.whl"
    expected.touch()
    assert installer._matching_local_wheel(tmp_path) == expected
