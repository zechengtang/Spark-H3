from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from types import ModuleType
import zipfile
from pathlib import Path

import tomllib
import pytest

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
    assert not (package / "h3_sparse_attention/spark_reweight_sm89.py").exists()
    assert not (package / "sol_attn").exists()
    assert (package / "patches/comfy-kitchen-spark-v0.2.36.patch").is_file()
    assert not list(package.rglob("__pycache__"))
    assert not (package / "assets").exists()
    assert sum(path.stat().st_size for path in package.rglob("*") if path.is_file()) < 3_000_000

    metadata = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == "0.1.7"
    assert metadata["tool"]["comfy"]["PublisherId"] == "test-publisher"
    assert metadata["tool"]["comfy"]["requires-comfyui"] == ">=0.38.0,<0.40.0"

    assert {
        "spark_h3_vdn8_14p4s_t2va.json",
        "spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json",
        "spark_h3_larryvrh_8step_lora_14p4s_t2va.json",
    } == {path.name for path in (package / "workflows").glob("*.json")}
    for workflow_path in (package / "workflows").glob("*.json"):
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        spark_nodes = [
            node
            for node in workflow["nodes"]
            if node.get("type") == "MiniMaxH3SparkAttentionSM120"
        ]
        assert len(spark_nodes) == 1
        spark = spark_nodes[0]
        assert spark["properties"]["cnr_id"] == "comfyui-spark-h3"
        assert spark["properties"]["ver"] == "0.1.7"
        assert spark["widgets_values_named"]["reblock_layout"] == "q_reuse_k"
        assert [entry["name"] for entry in spark["inputs"]] == [
            "model",
            "enabled",
            "steps",
            "warmup_mode",
            "warmup_ratio",
            "warmup_steps",
            "topk_mode",
            "topk_ratio",
            "topk_blocks",
            "dense_layers",
            "min_tokens",
            "strict",
            "tail_granularity",
            "reblock_layout",
        ]
        video_vae = next(
            node
            for node in workflow["nodes"]
            if node.get("type") == "VAELoader"
            and "video_vae" in node.get("widgets_values", [""])[0]
        )
        assert video_vae["widgets_values"] == [
            "minimax_h3_video_vae_fp16.safetensors"
        ]

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
assert 'MiniMaxH3SparkAttentionSM89' not in module.NODE_CLASS_MAPPINGS
reblock = __import__(
    'comfyui_spark_h3_package.comfyui_reblock_plan', fromlist=['unused']
)
assert reblock.__package__ == 'comfyui_spark_h3_package'
"""
    subprocess.run([sys.executable, "-I", "-c", probe, str(package)], check=True)


def test_standalone_zip_has_one_installable_root(tmp_path):
    builder = _load("build_comfyui_package_zip", ROOT / "tools/build_comfyui_package.py")
    wheels = tuple(
        tmp_path / f"comfy_kitchen-{base}+spark.h3.1-cp312-abi3-linux_x86_64.whl"
        for base in ("0.2.36", "0.2.37")
    )
    for wheel in wheels:
        wheel.touch()
    package = builder.assemble(
        tmp_path,
        version="0.1.0",
        publisher_id="test",
        kernel_wheels=wheels,
    )
    archive = builder.make_zip(package, tmp_path, version="0.1.0")
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
    assert names
    assert all(name.startswith("ComfyUI-Spark-H3/") for name in names)
    assert "ComfyUI-Spark-H3/install.py" in names
    assert "ComfyUI-Spark-H3/pyproject.toml" in names
    assert {
        "ComfyUI-Spark-H3/wheelhouse/" + wheel.name for wheel in wheels
    }.issubset(names)


def test_kernel_builder_applies_distinct_local_versions(tmp_path):
    kernel_builder = _load("spark_kernel_builder", ROOT / "comfyui/kernel_builder.py")
    for base in ("0.2.36", "0.2.37"):
        source = tmp_path / base
        source.mkdir()
        pyproject = source / "pyproject.toml"
        pyproject.write_text(
            f'[project]\nversion = "{base}"\n', encoding="utf-8"
        )
        kernel_builder._set_local_version(source, base)
        assert f'version = "{base}+spark.h3.1"' in pyproject.read_text(
            encoding="utf-8"
        )


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
    assert kernel_builder.build_wheel(
        ROOT, tmp_path, kitchen_base="0.2.36"
    ) == [expected]


def test_installer_selects_matching_base_from_local_wheels(tmp_path):
    installer = _load("spark_comfyui_installer", ROOT / "comfyui/install.py")
    try:
        from packaging.tags import sys_tags
    except ImportError:
        from pip._vendor.packaging.tags import sys_tags
    tag = next(iter(sys_tags()))
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheels = {
        base: wheelhouse / f"comfy_kitchen-{base}+spark.h3.1-{tag}.whl"
        for base in installer.SUPPORTED_KITCHEN_BASES
    }
    for wheel in wheels.values():
        wheel.touch()
    assert installer._matching_local_wheel(tmp_path, "0.2.36") == wheels["0.2.36"]
    assert installer._matching_local_wheel(tmp_path, "0.2.37") == wheels["0.2.37"]


def test_installer_resolves_supported_installed_base(monkeypatch):
    installer = _load("spark_comfyui_installer_base", ROOT / "comfyui/install.py")
    monkeypatch.setattr(installer.metadata, "version", lambda _name: "0.2.37")
    assert installer._resolve_kitchen_base() == "0.2.37"

    monkeypatch.setattr(installer.metadata, "version", lambda _name: "0.2.35")
    with pytest.raises(RuntimeError, match="ComfyUI 0.38.x"):
        installer._resolve_kitchen_base()


def test_installer_backend_probe_ignores_stale_parent_modules(tmp_path, monkeypatch):
    installer = _load("spark_comfyui_installer_probe", ROOT / "comfyui/install.py")
    package = tmp_path / "comfy_kitchen" / "backends"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cuda.py").write_text(
        "def spark_attn():\n    return None\n", encoding="utf-8"
    )
    dist_info = tmp_path / "comfy_kitchen-0.2.37+spark.h3.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: comfy-kitchen\n"
        "Version: 0.2.37+spark.h3.1\n",
        encoding="utf-8",
    )
    existing_pythonpath = installer.os.environ.get("PYTHONPATH")
    pythonpath = str(tmp_path)
    if existing_pythonpath:
        pythonpath += installer.os.pathsep + existing_pythonpath
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    # Model the first installer check having loaded comfy-kitchen 0.2.35 in
    # this process.  The fresh subprocess must see the updated package on disk
    # instead of these stale module objects.
    monkeypatch.setitem(sys.modules, "comfy_kitchen", ModuleType("comfy_kitchen"))
    monkeypatch.setitem(
        sys.modules, "comfy_kitchen.backends", ModuleType("comfy_kitchen.backends")
    )
    monkeypatch.setitem(
        sys.modules,
        "comfy_kitchen.backends.cuda",
        ModuleType("comfy_kitchen.backends.cuda"),
    )

    assert installer._backend_available("0.2.37")
    assert not installer._backend_available("0.2.36")


def test_installer_validation_does_not_reimport_cached_backend(monkeypatch, capsys):
    installer = _load("spark_comfyui_installer_validate", ROOT / "comfyui/install.py")
    monkeypatch.setattr(installer, "_backend_available", lambda _base: True)
    stale = ModuleType("comfy_kitchen.backends.cuda")
    monkeypatch.setitem(sys.modules, "comfy_kitchen.backends.cuda", stale)

    installer._validate_backend("0.2.37")

    assert "CUDA backend is ready" in capsys.readouterr().out
