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
    assert not (package / "h3_sparse_attention/processor.py").exists()
    assert not (package / "h3_sparse_attention/spark_integration.py").exists()
    assert not list(package.rglob("spark_reweight_sm*.py"))
    assert not (package / "sol_attn").exists()
    assert (package / "patches/comfy-kitchen-spark-v0.2.36.patch").is_file()
    assert not list(package.rglob("__pycache__"))
    assert (package / "assets/adaln_curve_projection.safetensors").is_file()
    assert (package / "convert_larryvrh_lora_comfyui.py").is_file()
    assert sum(path.stat().st_size for path in package.rglob("*") if path.is_file()) < 3_000_000

    metadata = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == "0.1.7"
    assert metadata["tool"]["comfy"]["PublisherId"] == "test-publisher"
    assert metadata["tool"]["comfy"]["requires-comfyui"] == ">=0.38.0,<0.40.0"

    assert {
        "spark_h3_vdn8_14p4s_t2va.json",
        "spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json",
        "spark_h3_larryvrh_8step_lora_14p4s_t2va.json",
        "spark_h3_dmad_4step_lora_5p2s_t2va.json",
        "spark_h3_lbh_official_lightx2v_4step_5p2s_i2va.json",
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
        assert spark["widgets_values_named"]["topk_mode"] == "topk_ratio"
        assert spark["widgets_values_named"]["dense_layers"] == 0
        expected_topk = (
            0.1
            if workflow_path.name
            == "spark_h3_lbh_official_lightx2v_4step_5p2s_i2va.json"
            else 0.2
        )
        assert spark["widgets_values_named"]["topk_ratio"] == expected_topk
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
        larry_nodes = [node for node in workflow["nodes"] if node.get("id") == 142]
        if workflow_path.name == "spark_h3_larryvrh_8step_lora_14p4s_t2va.json":
            assert len(larry_nodes) == 1
            assert larry_nodes[0]["type"] == "LoraLoaderModelOnly"
            assert larry_nodes[0]["widgets_values_named"] == {
                "lora_name": "minimax_h3_turbo_v4_step600_ema_comfyui_curve_bf16.safetensors",
                "strength_model": 1.0,
            }
            sampler = next(node for node in workflow["nodes"] if node.get("id") == 123)
            assert sampler["type"] == "KSamplerSelect"
            assert sampler["widgets_values_named"]["sampler_name"] == "euler"

        dmad_nodes = [
            node for node in workflow["nodes"]
            if node.get("type") == "MiniMaxH3DMADSampler"
        ]
        lbh_nodes = [
            node for node in workflow["nodes"]
            if node.get("type") == "MinimaxH3LatentUpscaler3D"
        ]
        if workflow_path.name == "spark_h3_dmad_4step_lora_5p2s_t2va.json":
            assert len(dmad_nodes) == 1
        else:
            assert not dmad_nodes
        if (
            workflow_path.name
            == "spark_h3_lbh_official_lightx2v_4step_5p2s_i2va.json"
        ):
            assert len(lbh_nodes) == 1
            assert lbh_nodes[0]["widgets_values"][0] == (
                "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
            )
            assert lbh_nodes[0]["widgets_values"] == [
                "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
                "target dimensions",
                1344,
                768,
                32,
                False,
                True,
                "cuda",
                "fp16",
            ]
            assert lbh_nodes[0]["widgets_values_named"] == {
                "model_name": "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
                "mode": "target dimensions",
                "mode.width": 1344,
                "mode.height": 768,
                "align": 32,
                "enable_temporal_chunking": False,
                "force_unload": True,
                "device": "cuda",
                "precision": "fp16",
            }
            assert spark["widgets_values_named"]["steps"] == 3
            assert spark["widgets_values_named"]["warmup_mode"] == "warmup_steps"
            assert spark["widgets_values_named"]["warmup_steps"] == 0
            assert spark["widgets_values_named"]["topk_ratio"] == 0.1
            assert any(
                node.get("type") == "LTXVSeparateAVLatent"
                for node in workflow["nodes"]
            )
            assert any(
                node.get("type") == "LTXVConcatAVLatent"
                for node in workflow["nodes"]
            )
            split = next(
                node for node in workflow["nodes"]
                if node.get("type") == "SplitSigmas"
            )
            assert split["widgets_values"] == [4]
            high_sigmas = next(
                node for node in workflow["nodes"]
                if node.get("type") == "ManualSigmas"
                and node.get("title") == "3 step Sigmas"
            )
            assert high_sigmas["widgets_values"] == [
                "0.9035, 0.6316, 0.3158, 0.0000"
            ]
            lora = next(
                node for node in workflow["nodes"]
                if node.get("type") == "LoraLoaderModelOnly"
            )
            assert "lightx2v_turbo_4step_v0.1" in lora["widgets_values"][0]
        else:
            assert not lbh_nodes

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
assert 'MiniMaxH3SparkAttentionSM89' in module.NODE_CLASS_MAPPINGS
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


def test_sm89_package_rewrites_workflows_and_archive_name(tmp_path):
    builder = _load("build_comfyui_package_sm89", ROOT / "tools/build_comfyui_package.py")
    package = builder.assemble(
        tmp_path,
        version="0.1.7",
        publisher_id="test",
        architecture="sm89",
    )
    for workflow_path in (package / "workflows").glob("*.json"):
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        spark = next(
            node
            for node in workflow["nodes"]
            if node.get("type") == "MiniMaxH3SparkAttentionSM89"
        )
        assert spark["properties"]["ver"] == "0.1.7"
    archive = builder.make_zip(
        package, tmp_path, version="0.1.7", architecture="sm89"
    )
    assert archive.name == "ComfyUI-Spark-H3-0.1.7-sm89.zip"
    tagged = builder.make_zip(
        package,
        tmp_path,
        version="0.1.7",
        architecture="sm89",
        cuda_tag="cu130",
    )
    assert tagged.name == "ComfyUI-Spark-H3-0.1.7-sm89-cu130.zip"


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

        sm89 = tmp_path / f"{base}-sm89"
        sm89.mkdir()
        sm89_pyproject = sm89 / "pyproject.toml"
        sm89_pyproject.write_text(
            f'[project]\nversion = "{base}"\n', encoding="utf-8"
        )
        kernel_builder._set_local_version(sm89, base, "sm89")
        assert (
            f'version = "{base}+spark.h3.sm89.1"'
            in sm89_pyproject.read_text(encoding="utf-8")
        )

    assert kernel_builder.ARCHITECTURES["sm89"]["cuda_archs"] == "89"
    assert kernel_builder.ARCHITECTURES["sm120"]["cuda_archs"] == "120f"
    assert kernel_builder.architecture_for_capability((8, 9)) == "sm89"
    assert kernel_builder.architecture_for_capability((12, 0)) == "sm120"


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

    sm89 = wheelhouse / f"comfy_kitchen-0.2.36+spark.h3.sm89.1-{tag}.whl"
    sm89.touch()
    assert installer._matching_local_wheel(tmp_path, "0.2.36", "sm89") == sm89
    assert not installer._wheel_matches(sm89.name, "0.2.36", "sm120")
    assert not installer._wheel_matches(wheels["0.2.36"].name, "0.2.36", "sm89")


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
    monkeypatch.setattr(
        installer, "_backend_available", lambda _base, _architecture: True
    )
    stale = ModuleType("comfy_kitchen.backends.cuda")
    monkeypatch.setitem(sys.modules, "comfy_kitchen.backends.cuda", stale)

    installer._validate_backend("0.2.37")

    assert "CUDA backend is ready" in capsys.readouterr().out
