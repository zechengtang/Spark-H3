from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path[:]
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


def test_standalone_package_is_small_complete_and_importable(tmp_path):
    builder = _load("build_comfyui_package", ROOT / "tools/build_comfyui_package.py")
    package = builder.assemble(
        tmp_path,
        version="0.1.7",
        publisher_id="test-publisher",
        platform_tag="linux-x86_64",
    )
    assert (package / "__init__.py").is_file()
    assert (package / "comfyui_backend.py").is_file()
    assert (package / "package_manager.py").is_file()
    assert "packaging>=24" in (package / "requirements.txt").read_text()
    mode_widgets = package / "web/spark_mode_widgets.js"
    assert mode_widgets.is_file()
    mode_widget_source = mode_widgets.read_text(encoding="utf-8")
    assert 'speed: { warmup_mode: "warmup_ratio", warmup_ratio: 0.0 }' in mode_widget_source
    assert 'fidelity: { warmup_mode: "warmup_ratio", warmup_ratio: 0.2 }' in mode_widget_source
    assert 'mode.value = "custom"' in mode_widget_source
    assert '"双采高分辨率二阶段：建议使用 mode=speed → 一键切换"' in mode_widget_source
    assert '"Dense warmup 加速：可使用 dense_backend=kitchen → 一键切换"' in mode_widget_source
    assert '"带视频条件的 Ref2VA：建议使用 sparse_scope=joint-video → 一键切换"' in mode_widget_source
    assert "{ serialize: false }" in mode_widget_source
    node_source = (package / "comfyui_nodes.py").read_text(encoding="utf-8")
    assert '"producer_chunk": (' in node_source
    assert 'range(0, tokens, producer_chunk)' in node_source
    assert not (package / "comfyui_ref2va.py").exists()
    assert (package / "h3_sparse_attention/landmark_tree_v2.py").is_file()
    assert not (package / "h3_sparse_attention/spark_reweight_sm89.py").exists()
    assert not (package / "h3_sparse_attention/processor.py").exists()
    assert not (package / "h3_sparse_attention/spark_integration.py").exists()
    assert not list(package.rglob("spark_reweight_sm*.py"))
    assert not (package / "sol_attn").exists()
    assert (package / "patches/comfy-kitchen-spark-v0.2.36.patch").is_file()
    assert (package / "patches/comfy-kitchen-spark-v0.2.39.patch").is_file()
    assert not list(package.rglob("__pycache__"))
    assert (package / "assets/adaln_curve_projection.safetensors").is_file()
    assert json.loads((package / "spark_h3_build.json").read_text()) == {
        "architecture": "sm120",
        "cuda_tag": "cu130",
        "kernel_wheels": [],
        "platform": "linux-x86_64",
    }
    assert (package / "convert_larryvrh_lora_comfyui.py").is_file()
    assert sum(path.stat().st_size for path in package.rglob("*") if path.is_file()) < 3_000_000

    metadata = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == "0.1.7"
    assert "Operating System :: OS Independent" in metadata["project"]["classifiers"]
    assert "Operating System :: POSIX :: Linux" not in metadata["project"]["classifiers"]
    assert metadata["tool"]["comfy"]["PublisherId"] == "test-publisher"
    assert metadata["tool"]["comfy"]["requires-comfyui"] == ">=0.38.0,<0.43.0"

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
        assert spark["widgets_values_named"]["dense_backend"] == "inherit"
        assert spark["widgets_values_named"]["dense_layers"] == 0
        expected_mode = (
            "speed"
            if workflow_path.name
            == "spark_h3_lbh_official_lightx2v_4step_5p2s_i2va.json"
            else "fidelity"
        )
        assert spark["widgets_values_named"]["mode"] == expected_mode
        expected_warmup_ratio = 0.0 if expected_mode == "speed" else 0.2
        assert spark["widgets_values_named"]["warmup_mode"] == "warmup_ratio"
        assert spark["widgets_values_named"]["warmup_ratio"] == expected_warmup_ratio
        assert spark["widgets_values_named"]["warmup_steps"] == {
            "spark_h3_vdn8_14p4s_t2va.json": 4,
            "spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json": 2,
            "spark_h3_larryvrh_8step_lora_14p4s_t2va.json": 2,
            "spark_h3_dmad_4step_lora_5p2s_t2va.json": 1,
            "spark_h3_lbh_official_lightx2v_4step_5p2s_i2va.json": 0,
        }[workflow_path.name]
        assert math.ceil(
            spark["widgets_values_named"]["steps"]
            * spark["widgets_values_named"]["warmup_ratio"]
        ) == {
            "spark_h3_vdn8_14p4s_t2va.json": 4,
            "spark_h3_lightx2v_768p_8step_lora_14p4s_t2va.json": 2,
            "spark_h3_larryvrh_8step_lora_14p4s_t2va.json": 2,
            "spark_h3_dmad_4step_lora_5p2s_t2va.json": 1,
            "spark_h3_lbh_official_lightx2v_4step_5p2s_i2va.json": 0,
        }[workflow_path.name]
        expected_topk = (
            0.1
            if workflow_path.name
            == "spark_h3_lbh_official_lightx2v_4step_5p2s_i2va.json"
            else 0.2
        )
        assert spark["widgets_values_named"]["topk_ratio"] == expected_topk
        assert [entry["name"] for entry in spark["inputs"]] == [
            "model",
            "mode",
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
            "dense_backend",
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
            assert spark["widgets_values_named"]["mode"] == "speed"
            assert spark["widgets_values_named"]["warmup_mode"] == "warmup_ratio"
            assert spark["widgets_values_named"]["warmup_ratio"] == 0.0
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
assert 'MiniMaxH3SolAttentionSM120' in module.NODE_CLASS_MAPPINGS
assert 'MiniMaxH3SparkAttentionSM86' not in module.NODE_CLASS_MAPPINGS
assert 'MiniMaxH3SparkAttentionSM89' not in module.NODE_CLASS_MAPPINGS
assert 'MiniMaxH3SparkAttentionSM86' not in module.NODE_DISPLAY_NAME_MAPPINGS
assert 'MiniMaxH3SparkAttentionSM89' not in module.NODE_DISPLAY_NAME_MAPPINGS
assert set(module.NODE_CLASS_MAPPINGS) == set(module.NODE_DISPLAY_NAME_MAPPINGS)
assert 'MiniMaxH3Ref2VA12FPS480P' not in module.NODE_CLASS_MAPPINGS
spark_inputs = module.NODE_CLASS_MAPPINGS[
    'MiniMaxH3SparkAttentionSM120'
].INPUT_TYPES()
assert spark_inputs['optional']['producer_chunk'][0] == 'INT'
assert spark_inputs['optional']['producer_chunk'][1]['default'] == 16384
assert spark_inputs['optional']['producer_chunk'][1]['step'] == 64
sol_inputs = module.NODE_CLASS_MAPPINGS['MiniMaxH3SolAttentionSM120'].INPUT_TYPES()
assert 'producer_chunk' not in sol_inputs['optional']
for workflow_path in (root / 'workflows').glob('*.json'):
    workflow = __import__('json').loads(workflow_path.read_text(encoding='utf-8'))
    spark_nodes = [
        node['type'] for node in workflow['nodes']
        if node.get('type', '').startswith('MiniMaxH3SparkAttention')
    ]
    assert spark_nodes == ['MiniMaxH3SparkAttentionSM120']
    assert spark_nodes[0] in module.NODE_CLASS_MAPPINGS
reblock = __import__(
    'comfyui_spark_h3_package.comfyui_reblock_plan', fromlist=['unused']
)
assert reblock.__package__ == 'comfyui_spark_h3_package'
"""
    subprocess.run([sys.executable, "-I", "-c", probe, str(package)], check=True)


def test_standalone_zip_has_one_installable_root(tmp_path):
    builder = _load("build_comfyui_package_zip", ROOT / "tools/build_comfyui_package.py")
    wheels = tuple(
        tmp_path
        / f"comfy_kitchen-{base}+spark.h3.sm120.cu130.1-cp312-abi3-linux_x86_64.whl"
        for base in ("0.2.36", "0.2.37", "0.2.39")
    )
    for wheel in wheels:
        wheel.touch()
    package = builder.assemble(
        tmp_path,
        version="0.1.0",
        publisher_id="test",
        kernel_wheels=wheels,
        platform_tag="linux-x86_64",
    )
    archive = builder.make_zip(
        package,
        tmp_path,
        version="0.1.0",
        platform_tag="linux-x86_64",
    )
    assert archive.name == "ComfyUI-Spark-H3-0.1.0-linux-x86_64-sm120.zip"
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
    assert names
    assert all(name.startswith("ComfyUI-Spark-H3/") for name in names)
    assert "ComfyUI-Spark-H3/install.py" in names
    assert "ComfyUI-Spark-H3/package_manager.py" in names
    assert "ComfyUI-Spark-H3/pyproject.toml" in names
    assert {
        "ComfyUI-Spark-H3/wheelhouse/" + wheel.name for wheel in wheels
    }.issubset(names)


def test_registry_package_routes_kitchen_wheels_to_releases(tmp_path):
    builder = _load(
        "build_comfyui_registry_package", ROOT / "tools/build_comfyui_package.py"
    )
    package = builder.assemble_registry(
        tmp_path / "output",
        version="0.1.3",
        publisher_id="zechengtang",
    )
    metadata = tomllib.loads((package / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "comfyui-spark-h3"
    assert metadata["project"]["version"] == "0.1.3"
    assert metadata["project"]["requires-python"] == ">=3.12"
    assert metadata["tool"]["comfy"]["PublisherId"] == "zechengtang"
    assert not (package / "wheelhouse").exists()
    identity = json.loads((package / "spark_h3_build.json").read_text())
    assert identity["distribution"] == "registry"
    assert identity["architectures"] == ["sm120", "sm86", "sm89"]
    assert identity["platforms"] == ["linux-x86_64", "windows-x86_64"]
    assert identity["kernel_wheels"] == []
    assert identity["backend_source"] == "github-release"
    assert identity["backend_release_tag"] == "comfyui-v0.1.5"
    assert (package / "package_manager.py").read_bytes() == (
        ROOT / "comfyui/package_manager.py"
    ).read_bytes()

    workflows = list((package / "workflows").glob("*.json"))
    assert len(workflows) == 15
    workflow_nodes = set()
    for workflow_path in workflows:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        workflow_nodes.update(
            node["type"]
            for node in workflow["nodes"]
            if node.get("type", "").startswith("MiniMaxH3SparkAttention")
        )
    assert workflow_nodes == {
        "MiniMaxH3SparkAttentionSM86",
        "MiniMaxH3SparkAttentionSM89",
        "MiniMaxH3SparkAttentionSM120",
    }

    source = (package / "__init__.py").read_text(encoding="utf-8")
    assert "_HIDDEN_ARCHITECTURE_NODES" not in source
    archive = builder.make_registry_zip(
        package, tmp_path / "output", version="0.1.3"
    )
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
    assert "pyproject.toml" in names
    assert "package_manager.py" in names
    assert not any(name.startswith("ComfyUI-Spark-H3/") for name in names)
    for script in ("install.py", "kernel_builder.py"):
        subprocess.run(
            [sys.executable, str(package / script), "--help"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )


def test_installer_routes_release_archive_to_matching_kitchen_wheel(tmp_path, monkeypatch):
    installer = _load("spark_comfyui_installer_release_archive", ROOT / "comfyui/install.py")
    from packaging.tags import sys_tags

    wheel_tag = next(sys_tags())
    matching = f"comfy_kitchen-0.2.39+spark.h3.sm120.cu130.1-{wheel_tag}.whl"
    other = f"comfy_kitchen-0.2.37+spark.h3.sm120.cu130.1-{wheel_tag}.whl"
    archive_path = tmp_path / "ComfyUI-Spark-H3-0.1.5-linux-x86_64-sm120-cu130.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"ComfyUI-Spark-H3/wheelhouse/{matching}", b"matching-wheel")
        archive.writestr(f"ComfyUI-Spark-H3/wheelhouse/{other}", b"other-wheel")

    release = {"assets": [{"name": archive_path.name, "browser_download_url": archive_path.as_uri()}]}
    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(json.dumps(release).encode()))
    monkeypatch.setattr(installer, "_runtime_platform_tag", lambda: "linux-x86_64")
    url = installer._matching_release_wheel("https://example.invalid/release", "0.2.39", "sm120", "cu130")
    assert url == archive_path.as_uri()
    installed = []
    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *_args, **_kwargs: archive_path.open("rb"))
    monkeypatch.setattr(installer, "_install", lambda path, **_kwargs: installed.append((Path(path).name, Path(path).read_bytes())))
    installer._install_release_asset(url, "0.2.39", "sm120", "cu130")
    assert installed == [(matching, b"matching-wheel")]


def test_registry_installer_uses_pinned_backend_release(tmp_path, monkeypatch):
    installer = _load("spark_comfyui_installer_registry_route", ROOT / "comfyui/install.py")
    (tmp_path / "spark_h3_build.json").write_text(json.dumps({
        "distribution": "registry",
        "architectures": ["sm120"],
        "platforms": [installer._runtime_platform_tag()],
        "cuda_tag": "cu130",
        "kernel_wheels": [],
        "backend_release_tag": "comfyui-v0.1.5",
    }), encoding="utf-8")
    monkeypatch.setattr(installer, "_node_root", lambda: tmp_path)
    monkeypatch.setattr(installer, "_validate_runtime", lambda **_kwargs: (13, 0))
    monkeypatch.setattr(installer, "_cuda_capability", lambda: (12, 0))
    monkeypatch.setattr(installer, "_backend_available", lambda *_args: False)
    monkeypatch.setattr(installer, "_validate_backend", lambda *_args: None)
    calls = []
    monkeypatch.setattr(installer, "_matching_release_wheel", lambda *args: calls.append(args) or "https://example.invalid/release.zip")
    monkeypatch.setattr(installer, "_install_release_asset", lambda *args: calls.append(args))
    assert installer.main(["--skip-dependencies", "--kitchen-base", "0.2.39"]) == 0
    assert calls == [
        (installer.RELEASE_API_PREFIX + "comfyui-v0.1.5", "0.2.39", "sm120", "cu130"),
        ("https://example.invalid/release.zip", "0.2.39", "sm120", "cu130"),
    ]


def test_release_package_keeps_cu129_experimental():
    builder = _load(
        "build_comfyui_package_release_cuda", ROOT / "tools/build_comfyui_package.py"
    )
    assert builder.RELEASE_CUDA_TAGS == ("cu130",)
    assert "cu129" in builder.EXPERIMENTAL_CUDA_TAGS
    with pytest.raises(ValueError, match="experimental build target"):
        builder.validate_cuda_tag("cu129")
    builder.validate_cuda_tag("cu129", experimental=True)
    builder.validate_cuda_tag("cu130")


def test_release_package_rejects_wrong_cuda_or_architecture_wheel(tmp_path):
    builder = _load(
        "build_comfyui_package_wheel_identity", ROOT / "tools/build_comfyui_package.py"
    )
    cu129 = (
        tmp_path
        / "comfy_kitchen-0.2.37+spark.h3.sm120.cu129.1-cp312-abi3-linux_x86_64.whl"
    )
    cu129.touch()
    with pytest.raises(ValueError, match="does not match sm120/cu130"):
        builder.assemble(
            tmp_path / "wrong-cuda",
            version="0.1.7",
            publisher_id="test",
            kernel_wheels=(cu129,),
            architecture="sm120",
            cuda_tag="cu130",
            platform_tag="linux-x86_64",
        )

    cu129_package = builder.assemble(
        tmp_path / "cu129-package",
        version="0.1.7",
        publisher_id="test",
        architecture="sm120",
        cuda_tag="cu129",
        platform_tag="linux-x86_64",
    )
    with pytest.raises(ValueError, match="package CUDA tag 'cu129'"):
        builder.make_zip(
            cu129_package,
            tmp_path,
            version="0.1.7",
            architecture="sm120",
            cuda_tag="cu130",
            platform_tag="linux-x86_64",
        )

    sm89 = (
        tmp_path
        / "comfy_kitchen-0.2.37+spark.h3.sm89.cu130.1-cp312-abi3-linux_x86_64.whl"
    )
    sm89.touch()
    with pytest.raises(ValueError, match="does not match sm120/cu130"):
        builder.assemble(
            tmp_path / "wrong-architecture",
            version="0.1.7",
            publisher_id="test",
            kernel_wheels=(sm89,),
            architecture="sm120",
            cuda_tag="cu130",
            platform_tag="linux-x86_64",
        )


def test_release_package_isolates_linux_and_windows_wheels(tmp_path):
    builder = _load(
        "build_comfyui_package_platform_identity",
        ROOT / "tools/build_comfyui_package.py",
    )
    windows_wheel = (
        tmp_path
        / "comfy_kitchen-0.2.37+spark.h3.sm120.cu130.1-cp312-abi3-win_amd64.whl"
    )
    windows_wheel.touch()
    with pytest.raises(ValueError, match="does not match platform linux-x86_64"):
        builder.assemble(
            tmp_path / "wrong-platform",
            version="0.1.7",
            publisher_id="test",
            kernel_wheels=(windows_wheel,),
            architecture="sm120",
            cuda_tag="cu130",
            platform_tag="linux-x86_64",
        )

    windows_package = builder.assemble(
        tmp_path / "windows-package",
        version="0.1.7",
        publisher_id="test",
        kernel_wheels=(windows_wheel,),
        architecture="sm120",
        cuda_tag="cu130",
        platform_tag="windows-x86_64",
    )
    assert json.loads(
        (windows_package / "spark_h3_build.json").read_text(encoding="utf-8")
    )["platform"] == "windows-x86_64"
    windows_archive = builder.make_zip(
        windows_package,
        tmp_path,
        version="0.1.7",
        architecture="sm120",
        cuda_tag="cu130",
        platform_tag="windows-x86_64",
    )
    assert windows_archive.name == (
        "ComfyUI-Spark-H3-0.1.7-windows-x86_64-sm120-cu130.zip"
    )
    with pytest.raises(ValueError, match="package platform 'windows-x86_64'"):
        builder.make_zip(
            windows_package,
            tmp_path,
            version="0.1.7",
            architecture="sm120",
            cuda_tag="cu130",
            platform_tag="linux-x86_64",
        )

    assert builder._wheel_platform_matches(
        "manylinux_2_28_x86_64", "linux-x86_64"
    )
    assert not builder._wheel_platform_matches("win_amd64", "linux-x86_64")
    assert not builder._wheel_platform_matches(
        "linux_x86_64", "windows-x86_64"
    )


def test_sm89_package_rewrites_workflows_and_archive_name(tmp_path):
    builder = _load("build_comfyui_package_sm89", ROOT / "tools/build_comfyui_package.py")
    package = builder.assemble(
        tmp_path,
        version="0.1.7",
        publisher_id="test",
        architecture="sm89",
        platform_tag="linux-x86_64",
    )
    for workflow_path in (package / "workflows").glob("*.json"):
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        spark = next(
            node
            for node in workflow["nodes"]
            if node.get("type") == "MiniMaxH3SparkAttentionSM89"
        )
        assert spark["properties"]["ver"] == "0.1.7"
        assert "title" not in spark
    probe = """
import importlib.util
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    'comfyui_spark_h3_sm89_package', root / '__init__.py',
    submodule_search_locations=[str(root)],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'MiniMaxH3SparkAttentionSM89' in module.NODE_CLASS_MAPPINGS
assert 'MiniMaxH3SparkAttentionSM120' not in module.NODE_CLASS_MAPPINGS
assert 'MiniMaxH3SolAttentionSM120' not in module.NODE_CLASS_MAPPINGS
assert 'MiniMaxH3SparkAttentionSM120' not in module.NODE_DISPLAY_NAME_MAPPINGS
assert 'MiniMaxH3SolAttentionSM120' not in module.NODE_DISPLAY_NAME_MAPPINGS
assert set(module.NODE_CLASS_MAPPINGS) == set(module.NODE_DISPLAY_NAME_MAPPINGS)
for workflow_path in (root / 'workflows').glob('*.json'):
    workflow = json.loads(workflow_path.read_text(encoding='utf-8'))
    spark_nodes = [
        node['type'] for node in workflow['nodes']
        if node.get('type', '').startswith('MiniMaxH3SparkAttention')
    ]
    assert spark_nodes == ['MiniMaxH3SparkAttentionSM89']
    assert spark_nodes[0] in module.NODE_CLASS_MAPPINGS
"""
    subprocess.run([sys.executable, "-I", "-c", probe, str(package)], check=True)
    archive = builder.make_zip(
        package,
        tmp_path,
        version="0.1.7",
        architecture="sm89",
        platform_tag="linux-x86_64",
    )
    assert archive.name == "ComfyUI-Spark-H3-0.1.7-linux-x86_64-sm89.zip"
    tagged = builder.make_zip(
        package,
        tmp_path,
        version="0.1.7",
        architecture="sm89",
        cuda_tag="cu130",
        platform_tag="linux-x86_64",
    )
    assert tagged.name == "ComfyUI-Spark-H3-0.1.7-linux-x86_64-sm89-cu130.zip"
    experimental_package = builder.assemble(
        tmp_path / "experimental-sm89",
        version="0.1.7",
        publisher_id="test",
        architecture="sm89",
        cuda_tag="cu129",
        platform_tag="linux-x86_64",
    )
    experimental = builder.make_zip(
        experimental_package,
        tmp_path,
        version="0.1.7",
        architecture="sm89",
        cuda_tag="cu129",
        experimental=True,
        platform_tag="linux-x86_64",
    )
    assert experimental.name == (
        "ComfyUI-Spark-H3-0.1.7-experimental-linux-x86_64-sm89-cu129.zip"
    )


@pytest.mark.parametrize(
    ("architecture", "registered", "hidden"),
    (
        (
            "sm86",
            "MiniMaxH3SparkAttentionSM86",
            (
                "MiniMaxH3SparkAttentionSM89",
                "MiniMaxH3SparkAttentionSM120",
                "MiniMaxH3SolAttentionSM120",
            ),
        ),
        (
            "sm89",
            "MiniMaxH3SparkAttentionSM89",
            (
                "MiniMaxH3SparkAttentionSM86",
                "MiniMaxH3SparkAttentionSM120",
                "MiniMaxH3SolAttentionSM120",
            ),
        ),
        (
            "sm120",
            "MiniMaxH3SparkAttentionSM120",
            ("MiniMaxH3SparkAttentionSM86", "MiniMaxH3SparkAttentionSM89"),
        ),
    ),
)
def test_release_package_registers_only_target_architecture_nodes(
    tmp_path, architecture, registered, hidden
):
    builder = _load(
        f"build_comfyui_package_node_isolation_{architecture}",
        ROOT / "tools/build_comfyui_package.py",
    )
    package = builder.assemble(
        tmp_path / architecture,
        version="0.1.7",
        publisher_id="test",
        architecture=architecture,
        platform_tag="linux-x86_64",
    )
    probe = """
import importlib.util
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
registered = sys.argv[2]
hidden = json.loads(sys.argv[3])
spec = importlib.util.spec_from_file_location(
    f'comfyui_spark_h3_isolated_{registered}', root / '__init__.py',
    submodule_search_locations=[str(root)],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert registered in module.NODE_CLASS_MAPPINGS
assert registered in module.NODE_DISPLAY_NAME_MAPPINGS
assert all(name not in module.NODE_CLASS_MAPPINGS for name in hidden)
assert all(name not in module.NODE_DISPLAY_NAME_MAPPINGS for name in hidden)
for workflow_path in (root / 'workflows').glob('*.json'):
    workflow = json.loads(workflow_path.read_text(encoding='utf-8'))
    spark_nodes = [
        node for node in workflow['nodes']
        if node.get('type', '').startswith('MiniMaxH3SparkAttention')
    ]
    assert len(spark_nodes) == 1
    assert spark_nodes[0]['type'] == registered
    assert spark_nodes[0]['type'] in module.NODE_CLASS_MAPPINGS
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            probe,
            str(package),
            registered,
            json.dumps(hidden),
        ],
        check=True,
    )
    archive = builder.make_zip(
        package,
        tmp_path,
        version="0.1.7",
        architecture=architecture,
        cuda_tag="cu130",
        platform_tag="linux-x86_64",
    )
    assert archive.name == (
        f"ComfyUI-Spark-H3-0.1.7-linux-x86_64-{architecture}-cu130.zip"
    )


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
        assert f'version = "{base}+spark.h3.sm120.cu130.1"' in pyproject.read_text(
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
            f'version = "{base}+spark.h3.sm89.cu130.1"'
            in sm89_pyproject.read_text(encoding="utf-8")
        )

        sm86 = tmp_path / f"{base}-sm86"
        sm86.mkdir()
        sm86_pyproject = sm86 / "pyproject.toml"
        sm86_pyproject.write_text(
            f'[project]\nversion = "{base}"\n', encoding="utf-8"
        )
        kernel_builder._set_local_version(sm86, base, "sm86")
        assert (
            f'version = "{base}+spark.h3.sm86.cu130.1"'
            in sm86_pyproject.read_text(encoding="utf-8")
        )

    assert kernel_builder.ARCHITECTURES["sm86"]["cuda_archs"] == "86"
    assert kernel_builder.ARCHITECTURES["sm89"]["cuda_archs"] == "89"
    assert kernel_builder.ARCHITECTURES["sm120"]["cuda_archs"] == "120f"
    assert kernel_builder._cuda_architecture("sm120", "cu129") == "120a"
    assert kernel_builder._cuda_architecture("sm120", "cu130") == "120f"
    assert kernel_builder._cuda_architecture("sm86", "cu130") == "86"
    assert kernel_builder._cuda_architecture("sm89", "cu130") == "89"
    assert kernel_builder.architecture_for_capability((8, 6)) == "sm86"
    assert kernel_builder.architecture_for_capability((8, 9)) == "sm89"
    assert kernel_builder.architecture_for_capability((12, 0)) == "sm120"


def test_kernel_builder_initializes_only_required_top_level_submodules(
    tmp_path, monkeypatch
):
    kernel_builder = _load(
        "spark_kernel_builder_submodules", ROOT / "comfyui/kernel_builder.py"
    )
    destination = tmp_path / "source"
    calls = []

    def run(command, *, cwd=None, env=None):
        del env
        calls.append((command, cwd))
        if command[:2] == ["git", "clone"]:
            destination.mkdir()

    monkeypatch.setattr(kernel_builder, "_run", run)
    monkeypatch.setattr(kernel_builder, "_patch_path", lambda _root, _base: tmp_path / "patch")
    monkeypatch.setattr(kernel_builder, "_set_local_version", lambda *_args, **_kwargs: None)
    kernel_builder.prepare_source(
        tmp_path,
        destination,
        source_url="https://example.invalid/comfy-kitchen.git",
    )

    clone = calls[0][0]
    assert "--recursive" not in clone
    submodule = next(command for command, _cwd in calls if command[:3] == ["git", "submodule", "update"])
    assert "--recursive" not in submodule
    assert submodule[-2:] == ["third_party/cutlass", "third_party/flash-attention"]


def test_kernel_builder_recognizes_an_existing_release_wheel(tmp_path, monkeypatch):
    kernel_builder = _load("spark_kernel_builder_repeat", ROOT / "comfyui/kernel_builder.py")
    expected = (
        tmp_path
        / "comfy_kitchen-0.2.36+spark.h3.sm120.cu129.1-cp312-abi3-linux_x86_64.whl"
    )
    expected.touch()

    class TemporarySource:
        def __enter__(self):
            return str(tmp_path / "temporary")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(kernel_builder.tempfile, "TemporaryDirectory", lambda **_kwargs: TemporarySource())
    monkeypatch.setattr(kernel_builder, "prepare_source", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(
        kernel_builder, "wheel_command", lambda *_args: ["test-builder"]
    )
    run_calls = []
    monkeypatch.setattr(
        kernel_builder,
        "_run",
        lambda *_args, **kwargs: run_calls.append(kwargs),
    )
    assert kernel_builder.build_wheel(
        ROOT,
        tmp_path,
        kitchen_base="0.2.36",
        cuda_tag="cu129",
        cuda_archs="120a",
    ) == [expected]
    assert run_calls[-1]["env"]["COMFY_CUDA_ARCHS"] == "120a"
    with pytest.raises(ValueError, match="does not match sm120/cu130"):
        kernel_builder.build_wheel(
            ROOT,
            tmp_path,
            kitchen_base="0.2.36",
            cuda_tag="cu130",
            cuda_archs="120a",
        )


def test_installer_selects_matching_base_from_local_wheels(tmp_path):
    installer = _load("spark_comfyui_installer", ROOT / "comfyui/install.py")
    from packaging.tags import sys_tags
    tag = next(iter(sys_tags()))
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheels = {
        base: wheelhouse
        / f"comfy_kitchen-{base}+spark.h3.sm120.cu130.1-{tag}.whl"
        for base in installer.SUPPORTED_KITCHEN_BASES
    }
    for wheel in wheels.values():
        wheel.touch()
    assert installer._matching_local_wheel(tmp_path, "0.2.36") == wheels["0.2.36"]
    assert installer._matching_local_wheel(tmp_path, "0.2.37") == wheels["0.2.37"]

    cu129 = (
        wheelhouse
        / f"comfy_kitchen-0.2.36+spark.h3.sm120.cu129.1-{tag}.whl"
    )
    cu129.touch()
    assert installer._matching_local_wheel(
        tmp_path, "0.2.36", "sm120", "cu129"
    ) == cu129
    assert not installer._wheel_matches(cu129.name, "0.2.36", "sm120", "cu130")
    assert not installer._wheel_matches(
        wheels["0.2.36"].name, "0.2.36", "sm120", "cu129"
    )

    sm89 = (
        wheelhouse
        / f"comfy_kitchen-0.2.36+spark.h3.sm89.cu130.1-{tag}.whl"
    )
    sm89.touch()
    assert installer._matching_local_wheel(tmp_path, "0.2.36", "sm89") == sm89
    assert not installer._wheel_matches(sm89.name, "0.2.36", "sm120")
    assert not installer._wheel_matches(wheels["0.2.36"].name, "0.2.36", "sm89")

    sm86 = (
        wheelhouse
        / f"comfy_kitchen-0.2.36+spark.h3.sm86.cu130.1-{tag}.whl"
    )
    sm86.touch()
    assert installer._matching_local_wheel(tmp_path, "0.2.36", "sm86") == sm86
    assert not installer._wheel_matches(sm86.name, "0.2.36", "sm89")


def test_installer_resolves_supported_installed_base(monkeypatch):
    installer = _load("spark_comfyui_installer_base", ROOT / "comfyui/install.py")
    monkeypatch.setattr(installer.metadata, "version", lambda _name: "0.2.37")
    assert installer._resolve_kitchen_base() == "0.2.37"

    monkeypatch.setattr(installer.metadata, "version", lambda _name: "0.2.35")
    with pytest.raises(RuntimeError, match="ComfyUI 0.38.x") as error:
        installer._resolve_kitchen_base()
    assert "ComfyUI 0.39.x/0.40.x" in str(error.value)


def test_installer_requires_opt_in_for_cu129_and_unknown_cuda(monkeypatch, capsys):
    installer = _load("spark_comfyui_installer_cuda", ROOT / "comfyui/install.py")
    torch_stub = ModuleType("torch")
    torch_stub.__version__ = "2.12.0+cu129"
    torch_stub.version = ModuleType("torch.version")
    torch_stub.version.cuda = "12.9"
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "triton", ModuleType("triton"))

    with pytest.raises(RuntimeError, match="not in the Spark-H3 release matrix"):
        installer._validate_runtime()
    assert installer._validate_runtime(experimental_cuda=True) == (12, 9)
    assert "experimental CUDA mode" in capsys.readouterr().err
    torch_stub.version.cuda = "13.0"
    torch_stub.__version__ = "2.11.9+cu130"
    assert installer._validate_runtime() == (13, 0)
    torch_stub.__version__ = "2.12.0+cu130"
    assert installer._validate_runtime() == (13, 0)
    torch_stub.__version__ = "2.13.0+cu130"
    assert installer._validate_runtime() == (13, 0)
    torch_stub.version.cuda = "13.2"
    assert installer._validate_runtime() == (13, 2)


def test_installer_validates_runtime_and_package_platform(monkeypatch, tmp_path):
    installer = _load(
        "spark_comfyui_installer_platform", ROOT / "comfyui/install.py"
    )
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    assert installer._runtime_platform_tag() == "linux-x86_64"
    monkeypatch.setattr(installer.sys, "platform", "win32")
    monkeypatch.setattr(installer.platform, "machine", lambda: "AMD64")
    assert installer._runtime_platform_tag() == "windows-x86_64"

    identity = {
        "architecture": "sm89",
        "cuda_tag": "cu130",
        "platform": "windows-x86_64",
        "kernel_wheels": [],
    }
    (tmp_path / "spark_h3_build.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    installer._validate_package_identity(
        tmp_path,
        architecture="sm89",
        cuda_tag="cu130",
        platform_tag="windows-x86_64",
    )
    with pytest.raises(RuntimeError, match="platform='windows-x86_64'"):
        installer._validate_package_identity(
            tmp_path,
            architecture="sm89",
            cuda_tag="cu130",
            platform_tag="linux-x86_64",
        )

    registry_identity = {
        "distribution": "registry",
        "architectures": ["sm86", "sm89", "sm120"],
        "cuda_tag": "cu130",
        "platforms": ["linux-x86_64", "windows-x86_64"],
        "kernel_wheels": [],
    }
    (tmp_path / "spark_h3_build.json").write_text(
        json.dumps(registry_identity), encoding="utf-8"
    )
    installer._validate_package_identity(
        tmp_path,
        architecture="sm86",
        cuda_tag="cu130",
        platform_tag="linux-x86_64",
    )
    with pytest.raises(RuntimeError, match="does not support sm90"):
        installer._validate_package_identity(
            tmp_path,
            architecture="sm90",
            cuda_tag="cu130",
            platform_tag="linux-x86_64",
        )


@pytest.mark.parametrize(
    ("cuda_version", "capability", "extra_args", "expected"),
    (
        ((13, 0), (12, 0), (), ("sm120", "cu130", None)),
        ((13, 0), (8, 6), (), ("sm86", "cu130", None)),
        ((13, 0), (8, 9), (), ("sm89", "cu130", None)),
        (
            (12, 9),
            (12, 0),
            ("--experimental-cuda",),
            ("sm120", "cu129", "120a"),
        ),
    ),
)
def test_installer_source_build_routes_isolated_identity(
    monkeypatch, cuda_version, capability, extra_args, expected
):
    installer = _load(
        f"spark_comfyui_installer_route_{capability}_{cuda_version}",
        ROOT / "comfyui/install.py",
    )
    calls = []
    monkeypatch.setattr(
        installer,
        "_validate_runtime",
        lambda **_kwargs: cuda_version,
    )
    monkeypatch.setattr(installer, "_cuda_capability", lambda: capability)
    monkeypatch.setattr(installer, "_backend_available", lambda *_args: False)
    monkeypatch.setattr(installer, "_validate_backend", lambda *_args: None)
    monkeypatch.setattr(
        installer,
        "_install_requirements",
        lambda _root: pytest.fail("--skip-dependencies must skip requirements"),
    )
    kernel_builder = ModuleType("kernel_builder")
    kernel_builder.install_from_source = lambda *_args, **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "kernel_builder", kernel_builder)

    assert installer.main(
        [
            "--source",
            "--skip-dependencies",
            "--kitchen-base",
            "0.2.37",
            *extra_args,
        ]
    ) == 0
    assert len(calls) == 1
    assert (
        calls[0]["architecture"],
        calls[0]["cuda_tag"],
        calls[0]["cuda_archs"],
    ) == expected


@pytest.mark.parametrize("cuda_version", ((13, 2), (13, 3), (14, 0)))
@pytest.mark.parametrize("explicit_wheel", (False, True))
def test_installer_automatically_tries_cu130_wheel_on_newer_cuda(
    tmp_path, monkeypatch, cuda_version, explicit_wheel
):
    installer = _load("spark_comfyui_installer_cu130_experiment", ROOT / "comfyui/install.py")
    from packaging.tags import sys_tags

    wheel_tag = next(sys_tags())
    wheel = tmp_path / "wheelhouse" / (
        f"comfy_kitchen-0.2.39+spark.h3.sm120.cu130.1-{wheel_tag}.whl"
    )
    wheel.parent.mkdir()
    wheel.touch()
    (tmp_path / "spark_h3_build.json").write_text(json.dumps({
        "architecture": "sm120", "cuda_tag": "cu130",
        "platform": installer._runtime_platform_tag(),
        "kernel_wheels": [wheel.name],
    }), encoding="utf-8")
    monkeypatch.setattr(installer, "_node_root", lambda: tmp_path)
    monkeypatch.setattr(installer, "_validate_runtime", lambda **_kwargs: cuda_version)
    monkeypatch.setattr(installer, "_cuda_capability", lambda: (12, 0))
    monkeypatch.setattr(installer, "_backend_available", lambda *_args: False)
    monkeypatch.setattr(installer, "_matching_release_wheel", lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected download")))
    installed = []
    monkeypatch.setattr(installer, "_install", lambda path, **_kwargs: installed.append(path))
    monkeypatch.setattr(installer, "_validate_backend", lambda *args: installed.append(args))
    arguments = ["--skip-dependencies", "--kitchen-base", "0.2.39"]
    if explicit_wheel:
        arguments += ["--wheel", str(wheel)]
    assert installer.main(arguments) == 0
    assert installed == [str(wheel), ("0.2.39", "sm120", "cu130")]


def test_installer_backend_probe_ignores_stale_parent_modules(tmp_path, monkeypatch):
    installer = _load("spark_comfyui_installer_probe", ROOT / "comfyui/install.py")
    package = tmp_path / "comfy_kitchen" / "backends"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cuda.py").write_text(
        "class Native:\n"
        "    def spark_attn(self):\n        return None\n"
        "_C = Native()\n"
        "def spark_attn():\n    return None\n",
        encoding="utf-8",
    )
    dist_info = tmp_path / "comfy_kitchen-0.2.37+spark.h3.sm120.cu130.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: comfy-kitchen\n"
        "Version: 0.2.37+spark.h3.sm120.cu130.1\n",
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
        installer, "_backend_available", lambda _base, _architecture, _cuda_tag: True
    )
    stale = ModuleType("comfy_kitchen.backends.cuda")
    monkeypatch.setitem(sys.modules, "comfy_kitchen.backends.cuda", stale)

    installer._validate_backend("0.2.37")

    assert "SM120 CU130 backend is ready" in capsys.readouterr().out


@pytest.mark.parametrize("tool", ("path-uv", "module-uv", "pip", "broken-path-uv"))
def test_package_manager_selects_existing_tools(monkeypatch, tmp_path, tool):
    manager = _load("spark_package_manager", ROOT / "comfyui/package_manager.py")
    python = str(tmp_path / "ComfyUI env" / "python")
    uv = str(tmp_path / "tools" / "uv")
    probes = []
    monkeypatch.setattr(
        manager.shutil,
        "which",
        lambda name: (
            uv if name == "uv" and tool in {"path-uv", "broken-path-uv"} else None
        ),
    )

    def available(command):
        probes.append(command)
        if tool == "path-uv":
            return True  # Both uv and pip exist; PATH uv must win.
        if tool in {"module-uv", "broken-path-uv"}:
            return command == [python, "-m", "uv"]
        return command == [python, "-m", "pip"]

    monkeypatch.setattr(manager, "_available", available)
    if tool == "pip":
        prefix = [python, "-m", "pip"]
        expected_install = [*prefix, "install"]
        expected_build = [*prefix, "wheel", "--no-deps", "--wheel-dir", str(tmp_path)]
    else:
        prefix = [uv] if tool == "path-uv" else [python, "-m", "uv"]
        expected_install = [*prefix, "pip", "install", "--python", python]
        expected_build = [
            *prefix,
            "build",
            "--wheel",
            "--python",
            python,
            "--out-dir",
            str(tmp_path),
        ]
    assert manager.install_command(python) == expected_install
    assert manager.wheel_command(python, tmp_path) == expected_build
    if tool == "path-uv":
        assert probes == [[uv], [uv]]


@pytest.mark.parametrize(
    "failure",
    (
        FileNotFoundError(),
        subprocess.TimeoutExpired("uv", 5),
        subprocess.CalledProcessError(1, "uv"),
    ),
)
def test_package_manager_treats_failed_probes_as_unavailable(monkeypatch, failure):
    manager = _load("spark_package_manager_probe", ROOT / "comfyui/package_manager.py")

    def run(*_args, **kwargs):
        assert kwargs["timeout"] == 5
        raise failure

    monkeypatch.setattr(manager.subprocess, "run", run)
    assert not manager._available(["uv"])


def test_package_manager_reports_missing_tools_without_bootstrap(monkeypatch):
    manager = _load(
        "spark_package_manager_missing", ROOT / "comfyui/package_manager.py"
    )
    monkeypatch.setattr(manager.shutil, "which", lambda _name: None)
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(manager.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="no usable uv or pip"):
        manager.install_command(sys.executable)
    assert calls == [
        [sys.executable, "-m", "uv", "--version"],
        [sys.executable, "-m", "pip", "--version"],
    ]


def test_installer_preserves_install_arguments(monkeypatch, tmp_path):
    installer = _load("spark_installer_commands", ROOT / "comfyui/install.py")
    prefix = ["uv", "pip", "install", "--python", sys.executable]
    targets = []

    def install_command(python):
        targets.append(python)
        return prefix[:]

    monkeypatch.setattr(installer, "install_command", install_command)
    calls = []
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("packaging>=24\n", encoding="utf-8")
    installer._install_requirements(tmp_path)
    wheel = str(tmp_path / "backend wheel.whl")
    installer._install(wheel, force=True)
    assert targets == [sys.executable, sys.executable]
    assert calls == [
        ([*prefix, f"-r{requirements}"], {"check": True}),
        ([*prefix, "--force-reinstall", "--no-deps", wheel], {"check": True}),
    ]


def test_installer_does_not_retry_failed_installs_with_another_tool(monkeypatch):
    installer = _load("spark_installer_failure", ROOT / "comfyui/install.py")
    selections = []

    def install_command(python):
        selections.append(python)
        return ["uv", "pip", "install", "--python", python]

    monkeypatch.setattr(installer, "install_command", install_command)
    calls = []
    failure = subprocess.CalledProcessError(2, "uv pip install")

    def run(command, **_kwargs):
        calls.append(command)
        raise failure

    monkeypatch.setattr(installer.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        installer._install("backend.whl", force=True)
    assert error.value is failure
    assert selections == [sys.executable]
    assert len(calls) == 1


@pytest.mark.parametrize("use_uv", (True, False))
@pytest.mark.parametrize("operation", ("build", "install"))
def test_kernel_builder_preserves_target_and_cuda_settings(
    monkeypatch,
    tmp_path,
    use_uv,
    operation,
):
    kernel = _load("spark_kernel_commands", ROOT / "comfyui/kernel_builder.py")
    manager = sys.modules[kernel.install_command.__module__]
    python = str(tmp_path / "target env" / "python")
    prefix = ["uv"] if use_uv else [python, "-m", "pip"]
    monkeypatch.setattr(manager, "_select", lambda _python: (prefix, use_uv))
    source = tmp_path / "source"
    monkeypatch.setattr(kernel, "prepare_source", lambda *_args, **_kwargs: source)
    calls = []
    monkeypatch.setattr(
        kernel,
        "_run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    kwargs = {"python": python, "kitchen_base": "0.2.36", "architecture": "sm89"}
    if operation == "build":
        output = tmp_path / "wheels"
        output.mkdir()
        wheel = (
            output
            / "comfy_kitchen-0.2.36+spark.h3.sm89.cu130.1-cp312-abi3-linux_x86_64.whl"
        )
        wheel.touch()
        assert kernel.build_wheel(ROOT, output, **kwargs) == [wheel]
        expected = (
            [
                *prefix,
                "build",
                "--wheel",
                "--python",
                python,
                "--out-dir",
                str(output),
                ".",
            ]
            if use_uv
            else [*prefix, "wheel", "--no-deps", "--wheel-dir", str(output), "."]
        )
        assert calls[0][1]["cwd"] == source
    else:
        kernel.install_from_source(ROOT, **kwargs)
        expected = (
            [*prefix, "pip", "install", "--python", python]
            if use_uv
            else [*prefix, "install"]
        )
        expected += ["--force-reinstall", "--no-deps", str(source)]
    assert len(calls) == 1
    assert calls[0][0] == expected
    assert calls[0][1]["env"]["COMFY_CUDA_ARCHS"] == "89"


@pytest.mark.parametrize("distribution", ("standalone", "registry"))
def test_packaged_installer_uses_uv_without_pip(tmp_path, distribution):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the no-pip integration test")
    builder = _load("spark_package_uv", ROOT / "tools/build_comfyui_package.py")
    if distribution == "registry":
        package = builder.assemble_registry(
            tmp_path / "build",
            version="0.1.0",
        )
        archive = builder.make_registry_zip(package, tmp_path, version="0.1.0")
    else:
        package = builder.assemble(
            tmp_path / "build",
            version="0.1.0",
            publisher_id="test",
            platform_tag="linux-x86_64",
        )
        archive = builder.make_zip(
            package,
            tmp_path,
            version="0.1.0",
            platform_tag="linux-x86_64",
        )
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(unpacked)
    package = (
        unpacked if distribution == "registry" else unpacked / builder.PACKAGE_NAME
    )

    # Use a small local wheel; this checks installation without CUDA or a network.
    wheel = tmp_path / "spark_installer_fixture-1.0-py3-none-any.whl"
    info = "spark_installer_fixture-1.0.dist-info"
    files = {
        "spark_installer_fixture.py": "VALUE = 42\n",
        f"{info}/METADATA": "Metadata-Version: 2.1\nName: spark-installer-fixture\nVersion: 1.0\n",
        f"{info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files[f"{info}/RECORD"] = "".join(
        f"{name},,\n" for name in (*files, f"{info}/RECORD")
    )
    with zipfile.ZipFile(wheel, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    (package / "requirements.txt").write_text(wheel.as_uri() + "\n", encoding="utf-8")
    backend_asset = wheel
    if distribution == "registry":
        backend_asset = tmp_path / "release.zip"
        with zipfile.ZipFile(backend_asset, "w") as bundle:
            bundle.write(wheel, f"ComfyUI-Spark-H3/wheelhouse/{wheel.name}")

    env = dict(
        os.environ,
        UV_OFFLINE="1",
        UV_NO_INDEX="1",
        UV_CACHE_DIR=str(tmp_path / "uv-cache"),
        UV_PYTHON_DOWNLOADS="never",
    )
    target = tmp_path / "ComfyUI env"
    other = tmp_path / "other project" / ".venv"
    for venv in (target, other):
        subprocess.run(
            [uv, "venv", "--python", sys.executable, str(venv)],
            check=True,
            capture_output=True,
            env=env,
        )
    relative_python = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    python = target / relative_python
    env["VIRTUAL_ENV"] = str(other)
    probe = """
import importlib.util
from pathlib import Path
import sys
assert importlib.util.find_spec('pip') is None
assert importlib.util.find_spec('spark_installer_fixture') is None
sys.path.insert(0, sys.argv[1])
import install as installer
installer._install_requirements(Path(sys.argv[1]))
installer._wheel_matches = lambda name, *_args: name == Path(sys.argv[3]).name
installer._install_release_asset(Path(sys.argv[2]).as_uri(), '0.2.39', 'sm120', 'cu130')
import spark_installer_fixture
assert spark_installer_fixture.VALUE == 42
assert Path(spark_installer_fixture.__file__).is_relative_to(sys.prefix)
assert importlib.util.find_spec('pip') is None
"""
    subprocess.run(
        [str(python), "-I", "-c", probe, str(package), str(backend_asset), str(wheel)],
        cwd=other.parent,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            str(other / relative_python),
            "-I",
            "-c",
            "import importlib.util; assert importlib.util.find_spec('spark_installer_fixture') is None; assert importlib.util.find_spec('pip') is None",
        ],
        check=True,
        capture_output=True,
    )
