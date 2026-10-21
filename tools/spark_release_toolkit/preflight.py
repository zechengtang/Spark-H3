"""Fail early when a release kit would use the wrong GPU or planner backend."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from config import ARCH, CAPABILITY, CUDA_TAG, KIT, PLATFORM

REQUIRED_TRITON_MODULES = (
    "landmark_projection.py",
    "landmark_v2_cosine_fast.py",
    "landmark_v2_cosine_triton.py",
    "landmark_v2_euclidean.py",
    "landmark_v2_fused_node.py",
    "landmark_v2_max.py",
    "landmark_v2_route.py",
)
PLANNER_FILES = REQUIRED_TRITON_MODULES + (
    "landmark_tree_clustering.py", "landmark_tree_v2.py",
    "landmark_tree_v2_triton.py", "landmark_v2_cosine.py",
)


def require_direct_triton_imports(plugin: Path) -> dict[str, str]:
    """Reject temporary no-Triton import stubs in release tests."""
    result = {}
    for name in REQUIRED_TRITON_MODULES:
        path = plugin / "h3_sparse_attention" / name
        source = path.read_bytes()
        tree = ast.parse(source, filename=str(path))
        direct = any(
            isinstance(node, ast.Import)
            and any(alias.name == "triton" for alias in node.names)
            for node in tree.body
        )
        if not direct:
            raise RuntimeError(f"{path}: top-level Triton import is missing; patched fallback source is not valid for a release benchmark")
        result[name] = hashlib.sha256(source).hexdigest()
    return result


def verify_packaged_source() -> None:
    expected = json.loads((KIT / "source-sha256.json").read_text(encoding="utf-8"))
    paths = [f"ComfyUI-Spark-H3/h3_sparse_attention/{name}" for name in PLANNER_FILES]
    paths.append("ComfyUI-Spark-H3/comfyui_reblock_plan.py")
    paths.append("ComfyUI-Spark-H3/comfyui_nodes.py")
    paths.append("lightx2v_workflow.json")
    for relative in paths:
        path = KIT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected.get(relative) != actual:
            raise RuntimeError(f"planner source changed after kit generation: {relative}")


def probe_triton() -> dict[str, object]:
    try:
        import torch
        import triton
        import triton.language as tl
    except ImportError as error:
        raise RuntimeError(
            "Triton is required for the GPU planner. In this Python environment, "
            f"install a PyTorch-compatible {'triton-windows' if PLATFORM == 'windows-x86_64' else 'triton'} build before building or validating; "
            "do not patch the planner to use a Torch fallback."
        ) from error
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"expected torch CUDA 13.0 ({CUDA_TAG}), found {torch.version.cuda}")
    if not torch.__version__.startswith("2.12."):
        raise RuntimeError(f"this release kit is pinned to torch 2.12.x, found {torch.__version__}")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != CAPABILITY:
        raise RuntimeError(f"expected {ARCH} GPU {CAPABILITY}, found {capability}")
    if not triton.__version__.startswith("3.7."):
        raise RuntimeError(
            f"torch 2.12 release kit expects Triton 3.7.x, found {triton.__version__}; "
            "verify the torch/Triton version pairing before proceeding"
        )
    globals()["tl"] = tl

    @triton.jit
    def add_one(x, y, BLOCK: tl.constexpr):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(y + offset, tl.load(x + offset) + 1)

    x = torch.arange(128, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    add_one[(1,)](x, y, BLOCK=128)
    torch.cuda.synchronize()
    if not torch.equal(y, x + 1):
        raise RuntimeError("Triton JIT probe produced an incorrect result")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "capability": capability,
        "triton": triton.__version__,
        "triton_path": str(Path(triton.__file__).resolve()),
        "jit_probe": "pass",
    }


def check_build_toolkit() -> dict[str, str]:
    """Check the actual CUDA compiler independently of the driver version."""
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    expected = re.fullmatch(r"cu(\d{2})(\d+)", CUDA_TAG)
    if expected is None:
        raise RuntimeError(f"invalid wheel CUDA tag: {CUDA_TAG}")
    expected_version = f"{int(expected.group(1))}.{int(expected.group(2))}"
    if torch.version.cuda != expected_version:
        raise RuntimeError(
            f"building {CUDA_TAG} requires PyTorch CUDA {expected_version}, found {torch.version.cuda}"
        )
    if not CUDA_HOME:
        raise RuntimeError("PyTorch could not locate CUDA Toolkit; set CUDA_HOME to CUDA 13.0")
    toolkit = Path(CUDA_HOME).resolve()
    compiler = toolkit / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")
    path_compiler = shutil.which("nvcc")
    if not compiler.is_file() or not path_compiler:
        raise RuntimeError(f"CUDA compiler missing from Toolkit or PATH: {compiler}")
    if not os.path.samefile(compiler, path_compiler):
        raise RuntimeError(
            f"CUDA compiler mismatch: PyTorch uses {compiler}, PATH uses {path_compiler}; "
            f"select one CUDA {expected_version} Toolkit before building"
        )
    for variable in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT", "CUDA_TOOLKIT_ROOT_DIR", "CUDA_ROOT"):
        configured = os.environ.get(variable)
        if configured and Path(configured).resolve() != toolkit:
            raise RuntimeError(f"{variable}={configured} differs from PyTorch CUDA Toolkit {toolkit}")
    for variable in ("CUDACXX", "CMAKE_CUDA_COMPILER", "CUDA_NVCC_EXECUTABLE"):
        configured = os.environ.get(variable)
        if configured and (not Path(configured).is_file() or not os.path.samefile(compiler, configured)):
            raise RuntimeError(f"{variable}={configured} differs from CUDA compiler {compiler}")
    output = subprocess.check_output(
        [str(compiler), "--version"], text=True, encoding="utf-8", errors="replace"
    )
    release = re.search(r"\brelease\s+(\d+\.\d+)(?:\.[0-9]+)?\b", output)
    if release is None or release.group(1) != expected_version:
        raise RuntimeError(
            f"building {CUDA_TAG} requires nvcc {expected_version}; "
            f"found {release.group(1) if release else output.strip()} at {compiler}"
        )
    return {"toolkit_root": str(toolkit), "compiler": str(compiler), "nvcc_release": release.group(1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=KIT / "preflight.json")
    parser.add_argument("--require-build-toolkit", action="store_true")
    args = parser.parse_args()
    verify_packaged_source()
    planner = require_direct_triton_imports(KIT / "ComfyUI-Spark-H3")
    report = {"architecture": ARCH, "cuda_tag": CUDA_TAG, "planner_sha256": planner, **probe_triton()}
    nvcc = shutil.which("nvcc")
    report["nvcc"] = {
        "path": nvcc,
        "version_output": subprocess.check_output(
            [nvcc, "--version"], text=True, encoding="utf-8", errors="replace"
        ).strip() if nvcc else None,
    }
    if args.require_build_toolkit:
        report["build_toolkit"] = check_build_toolkit()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {args.report}")


if __name__ == "__main__":
    main()
