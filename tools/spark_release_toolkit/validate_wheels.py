#!/usr/bin/env python3
"""Validate architecture-specific comfy-kitchen wheels without installing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from config import ARCH, CAPABILITY, CUDA_TAG, ELF_ARCH, LOCAL_VERSION, PLATFORM, EXTENSION_SUFFIX, wheel_base


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_extension(wheel: Path, target: Path) -> Path:
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist()
                 if name.startswith("comfy_kitchen/backends/cuda/_C") and name.endswith(EXTENSION_SUFFIX)]
        if len(names) != 1:
            raise RuntimeError(f"{wheel.name}: expected one {EXTENSION_SUFFIX}, found {names}")
        archive.extractall(target)
    return target / names[0]


def child(wheel: Path) -> None:
    # A loaded .pyd remains memory-mapped until this child process exits on
    # Windows, so TemporaryDirectory cannot remove it from inside the child.
    directory = tempfile.mkdtemp(prefix=f"spark-h3-{ARCH}-wheel-")
    try:
        target = Path(directory)
        extension = extract_extension(wheel, target)
        sys.path.insert(0, str(target))

        import torch
        import comfy_kitchen
        from comfy_kitchen.backends.cuda import spark_attn

        distributions = list(importlib.metadata.distributions(path=[str(target)]))
        versions = [dist.version for dist in distributions if dist.metadata["Name"].lower() == "comfy-kitchen"]
        if len(versions) != 1:
            raise RuntimeError(f"could not identify wheel version: {versions}")
        version = versions[0]
        if "+" + LOCAL_VERSION not in version:
            raise RuntimeError(f"wrong backend identity: {version}")
        if torch.version.cuda != "13.0" or torch.cuda.get_device_capability() != CAPABILITY:
            raise RuntimeError(
                f"expected CUDA 13.0 / {ARCH}, found {torch.version.cuda} / {torch.cuda.get_device_capability()}"
            )

        torch.manual_seed(20261012)
        q = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        first = spark_attn(q, k, v, video_tokens=256, topk_ratio=0.5)
        second = spark_attn(q, k, v, video_tokens=256, topk_ratio=0.5)
        torch.cuda.synchronize()
        if first.shape != q.shape or first.dtype != torch.bfloat16:
            raise RuntimeError(f"unexpected output: {first.shape} {first.dtype}")
        if not bool(torch.isfinite(first).all()) or not torch.equal(first, second):
            raise RuntimeError("native output is non-finite or non-deterministic")
        print(json.dumps({
            "version": version,
            "module": comfy_kitchen.__file__,
            "extension": str(extension),
            "shape": list(first.shape),
            "dtype": str(first.dtype),
            "finite": True,
            "repeat_bitwise": True,
        }))
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", type=Path, default=Path(__file__).parent / "wheelhouse")
    parser.add_argument("--report", type=Path, default=Path(__file__).parent / "wheel-validation.json")
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child.resolve())
        return

    wheels = sorted(args.wheelhouse.resolve().glob("comfy_kitchen-*.whl"))
    expected_bases = {"0.2.36", "0.2.37", "0.2.39"}
    if not 1 <= len(wheels) <= len(expected_bases):
        raise RuntimeError(f"expected one to three wheels for {sorted(expected_bases)} in {args.wheelhouse}, found {wheels}")
    results = []
    found_bases = set()
    for wheel in wheels:
        base = wheel_base(wheel.name)
        if base is None:
            raise RuntimeError(f"wrong {PLATFORM} {ARCH}/{CUDA_TAG} wheel name: {wheel.name}")
        found_bases.add(base)
        with tempfile.TemporaryDirectory(prefix=f"spark-h3-{ARCH}-cuobjdump-") as directory:
            extension = extract_extension(wheel, Path(directory))
            dump = subprocess.check_output(
                ["cuobjdump", "--list-elf", str(extension)],
                text=True, encoding="utf-8", errors="replace", stderr=subprocess.STDOUT,
            )
        architectures = sorted(set(re.findall(r"sm_\d+", dump)))
        if architectures != [ELF_ARCH]:
            raise RuntimeError(f"{wheel.name}: expected only {ELF_ARCH}, found {architectures}")
        raw = subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(wheel)],
            text=True, encoding="utf-8", errors="replace",
        )
        smoke = json.loads(raw.strip().splitlines()[-1])
        results.append({
            "wheel": str(wheel),
            "sha256": sha256(wheel),
            "architectures": architectures,
            "cubin_count": sum("ELF file" in line for line in dump.splitlines()),
            "smoke": smoke,
        })
        print(f"PASS {wheel.name} sha256={results[-1]['sha256']}")
    if len(found_bases) != len(wheels) or not found_bases <= expected_bases:
        raise RuntimeError(f"duplicate or unsupported comfy-kitchen bases: {found_bases}")
    report = {"valid": True, "architecture": ARCH, "cuda_tag": CUDA_TAG,
              "platform": PLATFORM, "wheels": results}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"REPORT {args.report.resolve()}")


if __name__ == "__main__":
    main()
