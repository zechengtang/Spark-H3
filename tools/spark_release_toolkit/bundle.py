"""Bundle the validated wheels and node source into a ComfyUI release archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from config import ARCH, CUDA_TAG, KIT, PLATFORM, wheel_base


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle(output: Path, audit_report: Path) -> dict:
    from preflight import require_direct_triton_imports, verify_packaged_source

    plugin = KIT / "ComfyUI-Spark-H3"
    verify_packaged_source()
    require_direct_triton_imports(plugin)
    wheels = sorted((KIT / "wheelhouse").glob("comfy_kitchen-*.whl"))
    if not wheels:
        raise RuntimeError("wheelhouse is empty; build or copy a wheel first")
    bases = set()
    for wheel in wheels:
        base = wheel_base(wheel.name)
        if base is None:
            raise RuntimeError(f"unexpected wheel for {ARCH}/{CUDA_TAG}: {wheel.name}")
        bases.add(base)
    if len(bases) != len(wheels):
        raise RuntimeError("duplicate comfy-kitchen base versions")
    if not audit_report.is_file():
        raise RuntimeError(f"wheel audit is required before bundling: {audit_report}")
    audit = json.loads(audit_report.read_text(encoding="utf-8"))
    audited = {Path(entry["wheel"]).name: entry["sha256"] for entry in audit.get("wheels", [])}
    actual = {wheel.name: sha256(wheel) for wheel in wheels}
    if (not audit.get("valid") or audit.get("architecture") != ARCH
            or audit.get("platform") != PLATFORM or audited != actual):
        raise RuntimeError("wheelhouse no longer matches the passing architecture-specific wheel audit")
    identity = json.loads((plugin / "spark_h3_build.json").read_text(encoding="utf-8"))
    if identity["architecture"] != ARCH or identity["cuda_tag"] != CUDA_TAG or identity["platform"] != PLATFORM:
        raise RuntimeError(f"plugin identity does not match release kit: {identity}")
    identity["kernel_wheels"] = [wheel.name for wheel in wheels]
    (plugin / "spark_h3_build.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"architecture": ARCH, "cuda_tag": CUDA_TAG, "platform": PLATFORM,
              "wheels": actual, "wheel_audit": str(audit_report.resolve())}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(plugin.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, (Path(plugin.name) / path.relative_to(plugin)).as_posix())
        for wheel in wheels:
            archive.write(wheel, f"{plugin.name}/wheelhouse/{wheel.name}")
    report["archive"] = str(output.resolve())
    report["archive_sha256"] = sha256(output)
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=KIT / "release" /
                        f"ComfyUI-Spark-H3-{PLATFORM.split('-')[0]}-{ARCH}-{CUDA_TAG}.zip")
    parser.add_argument("--audit-report", type=Path, help="wheel audit report (defaults to results-full, then results-audit)")
    args = parser.parse_args()
    audit_report = args.audit_report
    if audit_report is None:
        full_report = KIT / "results-full" / "wheel-validation.json"
        audit_report = full_report if full_report.is_file() else KIT / "results-audit" / "wheel-validation.json"
    print(json.dumps(bundle(args.output, audit_report), indent=2))


if __name__ == "__main__":
    main()
