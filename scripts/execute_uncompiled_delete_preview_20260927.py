"""Execute the reviewed exact-file cleanup; never remove directories.

Only rows tagged delete_candidate in the generated preview are eligible.
Preflight verifies each path is inside a confirmed allowlist root and is still
the same regular file/size recorded in the preview.  Symlinks are never unlinked.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from build_uncompiled_delete_preview_20260927 import (
    EXPS, GROUPS, OUTS, PREVIEW, PROTECTED_ROOTS,
)


LOG = PREVIEW.with_name("uncompiled_delete_executed_20260927.json")


def main() -> None:
    if LOG.exists():
        raise RuntimeError(f"Execution log already exists; refusing a second pass: {LOG}")
    roots = {(EXPS / name).resolve() for names in GROUPS.values() for name in names}
    roots |= {(OUTS / name).resolve() for names in GROUPS.values() for name in names}
    with PREVIEW.open(newline="") as f:
        rows = list(csv.DictReader(f))
    eligible = [r for r in rows if r["disposition"] == "delete_candidate"]
    if len(eligible) != 12619:
        raise RuntimeError(f"Unexpected candidate count {len(eligible)}")
    if sum(int(r["bytes"]) for r in eligible) != 192450420036:
        raise RuntimeError("Unexpected candidate byte total")
    # Complete validation before the first unlink. No path expansion, glob, or
    # recursively applied directory operation is used for deletion.
    for r in eligible:
        path = Path(r["path"])
        if not path.is_absolute() or r["file_type"] != "regular":
            raise RuntimeError(f"Invalid candidate type/path: {path}")
        if path.parent not in roots and not any(root in path.parents for root in roots):
            raise RuntimeError(f"Outside allowlist: {path}")
        if any(path == root or root in path.parents for root in PROTECTED_ROOTS):
            raise RuntimeError(f"Protected path: {path}")
        if path.is_symlink() or not path.is_file() or path.stat().st_size != int(r["bytes"]):
            raise RuntimeError(f"Changed candidate: {path}")
        if r["main_result_hardlink_alias"]:
            raise RuntimeError(f"Main-result hardlink alias: {path}")
    deleted = []
    errors = []
    for r in eligible:
        path = Path(r["path"])
        try:
            path.unlink()
            deleted.append({"path": str(path), "bytes": int(r["bytes"])})
        except OSError as exc:
            errors.append({"path": str(path), "error": str(exc)})
    report = {
        "preview": str(PREVIEW),
        "deleted_count": len(deleted),
        "deleted_bytes": sum(r["bytes"] for r in deleted),
        "error_count": len(errors),
        "errors": errors,
        "retained_blog_ablation": str(EXPS / "blog_ablation_4prompts_rerun_20260922"),
        "retained_main_bundle": "/autodl-fs/data/h3_experiments/blog50_four_main_selfcontained_20260927",
        "note": "Only exact regular-file rows were unlinked; no symlink or directory was removed.",
    }
    LOG.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError(f"Deletion errors: {len(errors)}; see {LOG}")


if __name__ == "__main__":
    main()
