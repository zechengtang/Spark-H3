#!/usr/bin/env python3
"""Preview or copy tracked development files into a local Hugging Face clone.

The Hugging Face repository keeps its own Git history, LFS rules, and media.
This tool never deletes files or creates a commit in that repository.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]
LOCAL_ONLY = {".gitignore", "README.md", "README_HF.md"}
LOCAL_PREFIXES = (".github/",)
HF_ONLY = {".gitattributes"}
HF_PREFIXES = ("comfyui/demos/",)


def git_files(repo: Path) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return {name.decode() for name in result.stdout.split(b"\0") if name}


def git_root(path: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return Path(result.stdout.strip()).resolve()


def is_local_only(name: str) -> bool:
    return name in LOCAL_ONLY or name.startswith(LOCAL_PREFIXES) or name.endswith(".safetensors")


def is_hf_only(name: str) -> bool:
    return name in HF_ONLY or name.startswith(HF_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_dir", type=Path, help="Existing local clone of the HF repository")
    parser.add_argument("--apply", action="store_true", help="Copy changed files; default is preview")
    parser.add_argument(
        "--replace-card", action="store_true", help="Also copy README_HF.md to HF README.md"
    )
    args = parser.parse_args()
    hf_dir = args.hf_dir.expanduser().resolve()
    if git_root(hf_dir) != hf_dir or hf_dir == SOURCE:
        parser.error("hf_dir must be a separate Git repository root")
    if args.replace_card and not args.apply:
        parser.error("--replace-card requires --apply")
    if args.apply:
        dirty_source = subprocess.run(
            ["git", "-C", str(SOURCE), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
        if dirty_source.strip():
            parser.error("development checkout has uncommitted tracked changes; commit or stash them first")
        dirty = subprocess.run(
            ["git", "-C", str(hf_dir), "status", "--porcelain"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
        if dirty.strip():
            parser.error("HF checkout has uncommitted changes; review them first")

    source_files = git_files(SOURCE)
    hf_files = git_files(hf_dir)
    managed = sorted(name for name in source_files if not is_local_only(name))
    changes: list[tuple[str, str]] = []
    for name in managed:
        src, dst = SOURCE / name, hf_dir / name
        if not src.is_file():
            parser.error(f"tracked source file is missing: {name}")
        if dst.is_symlink() or (dst.exists() and not dst.is_file()):
            parser.error(f"target is not a regular file: {name}")
        if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
            changes.append((name, name))

    card_source, card_target = SOURCE / "README_HF.md", hf_dir / "README.md"
    card_differs = not filecmp.cmp(card_source, card_target, shallow=False)
    for _, target in changes:
        print(f"{'ADD' if target not in hf_files else 'UPDATE'} {target}")
    if card_differs:
        print("CARD REVIEW README_HF.md -> README.md")
    extra = sorted(name for name in hf_files - source_files if not is_hf_only(name))
    for name in extra:
        print(f"HF-ONLY REVIEW {name}")
    print(f"Summary: {len(changes)} shared changes; {len(extra)} HF-only review files")

    if args.apply:
        for source, target in changes:
            destination = hf_dir / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / source, destination)
        if args.replace_card and card_differs:
            shutil.copy2(card_source, card_target)
        print("Files copied. Review the HF Git diff, then commit and push separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
