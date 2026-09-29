"""Read-only, per-file preview for confirmed ineffective-compile H3 runs.

This script never removes experiment artifacts.  It records regular files and
symlinks separately so a shared conditioning target cannot be mistaken for an
owned output.  The allowlist is intentionally narrower than the full inventory.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path


DATA = Path("/autodl-fs/data")
EXPS = DATA / "h3_experiments"
OUTS = DATA / "h3_outputs"
HERE = Path(__file__).resolve().parents[1]
PREVIEW = HERE / "docs" / "uncompiled_delete_preview_20260927.tsv"
SUMMARY = HERE / "docs" / "uncompiled_delete_preview_20260927_summary.json"

GROUPS = {
    "protocol_false": [
        "blog_ablation_4prompt_nowarm_layer0_5s480p_seed42_20260923",
        "vbench_core5_25prompt_nowarm_layer0_5s480p_seed42_20260923",
        "step4_sparsity_10prompt_5s480p_seed42_20260923",
        "step4_sparsity_25prompt_5s480p_seed42_20260924",
        "step5_sparsity_25prompt_5s480p_seed42_20260924",
        "step5_sparsity_blog4_5s768p_seed42_20260924",
        "early_steps_sparsity_10prompt_5s480p_seed42_20260924",
        "early_steps_sparsity_25prompt_5s480p_seed42_20260924",
        "cumulative_step4_sparsity_25prompt_5s480p_seed42_20260924",
        "cumulative_step4_sparsity_blog4_5s768p_seed42_20260924",
    ],
    "runner_no_torch_compile": [
        "dense_step_ablation_5s_480p_20260920",
        "dense_layer_ablation_5s_480p_20260920",
        "slope_budget_5s_480p_20260920",
        "exp_budget_5s_480p_20260920",
        "sigma_budget_5s_480p_20260920",
        "shallow_slope_5s_480p_20260920",
        "reverse_layer_probe_5s_480p_20260921",
        "giraffe_768p10s_seed42_three_variants_20260923",
        "diffusers_sol_spark_4prompt_5s768p_20260926",
    ],
    "nested_contract_false": ["fasth3_vbench50_four_variants_20260921"],
    # Blog-facing four-prompt ablation is deliberately retained despite its
    # ineffective compile hook: the display blog and its evidence must agree.
    "inherited_no_compile_route": [
        "diffusers_spark_reweight_precision_25prompt_10s768p_20260926",
        "diffusers_spark_reweight_components_10prompt_10s768p_20260926",
        "diffusers_spark_reweight_components_50prompt_10s768p_20260926",
        "diffusers_spark_anchor_tail_10prompt_10s768p_aligned_20260926",
        "diffusers_spark_tail_granularity_50prompt_10s768p_20260927",
    ],
}

# Different parts of the blog main result are outside the deletion allowlist.
PROTECTED_ROOTS = [
    OUTS / "vbench20pct_768p10s_seed42_20260913",
    OUTS / "topk_reblock_reweight_50prompt_20260920",
    EXPS / "topk_reblock_reweight_50prompt_20260920",
    EXPS / "vbench20pct_768p10s_seed42_20260913",
    DATA / "h3_repos" / "MiniMax-H3-Sparse" / "reports" / "vbench20pct_768p10s_seed42_20260913",
    DATA / "h3_repos" / "MiniMax-H3-Sparse" / "reports" / "min10_topk10_reweight_l2_50prompt_20260915",
]

REFERENCE_ROOTS = [
    HERE / "README.md",
    HERE / "docs",
    HERE.parent / "MiniMax-H3-Experiments" / "README.md",
    HERE.parent / "MiniMax-H3-Experiments" / "docs",
    HERE.parent / "MiniMax-H3-Experiments" / "scripts",
    HERE.parent / "MiniMax-H3-Benchmark" / "README.md",
    HERE.parent / "MiniMax-H3-Benchmark" / "docs",
    HERE.parent / "MiniMax-H3-Benchmark" / "scripts",
]


def protected_inodes() -> dict[tuple[int, int], str]:
    result = {}
    for base in PROTECTED_ROOTS:
        if not base.exists():
            continue
        for root, _, files in os.walk(base, followlinks=False):
            for name in files:
                path = Path(root) / name
                if not path.is_symlink():
                    stat = path.stat()
                    result[(stat.st_dev, stat.st_ino)] = str(path)
    return result


def known_references(name: str) -> str:
    refs: list[str] = []
    for base in REFERENCE_ROOTS:
        if not base.exists():
            continue
        cmd = ["rg", "-l", "-F", "--glob", "!uncompiled_delete_preview_20260927*", name, str(base)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode in (0, 1):
            refs.extend(result.stdout.splitlines())
    if name == "blog_ablation_4prompts_rerun_20260922":
        refs.append(f"{HERE}/docs/blogs/spark-attn/README.md:73-80,105-112 (semantic ablation tables)")
    return ";".join(sorted(set(refs)))


def make_row(path: Path, evidence: str, experiment: str,
             inode_map: dict[tuple[int, int], str], refs: str) -> dict[str, object]:
    is_link = path.is_symlink()
    link_target = os.readlink(path) if is_link else ""
    stat = path.lstat()
    # Symlinks in these output trees point to protected conditioning and are
    # harmless to unlink only after review. Never traverse their targets.
    disposition = "review_symlink" if is_link else "delete_candidate"
    if "conditioning_cache" in path.parts or path.name == "conditioning_manifest.json":
        disposition = "protect_shared_input"
    shared_alias = "" if is_link else inode_map.get((stat.st_dev, stat.st_ino), "")
    if shared_alias:
        disposition = "protect_main_result_hardlink_alias"
    return {
        "disposition": disposition,
        "path": str(path),
        "bytes": stat.st_size,
        "file_type": "symlink" if is_link else "regular",
        "link_target": link_target,
        "evidence": evidence,
        "experiment": experiment,
        "known_reference": refs,
        "main_result_hardlink_alias": shared_alias,
    }


def main() -> None:
    if PREVIEW.with_name("uncompiled_delete_executed_20260927.json").exists():
        raise RuntimeError("Cleanup already executed; refusing to overwrite the frozen pre-deletion preview")
    rows: list[dict[str, object]] = []
    inode_map = protected_inodes()
    for evidence, names in GROUPS.items():
        for name in names:
            refs = known_references(name)
            for base in (EXPS / name, OUTS / name):
                if not base.exists() and not base.is_symlink():
                    continue
                for root, dirs, files in os.walk(base, followlinks=False):
                    for entry in dirs + files:
                        path = Path(root) / entry
                        if path.is_symlink() or path.is_file():
                            rows.append(make_row(path, evidence, name, inode_map, refs))
    rows.sort(key=lambda r: str(r["path"]))
    for row in rows:
        path = Path(str(row["path"]))
        if any(path == base or base in path.parents for base in PROTECTED_ROOTS):
            raise RuntimeError(f"Protected main-result path in deletion preview: {path}")
    with PREVIEW.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_disposition = {}
    by_evidence = {}
    for row in rows:
        key = str(row["disposition"])
        by_disposition.setdefault(key, {"files": 0, "bytes": 0})
        by_disposition[key]["files"] += 1
        by_disposition[key]["bytes"] += int(row["bytes"])
        key = str(row["evidence"])
        by_evidence.setdefault(key, {"files": 0, "bytes": 0})
        by_evidence[key]["files"] += 1
        by_evidence[key]["bytes"] += int(row["bytes"])
    SUMMARY.write_text(json.dumps({
        "note": "Read-only deletion preview; no artifacts have been removed.",
        "candidate_roots": sum(len(v) for v in GROUPS.values()),
        "protected_roots": [str(p) for p in PROTECTED_ROOTS],
        "by_disposition": by_disposition,
        "by_evidence": by_evidence,
        "preview_tsv": str(PREVIEW),
    }, indent=2) + "\n")
    print(SUMMARY.read_text())


if __name__ == "__main__":
    main()
