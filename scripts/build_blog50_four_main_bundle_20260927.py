"""Build/verify a byte-preserving, relocatable four-arm blog50 artifact bundle.

Run with ``build`` or ``verify``.  Source files are never modified or removed.
Hard links make the bundle path-independent without duplicating ~42 GiB on the
same filesystem; JSON absolute paths are handled by relocation_map.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


DATA = Path("/autodl-fs/data")
REPO = Path(__file__).resolve().parents[1]
OUT = DATA / "h3_experiments" / "blog50_four_main_selfcontained_20260927"
# The old aggregate JSON mixed single-call ATTN microbenchmarks with
# full-denoising and 50-prompt quality results. Keep its historical four-row
# source recoverable from Git without restoring the misleading live file.
HISTORICAL_BLOG_REF = "e6d5832c05d51f6410a080a1d67c9d414cdd0f2b:docs/blogs/spark-attn/results_50prompt.json"

SRC = {
    "dense_sol_output": DATA / "h3_outputs/vbench20pct_768p10s_seed42_20260913",
    "spark_output": DATA / "h3_outputs/topk_reblock_reweight_50prompt_20260920",
    "dense_sol_experiment": DATA / "h3_experiments/vbench20pct_768p10s_seed42_20260913",
    "spark_experiment": DATA / "h3_experiments/topk_reblock_reweight_50prompt_20260920",
    "sparse_reports": DATA / "h3_repos/MiniMax-H3-Sparse/reports",
}

SELECT = {
    "dense_sol_output": ["dense", "sol", "conditioning_cache", "conditioning_manifest.json", "regeneration_20260920.json"],
    "spark_output": ["latents", "videos/topk10_reblock_global_reweight", "videos/topk20_reblock_global_reweight"],
    "dense_sol_experiment": ["isolated_run", "protocol.json", "reuse_10pct_dense_sol.json", "metrics", "vbench", "quality/sol"],
    "spark_experiment": ["protocol.json", "results.json", "quality_results.json", "quality_protocol.json", "timing_35case.json", "runner_source.py", "source_manifest.json", "records", "snapshot", "vbench", "quality_work"],
    "sparse_reports": ["vbench20pct_768p10s_seed42_20260913", "min10_topk10_reweight_l2_50prompt_20260915", "optimized_three_methods_warmed_20260917", "tau1_global_reweight_fanout16_16_8_50prompt_20260918"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_sources():
    for key, choices in SELECT.items():
        root = SRC[key]
        for choice in choices:
            chosen = root / choice
            if not chosen.exists():
                raise FileNotFoundError(chosen)
            if chosen.is_file():
                yield key, chosen
            else:
                for base, dirs, names in os.walk(chosen, followlinks=False):
                    for name in sorted(names):
                        path = Path(base) / name
                        if path.is_symlink():
                            raise RuntimeError(f"Unexpected symlink in bundle source: {path}")
                        if path.is_file():
                            yield key, path


def derive_results() -> dict:
    source_bytes = subprocess.check_output(["git", "show", HISTORICAL_BLOG_REF], cwd=REPO)
    source = json.loads(source_bytes)
    wanted = {"dense", "sol", "topk10_reblock_global_reweight", "topk20_reblock_global_reweight"}
    rows = [{k: v for k, v in r.items() if k != "attention_speedup"}
            for r in source["rows"] if r["method"] in wanted]
    if len(rows) != 4 or {r["method"] for r in rows} != wanted:
        raise ValueError("Expected exactly four primary rows")
    return {
        "source_document_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_document_git_ref": HISTORICAL_BLOG_REF,
        "prompt_count": source["prompt_count"],
        "nfe_convention": source["nfe_convention"],
        "speedup_definition": source["speedup_definition"],
        "rows": rows,
        "note": "Derived four-row view from the historical Git snapshot. Mixed-protocol attention_speedup fields are omitted; see relocation_map.json for embedded historical absolute paths.",
    }


def expected_counts():
    checks = {
        "dense latent": OUT / "dense_sol_output/dense/latents",
        "dense video": OUT / "dense_sol_output/dense/videos",
        "sol latent": OUT / "dense_sol_output/sol/latents",
        "sol video": OUT / "dense_sol_output/sol/videos",
        "Spark latent": OUT / "spark_output/latents",
        "Spark10 video": OUT / "spark_output/videos/topk10_reblock_global_reweight",
        "Spark20 video": OUT / "spark_output/videos/topk20_reblock_global_reweight",
    }
    extensions = {"latent": ".pt", "video": ".mkv"}
    result = {}
    for label, directory in checks.items():
        ext = extensions[label.split()[-1]]
        files = list(directory.glob(f"*{ext}"))
        if label == "Spark latent":
            for arm in ("topk10", "topk20"):
                arm_files = [p for p in files if p.name.startswith(arm + "_")]
                if len(arm_files) != 50:
                    raise ValueError(f"{arm} latent count is {len(arm_files)}, not 50")
                result[f"{arm} latent"] = 50
        elif len(files) != 50:
            raise ValueError(f"{label} count is {len(files)}, not 50")
        else:
            result[label] = 50
    return result


def verify_relocation() -> dict[str, int]:
    """Resolve saved manifest data paths strictly inside this bundle."""
    mappings = json.loads((OUT / "relocation_map.json").read_text())["mappings"]

    def resolve(raw: str) -> Path:
        source = Path(raw)
        for entry in sorted(mappings, key=lambda m: len(m["source_prefix"]), reverse=True):
            prefix = Path(entry["source_prefix"])
            if source == prefix or prefix in source.parents:
                result = OUT / entry["bundle_prefix"] / source.relative_to(prefix)
                if not result.is_file() or OUT not in result.parents:
                    raise RuntimeError(f"Unresolved bundled reference: {raw} -> {result}")
                return result
        raise RuntimeError(f"Unmapped historical reference: {raw}")

    checked = {"conditioning_samples": 0, "dense_sol_latent_video": 0,
               "spark_latent": 0}
    condition = json.loads((OUT / "dense_sol_output/conditioning_manifest.json").read_text())
    resolve(condition["samples_path"])
    checked["conditioning_samples"] += 1
    for arm in ("dense", "sol"):
        manifest = json.loads((OUT / f"dense_sol_output/{arm}/generation_manifest.json").read_text())
        for record in manifest["records"]:
            resolve(record["latent_path"])
            resolve(record["output_path"])
            checked["dense_sol_latent_video"] += 2
    spark = json.loads((OUT / "spark_experiment/results.json").read_text())
    for record in spark["rows"]:
        resolve(record["latent_path"])
        checked["spark_latent"] += 1
    if checked != {"conditioning_samples": 1, "dense_sol_latent_video": 200,
                   "spark_latent": 100}:
        raise RuntimeError(f"Unexpected relocation counts: {checked}")
    return checked


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for key, source in iter_sources():
        rel = Path(key) / source.relative_to(SRC[key])
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.stat().st_ino != source.stat().st_ino or dest.stat().st_dev != source.stat().st_dev:
                raise RuntimeError(f"Existing bundle file is not the original hard link: {dest}")
        else:
            os.link(source, dest)
        entries.append({"path": str(rel), "bytes": source.stat().st_size, "sha256": sha256(source), "source": str(source)})
    (OUT / "results_four_main.json").write_text(json.dumps(derive_results(), ensure_ascii=False, indent=2) + "\n")
    (OUT / "relocation_map.json").write_text(json.dumps({
        "note": "For every absolute path embedded in an original manifest, map its longest matching source_prefix to bundle_prefix; reject unmatched paths.",
        "mappings": [{"source_prefix": str(v), "bundle_prefix": k} for k, v in SRC.items()],
    }, indent=2) + "\n")
    (OUT / "file_index.json").write_text(json.dumps({
        "file_count": len(entries), "total_bytes": sum(e["bytes"] for e in entries),
        "records": entries,
    }, indent=2) + "\n")
    verify()


def verify() -> None:
    index = json.loads((OUT / "file_index.json").read_text())
    for entry in index["records"]:
        path = OUT / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Bundle integrity failure: {path}")
    counts = expected_counts()
    relocation = verify_relocation()
    rows = json.loads((OUT / "results_four_main.json").read_text())["rows"]
    assert len(rows) == 4
    print(json.dumps({"bundle": str(OUT), "files_verified": len(index["records"]),
                      "bytes_verified": index["total_bytes"], "counts": counts,
                      "relocated_references": relocation}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"build", "verify", "verify-links"}:
        raise SystemExit("Usage: build_blog50_four_main_bundle_20260927.py build|verify|verify-links")
    if sys.argv[1] == "verify-links":
        print(json.dumps(verify_relocation(), indent=2))
    else:
        (build if sys.argv[1] == "build" else verify)()
