# Compile-off cleanup audit (2026-09-27)

The exact-file cleanup has completed. It removed **12,619 regular files**, totaling
**192,450,420,036 bytes**, with zero errors. No directories or symlinks were
removed. The execution log is `uncompiled_delete_executed_20260927.json`;
the frozen per-file preview is `uncompiled_delete_preview_20260927.tsv`.

The deletion was restricted to 25 evidence-backed experiment names and their
same-named output trees. Each TSV row records its path, size, compile-off
evidence, known repository references, and disposition. Evidence categories
were explicit false protocol flags, explicit `--no-torch-compile` runners,
nested false contracts, or the reconstructed inherited no-compile route
helper. A missing compile field by itself was never treated as evidence.

**Protected:** all four compiled 50-prompt main arms, their Dense/Sol references,
conditioning, scores, frozen code, and their original manifests; 212 input
files/links in candidate trees; and the entire blog-facing four-prompt
`blog_ablation_4prompts_rerun_20260922` experiment plus its output tree.
The latter remains because the display blog and integration files were not
changed, so its tables must retain their source evidence. This run's old
Benchmark hook did not effectively compile blocks; its four-prompt numbers
are internally matched but must not be compared directly to the compiled
50-prompt absolute PSNR. This caveat lives in this audit, not the display blog.

The preview found 12 conditioning symlinks, all protected; ten point to the
compiled September-13 conditioning root and two point from a 10-prompt
reweight tree to a 25-prompt conditioning tree. No remaining deletion candidate
was a hard-link alias of the protected 50-prompt main files. Immediately before
deletion, a process-name scan found no active candidate generator/scorer, and
a scan of `docs/blogs/` found no remaining candidate-name reference.

## Self-contained four-main-arm bundle

`/autodl-fs/data/h3_experiments/blog50_four_main_selfcontained_20260927/`
contains **2,345 files** and **44,410,415,370 indexed bytes**. It includes
50 latent and 50 MKV video files each for Dense, Sol, Spark-H3-10pct, and
Spark-H3-20pct; conditioning tensors; generation manifests; frozen code and
source manifests; protocols; quality/VBench/timing data; and supporting
reports. `results_four_main.json` is a derived four-row view; the original
`docs/blogs/spark-attn/results_50prompt.json` is unchanged. Original file
bytes and hashes are preserved using hard links on the same filesystem.
`file_index.json` records relative path, byte count, SHA-256, and source path;
`relocation_map.json` maps embedded historical absolute paths to bundle paths.
The builder and a separate `verify` run checked all indexed SHA-256 values and
all eight arm/artifact counts. To repeat verification:

```bash
python scripts/build_blog50_four_main_bundle_20260927.py verify
```

Hard links make the bundle independent of its original directory names, but
do not protect against device failure. Copy/tar it to a second filesystem for
physical archival. The broad bundle includes supporting historical reports;
only four primary rows appear in its derived results file.

## Scope limit and remaining references

This was not a blanket purge of every experiment lacking a compile flag.
Mixed eager/compiled diagnostics, nested old isolated-run smoke artifacts,
the six-row Fast-H3 0753 archive, four cosine-temporal captures, and
ambiguous records were not approved for this pass. The inventory document
lists them for separate review. Non-blog documents describing deleted
reweight/tail data are marked retired; their numerical tables are historical
records of a protocol error, not current benchmark evidence. Some old
experiment scripts still name deleted output paths and should be treated as
historical, not runnable current benchmark entry points. The preview preserves
the old paths and references without leaving an active display-blog dependency.
