# BF16 Spark query/block approximate tail: 50-prompt extension

> **Retired/paused protocol (2026-09-27):** This setup reused 20 compile-off
> samples and did not finish the planned 50-prompt run. It is not a valid
> compiled four-arm main-result comparison. Its artifacts are itemized in
> `docs/uncompiled_delete_preview_20260927.tsv`; its compile-off reused
> artifacts were removed after review.

This running experiment extends the aligned 10-prompt comparison to all 50
prompts in the VBench core-five 20% subset. It fixes 10 s / 1344×768,
seed 42, 20 requested grid steps (19 transformer evaluations), and GPU0–3.
Both arms use Spark-H3 TopK10, power-of-two fanout16 reblocking, dense video
tail, BF16 global anchor, Tensor Core BF16 weighted K/V summaries, and
stored-key log-mass correction. The only intended intervention is
`sol_tail_granularity=query` versus `block` for the approximate branch.

The aligned 10-prompt source provides 10 matching query and 10 matching block
videos/latents, reused by sample ID and prompt SHA-256. The remaining 40 per
arm require new generation. The older 2026-09-20 canonical BF16 query 50-prompt
run cannot be substituted: although the dataset overlaps, all 10 overlapping
query latent SHA-256 hashes differ from the aligned run. The old saved kernel
source uses threshold TopK routing, whereas the aligned run records 735 packed
external TopK route calls. The old isolated pipeline also installs the real
per-transformer-block `torch.compile` acceleration hook. The open-source
Benchmark hook has since been repaired, but this run deliberately retains the
aligned source's explicit `--no-torch-compile` setting; changing both route
and compile would invalidate the one-variable comparison. The same
conditioning-cache symlink, prompt hashes, dense-reference hashes, seed, and
reblock root layout were verified for matching samples. The 10 s video count
is a multiple of 64, so dense tail handling does not explain this particular
divergence. The separate effects of routing and compilation have not been
isolated; its quality/speed is not a controlled baseline for this comparison.

Quality evaluation will pair each arm with the same dense videos for PSNR,
SSIM, and LPIPS. VBench core-five will score each prompt on its assigned
dimensions; raw per-prompt/dimension scores and video hashes will be retained.
No official overall VBench score is implied. Reused and newly generated
denoise times must not be treated as a paired runtime comparison. In addition,
the Diffusers block-granularity consumer path is functionally aligned but not
optimized, so measured block/query speed does not establish the algorithmic
cost of block-level approximation.

Runner: `scripts/diffusers_spark_tail_granularity_50prompt_20260927.py`.
The run started after the reweight-component 50-prompt experiment finished.
Its prepared protocol verifies 10 reused samples per arm and 80 new arm–prompt
pairs, and full configuration comparison shows exactly one differing field:
`sol_tail_granularity`. Results pending.
