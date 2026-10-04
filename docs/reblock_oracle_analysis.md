# Reblock Top-K versus oracle Top-K

This note records the October 2, 2026 fixed-Q/K/V investigation of how much
routing quality remains beyond the current reblocked mean route, including the
idealized “per-row Top-10%” mask and route scores that spend more than the
current `(N/64)^2` score budget while retaining the K64 execution interface.

## Expanded validation: 10 prompts, all evaluations and layers

The expanded October 2 run reproduces the **approximately 96% mean-to-oracle
mass efficiency** conclusion on a substantially broader sample. The relevant
sparse-stage result is **95.9832%**, with a remaining absolute mass gap of
**3.2117 percentage points**. This supports prioritizing reblock layouts and
the approximate branch over a wholesale replacement of the mean route score.
It does **not** imply that all query blocks or layers are already close to the
oracle, or that generated-video quality has a corresponding 4% upper bound.

### Inputs and coverage

- Dataset: the same ten VBench cases used by the existing Table 3/4 ten-prompt
  studies: IDs `1,6,11,15,21,27,32,38,44,50` from
  `MiniMax-H3-Benchmark/vbench_core5_percent_subsets/20pct/samples.json`.
  Generation uses their cached expanded multimodal prompts; the short labels
  below identify the cases rather than reproduce the full conditioning text.
- Current Diffusers BF16 Dense trajectory, seed 42, 1344x768, requested 120
  frames (aligned to 124), 20 requested scheduler points and 19 actual model
  evaluations. Diagnostics inspect post-RoPE Q/K and call the original Dense
  attention processor; sparse results never alter the trajectory.
- All evaluation indices `0–18`, all transformer layer indices `0–49`, and
  16 heads out of the model's **56**:
  `0,1,2,3,7,11,15,19,23,27,31,35,39,43,49,55`.
  The earlier capture contained only heads `0–3`, not all model heads.
- Sixteen uniformly spaced Q64 blocks per head, versus eight previously.
  Total: **9,500 layer/evaluation units and 2,432,000 Q64/head samples**.
- Primary statistics exclude dense warmup evaluations `0–3` and permanently
  dense layer `0`: evaluations `4–18`, layers `1–49`,
  **1,881,600 Q64/head samples**. All-sample and dense-only results are retained
  separately in `results.json`.
- Current production Spark fanout-16 flat/midpoint32 reblock, legacy midpoint
  arithmetic, and group size one. Geometry is `(37,24,42)` with 37,296 video
  tokens: 582 complete K64 blocks plus a 48-token tail. Every selector retains
  exactly **58 K64 blocks** for each sampled Q64/head unit.

Mean scoring and exact probabilities are evaluated in FP32 from the observed
BF16 Q/K, matching the earlier diagnostic definition. This is a fixed-budget
score comparison, not a measurement of threshold-tie behavior in a production
sparse kernel. Primary probability normalization uses complete video keys,
as in the earlier 96.13% experiment.

### Aggregate and per-prompt results

| Metric, actual sparse scope | Result |
|---|---:|
| Mean-route retained video attention mass | 0.781938 |
| Shared-K64 oracle retained video attention mass | 0.814055 |
| Mean per-unit mean/oracle mass efficiency | **95.9832%** |
| Prompt-cluster bootstrap 95% interval for efficiency | **95.9031%–96.0684%** |
| Remaining absolute video attention-mass gap | **3.2117pp** |
| Recall of oracle-selected K64 blocks | 82.0366% |

Efficiency is `mean(M_mean / M_oracle)` across Q64/head units, not
`mean(M_mean) / mean(M_oracle)`. The latter is about 96.05%; both support the
same conclusion. The bootstrap resamples the ten **prompts** (10,000 draws,
seed 42), rather than treating millions of correlated units as independent.

| Case | Short prompt label | Mean mass | Oracle mass | Efficiency | Gap (pp) |
|---|---|---:|---:|---:|---:|
| 1 | Person washing dishes | 0.7788 | 0.8113 | 95.901% | 3.249 |
| 6 | Train crossing a tall bridge | 0.7897 | 0.8231 | 95.874% | 3.342 |
| 11 | Sheep taking a peaceful walk | 0.7846 | 0.8160 | 96.095% | 3.145 |
| 15 | Aquarium | 0.7833 | 0.8159 | 95.937% | 3.266 |
| 21 | Football field | 0.7873 | 0.8176 | 96.216% | 3.034 |
| 27 | Phone booth | 0.7803 | 0.8136 | 95.854% | 3.325 |
| 32 | Storm trooper vacuuming the beach | 0.7829 | 0.8148 | 95.988% | 3.191 |
| 38 | 3D model of a Victorian house | 0.7773 | 0.8092 | 96.003% | 3.198 |
| 44 | Gwen Stacy reading a book | 0.7714 | 0.8048 | 95.783% | 3.339 |
| 50 | The Bund, Shanghai | 0.7839 | 0.8142 | 96.181% | 3.027 |

All ten prompt averages lie between **95.78% and 96.22%**. On the new prompts,
restricting the diagnostic to the old step/layer/head scope gives **96.43%**;
expanding coverage lowers it to **95.98%**. Thus the old layer selection was
somewhat optimistic, but the overall approximately-96% result survives.

The original four heads average **95.92%**, and the twelve added heads average
**96.00%**. Individual sampled-head averages range from **95.47%** (head 39)
to **96.53%** (head 55), so expanding head coverage does not reveal a broad
low-efficiency head population.

As a denominator check, the diagnostic also includes all context and tail
keys in each query's softmax denominator, reweights the block masses, and
reranks the oracle. The resulting **selected-video** mass efficiency is
**95.9027%**, with an absolute full-denominator mass gap of **2.8937pp**.
These numerators count selected complete video blocks; always-exact context
and tail mass is not added to either numerator. The conclusion is therefore
not an artifact of omitting their normalization contribution.

### Local gaps and interpretation

The overall bound is an **average bound**, not a per-unit guarantee:

- Evaluation averages decline from **96.37%** at eval 4 to **95.60%** at eval 18.
- Layer averages range from **92.66%** at layer 25 to **98.80%** at layer 1.
  Layer 25 has a **5.89pp** average mass gap; its worst evaluation/layer
  stratum is eval 16/layer 25 at **92.30%**, with a **6.23pp** gap.
- Per-unit median efficiency is **97.42%**, the fifth percentile is **87.15%**,
  and the first percentile is **77.74%**. **8.58%** of sampled units are below
  90%, and **1.44%** are below 80%.

Under the fixed layout, Q64/K64 shared mask and fixed per-Q64 quota, even an
exact mass selector can recover only the remaining **3.21pp average gap**.
The evidence consequently supports lowering the priority of a uniformly more
expensive route-score replacement. Better reblock layouts can raise the shared
oracle itself, while improved approximate summaries can better represent the
unselected mass. Those are reasonable primary research directions, although
this experiment does not establish their achievable quality gains.

The local tail also rules out the stronger claim that route-score work is
uniformly pointless. A targeted correction for layer 25 or difficult Q64
blocks may still be useful; its benefit must be assessed against actual cost
and generated quality. Attention-mass efficiency alone cannot bound output
error or end-to-end quality. Coverage remains one seed and the 5s shape,
16/56 heads, and sampled rather than exhaustive Q64 blocks.

### Reproduction and integrity

- Runner: `scripts/verify_shared_k64_oracle_10prompt_20261002.py`.
- Experiment directory:
  `/autodl-fs/data/h3_experiments/shared_k64_oracle_10prompt_allsteps_alllayers_20261002_v2`.
- `results.json` contains prompt/evaluation/layer/head summaries, 735 joint
  evaluation/layer strata, quantiles, bootstrap interval, and case records.
  `units/` retains all 9,500 per-layer diagnostic records.
- `current_route_config.json`, `effective_source.json`,
  `inference_runner_source.py`, and `analysis_source.py` record the effective
  configuration and sources. Startup attempts using an incorrect head-count
  assumption or obsolete frozen-config parameters produced no samples;
  their logs were preserved.
- `transformer_weight_identity.json` verifies SHA-256 identity of all thirteen
  transformer shards plus config/index between the shared-store model and
  `/root/h3_local/h3_diffusers/transformer`. Inference continued using the
  shared-store path after the verification reads warmed the cache.
- Two RTX PRO 6000 Blackwell GPUs, one worker/five prompts per GPU. Each
  instrumented generation took approximately 278–291 seconds; these are
  diagnostic runtimes, not Dense/Spark benchmark timings.
- Counts, sparse-scope group sizes, first production-permutation bijections,
  synthetic selector/normalization checks, and finite Dense latent outputs
  passed. Both GPUs were released after generation.

## Definitions

The phrase “mask Top-90” is ambiguous. For a useful sparse-attention upper
bound, this study interprets it as **mask the bottom 90% of keys independently
for every query row and retain the top 10%**. Literally masking the 90% highest
scoring keys would be a lower bound, not an oracle.

The production budget is matched exactly. With 582 complete K64 blocks at the
captured 5-second shape, Top-K 10% retains `round(0.1 * 582) = 58` blocks, or
`58 * 64` key tokens per query. The token oracle retains exactly that many
individual keys; it does not use a favorable `ceil(0.1 * tokens)` budget.

Three upper bounds were measured:

1. **Shared K64 oracle:** one optimal K64 mask shared by all 64 rows of a Q64
   block. This is the upper bound of the current execution interface.
2. **Row K64 oracle:** each query row chooses its own K64 blocks at the same
   token-pair budget. The current kernel cannot express this mask.
3. **Row token oracle:** each query row chooses arbitrary individual key tokens.
   This ignores the K64 execution constraint and is an absolute support oracle.

All rows use frozen Dense-reference Q/K/V from two prompts, evaluations
4/11/18, layers 12/24/36/48, four captured heads (0–3 out of the model's 56
heads), and eight sampled query blocks per
head: 768 sampled Q64/head units in total. The current fanout-16 reblocking and
10% budget are shared by every method.

## Initial two-prompt oracle gap

| Selector | Retained attention mass | Hard-mask relative L2 | Output cosine |
|---|---:|---:|---:|
| Current native mean | 0.8212 | 0.0975 | 0.9895 |
| Shared K64 oracle | 0.8542 | 0.0737 | 0.9956 |
| Per-row K64 oracle | 0.9037 | 0.0439 | 0.9982 |
| Per-row token oracle | 0.9582 | 0.0187 | 0.9995 |

The answer depends on which interface is considered:

- Under the current shared Q64-to-K64 mask, the mean route already reaches
  **96.13% mass efficiency**. The remaining selected-mass gap is 3.30
  percentage points. This is real but not large.
- Allowing a separate K64 mask for every query row exposes an additional 4.95
  points beyond the shared K64 oracle.
- Allowing arbitrary token masks exposes another 5.46 points, reaching 95.82%
  retained mass at the same logical token-pair budget.

The last number is not an executable 10%-density result on the current kernel.
The per-row token oracle touches an average **46.49%** of physical K64 parents
per row. Unioning the selected keys across the 64 rows of one Q64 block touches
**81.51%** of K64 parents. Converting that mask back to the present block format
would therefore destroy most of the intended sparsity.

## More expensive scores that still emit K64 masks

| Route score | Relative score work | Block recall | Mass efficiency | Selected mass | Hard-mask relative L2 |
|---|---:|---:|---:|---:|---:|
| Native Q64/K64 mean | 1x | 0.8180 | 0.9613 | 0.8212 | 0.0975 |
| Q32/K32 normalized mass | 4x | 0.8273 | 0.9648 | 0.8242 | 0.0957 |
| Q16/K64 normalized mass | 4x | 0.8241 | 0.9638 | 0.8234 | 0.0959 |
| Q8/K64 normalized mass | 8x | 0.8271 | 0.9651 | 0.8245 | 0.0950 |
| Q4/K64 normalized mass | 16x | 0.8317 | 0.9670 | 0.8261 | 0.0936 |
| Per-query/K64 normalized mass | 64x | 0.8415 | 0.9697 | 0.8285 | 0.0907 |
| Diagonal cumulant, lambda=1 | 4x | 0.8320 | 0.9658 | 0.8250 | 0.0955 |
| Exact LME of K64 tokens for mean Q | 64x | 0.8700 | 0.9760 | 0.8337 | 0.0908 |
| Shared K64 oracle | oracle | 1.0000 | 1.0000 | 0.8542 | 0.0737 |

The inexpensive candidates do not reveal a strong replacement for mean
routing:

- Q16/K64 gains only +0.62pp block recall and +0.22pp selected mass.
- Q32/K32 gains +0.93pp recall and +0.30pp mass. Raising both route axes to
  32-token resolution therefore spends work at a K granularity the execution
  kernel cannot use directly, with little benefit.
- Refining only Q while retaining K64 is directionally consistent, but even
  Q4/K64 at 16x score work gains only +1.37pp recall and +0.50pp mass.
- The best 4x candidate is the diagonal second-order cumulant at lambda=1:
  +1.41pp recall, +0.38pp selected mass, and 2.09% lower hard-mask relative L2.
  Sweeping lambda=1.5/2/3/4 gets progressively worse, so the result is not
  hiding a stronger high-variance operating point.
- Exact mean-Q/K64 LME is substantially better but needs 64x score pairs and
  still recovers only about 38% of the selected-mass gap to the shared oracle.

No sub-64 query or second-order candidate passed the predeclared screen of at
least +1.5pp block recall together with +0.3pp mass-efficiency gain. An
end-to-end generation expansion was therefore not started.

## Why the theoretical space is constrained

For one Q64 block and a fixed number of selected K64 blocks, summing the exact
per-row attention mass for each K64 parent and taking Top-K is already the
optimal shared mask. A different route score can only approximate that shared
oracle; it cannot recover the much larger per-row oracle gap without changing
the mask interface.

This separates two possible research directions:

1. **Better score, same kernel:** limited to the 3.30-point shared-oracle gap.
   Mean routing already captures most of it, and the tested 4x–16x scores
   recover only a small fraction.
2. **Richer mask interface:** potentially accesses the larger row-K64 gap, but
   requires multiple masks inside a Q64 block or a smaller executable query
   tile. Merely computing an N/32 route and unioning it back to Q64 cannot
   realize this benefit; the union raises physical density.

The most useful next kernel-level experiment is therefore not another global
mean surrogate. It is a grouped-query K64 mainloop—if hardware efficiency
allows it—that applies 2–4 independently routed subgroups within a physical Q64
tile without materializing their union. Its acceptance criterion should include
actual physical KV loads, not only logical token-pair density.

## Consistency with earlier experiments

- The older 10-second blog capture improved attention-mass recall from 68.66%
  before reblocking to 81.54% after reblocking; reblocked mass efficiency was
  91.71%. Reblocking itself was therefore a large improvement.
- The newer frozen 5-second capture used here places native mean at 96.13% of
  the shared K64 oracle, leaving much less room for another score-only change.
- Earlier exact mean-Q LME generation took 437.48s versus the reused 341.98s
  Spark timing, about 27.9% slower. Its PSNR was slightly lower, and the small
  SSIM/LPIPS improvement did not translate to a consistent VBench gain. Better
  fixed-trajectory mass recall is therefore not sufficient evidence of better
  generation quality.

## Reproducibility

Primary diagnostic:

- Script: `scripts/diagnose_reblock_token_oracle_candidates_2gpu_20261001.py`
- Experiment:
  `/autodl-fs/data/h3_experiments/reblock_token_oracle_candidates_2prompt_20261001`
- Full rows and aggregate metrics: `results.json`
- Human-readable table: `REPORT.md`

Follow-up Q-granularity and cumulant sweep:

- Script: `scripts/diagnose_query_granularity_cumulant_sweep_2gpu_20261002.py`
- Experiment:
  `/autodl-fs/data/h3_experiments/query_granularity_cumulant_sweep_2prompt_20261002`
- Full rows and aggregate metrics: `results.json`
- Human-readable table: `REPORT.md`

Related prior records:

- `/autodl-fs/data/h3_experiments/blog_ablation_4prompts_rerun_20260922/attention_mass_recall.json`
- `/autodl-fs/data/h3_experiments/global_anchor_route_oracles_2prompt_20260930_v2/REPORT.md`
- `/autodl-fs/data/h3_experiments/route_objective_hypotheses_2prompt_20260930/REPORT.md`
- `/autodl-fs/data/h3_experiments/exact_lme_meanq_table34_10prompt_20260930/REPORT.md`
- `/autodl-fs/data/h3_experiments/qrow_normalized_route_2prompt_20260930_v2/REPORT.md`
- `/autodl-fs/data/h3_experiments/q32_k32_normmass_recall_2prompt_20261001/REPORT.md`
