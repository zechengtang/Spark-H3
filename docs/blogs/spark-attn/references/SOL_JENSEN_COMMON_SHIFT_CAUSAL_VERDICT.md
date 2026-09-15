# Jensen Common-Shift Causal Diagnostic Verdict

## Verdict

The experiment supports two claims and rejects two stronger claims under the
frozen thresholds.

1. **Supported:** the omitted Jensen correction is predominantly a query-row
   common shift in the original ordering. All five fixtures pass the registered
   `rho_token >= 0.80` criterion.
2. **Supported causally:** orders three and above can substantially worsen the
   routing-relevant residual even when block centroids, covariance, the
   vanishing first-order term, and the complete second-order correction are
   fixed. The paired tail intervention passes in `1779/2240 = 79.42%` of
   strata, above the frozen 70% threshold.
3. **Not supported:** the registered residual-token redistribution did not
   establish moment homogenization as the cause of the common shift. Only
   `339/4480 = 7.57%` of strata pass the joint rho/residual threshold.
4. **Not supported as stated for Morton:** Morton makes routing worse, but the
   full conjunction "mean Jensen gap down >=10%, centered gap RMS up >=10%,
   mass regret up >=10%" holds on only `1/5` fixtures, below the required 3/5.
   The current Morton failure therefore cannot be described simply as a
   consistent low-order-improves/high-order-worsens tradeoff.

## Setup

- Five real post-QK-norm/post-RoPE fixtures: counters 0, 25, 50, 75, 99.
- Shape per fixture: `N=37296`, `H=56`, `D=128`.
- Complete 64-token blocks: 582; fixed exact route budget: 147 blocks.
- Compared original contiguous blocks with independent Q/K Morton
  `[32,16,8,4]`, partition factor 4, key64.
- Identity statistics use eight query blocks per head; causal interventions use
  two per head and four fixed seeds. Every measurement uses all 582 KV blocks.
- Query sampling is paired by original token anchor. Morton query blocks are
  located independently per head through `inv_perm_q`.
- Two RTX PRO 6000 GPUs; synchronized full-run wall time: 40.50 seconds.

No V tensor, sparse output approximation, or generated video participates in
this diagnostic.

## Hard correctness gate

All gates pass. Maxima over both orderings and all five fixtures:

| Invariant | Observed maximum | Limit |
|---|---:|---:|
| exact log-mean-exp reconstruction | 0 | 1e-6 |
| centered first-order term | 6.94e-15 | 2e-7 |
| intervention centroid error | 8.88e-16 | 1e-6 |
| block-mean score error after intervention | 8.88e-16 | 1e-6 |
| tail-intervention covariance relative error | 6.26e-16 | 1e-6 |
| tail-intervention second-order error | 1.78e-15 | 1e-6 |
| softmax change from adding a row constant | 8.33e-17 | 5e-7 |
| negative-control error | 6.39e-14 | 1e-6 |

The common shift changes no deterministic top-k index.

## Real original versus Morton blocks

`gap` is the mean exact Jensen correction. `residual RMS` removes the per-query
mean across KV blocks and is therefore routing-relevant.

| Counter | gap original -> Morton | residual RMS original -> Morton | token rho original -> Morton | mass regret original -> Morton |
|---:|---:|---:|---:|---:|
| 0 | 1.918 -> 1.792 | 0.472 -> 0.654 | .955 -> .867 | .0103 -> .0217 |
| 25 | 6.069 -> 6.499 | 1.558 -> 1.520 | .870 -> .860 | .1583 -> .1894 |
| 50 | 1.908 -> 1.747 | 0.524 -> 0.763 | .956 -> .848 | .0090 -> .0246 |
| 75 | 5.578 -> 6.089 | 1.465 -> 1.570 | .864 -> .841 | .1022 -> .1350 |
| 99 | 264.564 -> 216.315 | 87.760 -> 189.237 | .990 -> .804 | .0207 -> .0715 |

Mean-route top-147 overlap falls from a five-fixture macro average of 81.66%
to 76.85%. Mass regret increases on 5/5 fixtures. Centered exact-correction RMS
increases on 4/5 fixtures, while the mean gap decreases on three and increases
on two. Only counter 99 clears every registered 10% Morton-tradeoff threshold.

This distinction explains why a locality metric or even a smaller average gap
is insufficient: routing observes variation across KV blocks, not the omitted
row constant.

## Common-shift result

Original-order exact-correction rho values are:

- block level: macro mean 0.9610, median 0.9667;
- token level: macro mean 0.9272, median 0.9554;
- all five token-level fixtures exceed 0.80.

Morton lowers the macro token rho to 0.8439. Thus the correction remains large,
but substantially less of it is removable as a row constant.

The registered moment redistribution produces effects in the expected
direction but far below the causal threshold. Homogeneous minus heterogeneous
median changes are:

| Ordering | Endpoint | rho gain | residual RMS reduction |
|---|---|---:|---:|
| original | block | .0096 | 10.07% |
| original | token | .0266 | 8.77% |
| Morton | block | .0027 | 3.17% |
| Morton | token | .0061 | 1.55% |

Consequently, approximate cross-block moment homogeneity remains a plausible
analytic explanation, but this intervention does not identify it as the
dominant empirical cause. A shared query-dependent energy/covariance component
or learned head-wide structure may account for much of the already-high rho.

## Higher-order causal result

For fixed Q, the Householder intervention transforms each K residual matrix as
`R' = U R`, with `U` orthogonal and `U 1 = 1`. It therefore keeps centroid,
`R.T@R`, and the exact second cumulant fixed, while changing only orders three
and above.

Relative to the diffuse-tail condition, the concentrated-tail condition has:

- original ordering: median centered `H3+` RMS increase 242%, median mass
  regret increase 958%;
- Morton ordering: median centered `H3+` RMS increase 207%, median mass regret
  increase 396%.

This is direct causal evidence for the user's concern: reducing or holding the
0--2 order geometry does not constrain the tail-sensitive log-sum-exp error.
It establishes the possibility and mechanism, not that the synthetic tail
concentration is exactly what Morton does naturally.

## Taylor ladder

Adding the second cumulant improves mean-route mass regret on the first four
fixtures in both orderings. The third and fourth truncations are not monotonic,
and at counter 99 even the second-order approximation is worse than block mean.
At that late fixture the finite cumulant series has enormous terms and is not a
useful numerical approximation to log-sum-exp. The authoritative statistic is
therefore the exact centered correction `G - mean_K(G)`; finite Taylor terms
are mechanism probes only.

## Implication

The experiment validates the general danger but not the strongest Morton-
specific story. A clustering objective should not optimize only Euclidean
compactness, centroid score error, or average Jensen gap. To be relevant to SOL
routing it must directly reduce the cross-KV-block dispersion of exact or
tail-aware log mass near the top-k cutoff.

Before spending more work on a new clustering kernel, the highest-value next
experiment is a cheap scoring ablation using a bounded tail-aware correction
(rather than a raw Taylor polynomial), evaluated against centered exact mass
regret. V-side/video validation should follow only if that routing criterion
improves consistently.

## Reproduction

```bash
PYTHONPATH=. pytest -q h3_sparse_attention/tests/test_jensen_common_shift.py
PYTHONPATH=. python scripts/analyze_jensen_common_shift.py \
  --mode full --devices cuda:0,cuda:1 \
  --output benchmarks/sol_jensen_common_shift_rtx_pro_6000.json \
  --raw-output /autodl-fs/data/h3_experiments/jensen_common_shift/full_raw.json
```

See `SOL_JENSEN_COMMON_SHIFT_CAUSAL_MANIFEST.json` for fixture and result hashes.
