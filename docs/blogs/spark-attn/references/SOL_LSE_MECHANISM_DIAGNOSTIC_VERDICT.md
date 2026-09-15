# SOL LSE correction mechanism diagnostic verdict

## Material Passport

- Origin: lightweight engineering/mechanism experiment
- Date: 2026-09-01 UTC
- Final verification status: **VERIFIED on independent v3**
- Final plan: `SOL_LSE_MECHANISM_DIAGNOSTIC_V3_PLAN.md`
- Final output:
  `/autodl-fs/data/h3_experiments/sol_lse_mechanism_diagnostic_v3`
- Final `metrics.json` SHA-256:
  `fabaeef6bea76ecc671ed2fd67e286a274b8040646823cc4474402006fb1643d`

## Verdict

The prior row-shared result is real but insufficient for the current
correction.  On 61,440 sampled production query rows, median `rho` remains
high at `0.8893`, confirming a strong row-common Jensen-gap component.
Nevertheless, the exact-selected blocks systematically underestimate the
mass correction required by the non-routed branch, and a scalar mass fix does
not repair its value-weighted numerator.

The dominant output mechanism is the latter: even the oracle scalar correction
that gives the average branch its exact denominator makes final attention
output substantially worse than leaving the branch uncorrected.  Therefore
the video degradation is not evidence that Jensen/LSE mass correction is
algebraically wrong.  It is evidence that correcting only the denominator
increases the influence of a poor compressed-V direction.

## Independent v3 results

V3 used previously uninspected VBench samples 5/6 with the registered 20-step,
seed-42, 240-frame, 1344x768 corrected-radix trajectory.

| Diagnostic | V3 result | Frozen decision |
|---|---:|---|
| `rho` median | 0.8893 | common component confirmed |
| exact-set `c_hat` median | 2.6197 | — |
| non-routed unweighted gap median | 2.9762 | — |
| ideal non-routed mass gap `c_A` median | 3.4680 | — |
| signed `c_hat-c_A` median | -0.6597 | exact route underestimates correction |
| `abs(c_hat-c_A)` median / p90 | 0.6945 / 2.1530 | gap-transfer failure |
| uncorrected mass log-error median | 3.4680 | — |
| corrected mass log-error median | 0.6945 | correction effective (80.0% reduction) |
| compressed-vs-exact average-output rel-L2 median | 0.5009 | value-numerator failure |
| compressed-vs-exact average-output cosine median | 0.8793 | value-numerator failure |

Final output relative-L2 against exact dense attention on the identical route:

| Variant | Pooled relative-L2 | Ratio to uncorrected |
|---|---:|---:|
| uncorrected compressed SOL | 0.10514 | 1.000x |
| current exact-set `c_hat` correction | 0.15918 | 1.514x |
| oracle scalar mass-only `c_A` correction | 0.24361 | 2.317x |

Thus the current correction really does improve the approximate scalar mass,
but worsens vector output.  Its systematic under-correction actually limits
the damage: making the denominator exactly right while keeping the same
compressed value vector is worse still.

The pure-threshold versus exact-budget route symmetric difference was 4.24%
of exact-budget selected memberships, below the frozen 5% materiality
threshold.  It is a real implementation difference but did not meet the
registered criterion for explaining the degradation.

## Correctness gate and sequential audit

V3 passed every frozen implementation check:

- 30/30 capture units and 61,440/61,440 finite sampled rows;
- route equality 100% over 1,104,000 cells;
- mean-gap maximum absolute error `0.05323 <= 0.08`;
- final-output FP32/Triton pooled relative-L2 `0.001953 <= 0.005`; and
- maximum per-row relative-L2 `0.01023 <= 0.020`.

The sequential history is retained rather than silently reclassified:

- V1 samples 1/2 failed its absolute-output gate because that gate ignored the
  scale of real V coordinates.  Its `metrics.json` remains `passed=false`.
- V2 samples 3/4 used a scale-aware gate but missed the registered gap limit by
  `7.79e-5` (`0.0650779 > 0.065`), so it also remains `passed=false`.
- V3 samples 5/6 froze `0.08` before capture and passed.  All six scientific
  decisions had the same direction in v1, v2, and v3.

## Interpretation and next implementation implication

For the non-routed branch, exact attention needs both

```text
sum_j exp(logit_j)
sum_j exp(logit_j) * V_j
```

The current kernel estimates the first quantity but continues to approximate
the second with a uniformly pooled `Vsum`.  A principled follow-up must model
the logit--V correlation or a vector numerator correction; another scalar
branch multiplier cannot solve the measured error.  Capping the correction
could reduce damage but would be a heuristic, not an LSE-correct result.

No videos were decoded or experiment history rewritten.  Raw Q/K/V captures
remain outside Git.
