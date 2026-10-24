# SM90 / SM120 fused attention audit

The comparison concerns the H20 `packed` external-route path. Both candidate mainloop changes were measured with fixed packed routes and precomputed summaries. The KC change is now the default SM90 packed path; the log-mass experiment was removed after showing a small regression.

| Priority | Finding | Evidence | Outcome |
| --- | --- | --- | --- |
| Enabled | SM90 skipped route QK WGMMA but still transferred and waited for a KC tile in every route group. | `spark_reweight_sm90.py` now loads the useful AK tile on the existing K pipeline instead, preserving the K/V phase discipline. | Nine-seed replication found a repeatable but small 0.12%–0.37% mean improvement across the three frame-equivalent shapes. Full-model impact remains unmeasured. |
| Removed | An experimental change attempted to combine log-mass publication with the route-mask CTA barrier. | It needed a warp broadcast and alternating shared-memory buffers to preserve correctness. | Nine-seed replication found a repeatable 0.05%–0.08% mean regression at the three target shapes, so the experiment was removed. |
| Addressed | SM90's external-route selector synchronized route-QK partials and loaded a threshold that the external-route branch does not use. | `sol_attn/sm90/mainloop.py::sol_attn_build_route_mask_from_acc` now gates both operations on `not self.external_route`; the route-mask barrier still publishes the selected mask. | Measure separately from the KC pipeline change. |

The historical A/B used fixed Q/K/V, packed routes (10% target density), and summaries, alternating baseline, KC-only, log-mass-only, and combined runs on one H20. The current `scripts/benchmark_sm90_kernel_optimizations.py` retains the baseline-versus-KC comparison. The three frame counts map to 37/72/102 latent frames and 37,296/72,576/102,816 **video** tokens, using a 24×42 spatial patch grid. They are shape proxies, not actual full-model frame runs: audio and text tokens, model layers, and summary generation are excluded. Each row below gives the historical median of seven timed calls in milliseconds; all variants produced bitwise-equal outputs.

| Frame-equivalent shape | Baseline | KC-only | Log-mass-only | Both |
| --- | ---: | ---: | ---: | ---: |
| 124f / 37,296 video tokens | 35.067 | 34.956 | 35.062 | 34.986 |
| 243f / 72,576 video tokens | 125.997 | 125.900 | 126.096 | 125.902 |
| 345f / 102,816 video tokens | 252.586 | 251.675 | 252.710 | 251.806 |

The short-sequence (4,160 tokens, 2 heads) medians were 0.532, 0.550, 0.482, and 0.489 ms respectively. The attention-call timing includes wrapper launches and output allocation.

To check whether the trend persists across inputs, the same test was repeated with nine independent seeds (42–50) per frame-equivalent shape and nine interleaved rounds per seed. Each seed changes Q/K/V, route, and summaries. The table reports the mean of each seed's paired median percentage change; positive means faster. Wins count seeds with a lower median time than baseline.

| Shape | KC-only gain / wins | Log-mass-only gain / wins | Both gain / wins |
| --- | ---: | ---: | ---: |
| 124f | +0.237% / 8 of 9 | −0.080% / 1 of 9 | +0.256% / 9 of 9 |
| 243f | +0.123% / 9 of 9 | −0.052% / 0 of 9 | +0.095% / 9 of 9 |
| 345f | +0.370% / 9 of 9 | −0.049% / 0 of 9 | +0.348% / 9 of 9 |

The individual JSON files and seed-level bootstrap intervals remain as local,
gitignored evidence under `reports/h20_10s768p_20261022/replicated/`; they are
not part of the repository. This replication supports a small KC benefit and a
small log-mass regression for the tested synthetic inputs. It does not measure
model-level latency or remove possible systematic benchmark effects.

The KC optimization is enabled by default for SM90 packed routes. Set `H3_SM90_REUSE_K_PIPELINE_FOR_AK=0` only to reproduce the old path for A/B comparison. The log-mass optimization has no runtime switch or implementation. The short-sequence measurements have more timing noise and are exploratory. The targeted SM90 suite (`test_spark_reweight_sm90_sm100.py` and `test_spark_route_backends_sm90_sm100.py`) passed all 24 tests with the KC path enabled.
