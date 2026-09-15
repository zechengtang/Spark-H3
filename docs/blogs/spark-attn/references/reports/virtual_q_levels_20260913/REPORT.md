# Virtual hierarchical-Q representative benchmark

Cases: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]. Seed42;20 steps;20% warmup;120 frames;832×480;24fps.

One mean Q anchor per virtual group produces empirical softmax-weighted K and V and anchor-logmass tangent. prev1/prev2 use actual penultimate/antepenultimate landmark-v2 hierarchy cuts. Actual64 query/key routing, additional10% exact budget, band and context behavior remain unchanged. Each trajectory realizes its own masks.

| Variant | PSNR | SSIM | LPIPS | ΔPSNR vs mean | ΔPSNR vs leaf64 | Denoise s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean | 20.868483 | 0.724343 | 0.189569 | +0.000000 | -0.632463 | 55.103 | 66.053 |
| leaf64 | 21.500946 | 0.746388 | 0.174035 | +0.632463 | +0.000000 | 57.672 | 67.563 |
| prev1 | 21.540448 | 0.748298 | 0.175276 | +0.671965 | +0.039502 | 60.811 | 66.938 |
| prev2 | 21.472140 | 0.741616 | 0.177724 | +0.603657 | -0.028806 | 55.923 | 66.358 |

PSNR pools pixels per video then arithmetic mean across cases; SSIM Gaussian11 sigma1.5 aligned-frame mean; LPIPS AlexNet v0.1 RGB[-1,1], arithmetic mean over aligned frames. FFV1 level3+PCM float audio; decoded RGB byte equality.

mean and leaf64 reuse previously audited all25 records with explicit source paths and hashes; leaf64 corresponds to prior a1_tilt. New prev1/prev2 runs are sharded across GPUs1–3. Timings include first-use compilation and are reported as measured; different GPUs and historical baseline execution conditions limit direct speed comparisons.

Per-case metrics, deltas, execution GPU identities, memory, source hashes, actual virtual layouts and output hashes are in summary.json and shard records. Large outputs: `/autodl-fs/data/h3_outputs/virtual_q_levels_20260913`.
