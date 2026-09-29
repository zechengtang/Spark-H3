# GPU0–3 experiment queue (2026-09-27)

Run these experiments **serially**; do not let later jobs compete with an earlier job for GPU0–3.

1. **Completed, but not comparable with the compiled historical baseline: 10s/768p global-reweight component ablation, 50 prompts.** Four arms (`full`, `weights_only`, `bias_only`, `none`), FP32 anchor/comfy-FP32 summary/pre-round log-mass. Runner: `scripts/diffusers_spark_reweight_components_50prompt_20260926.py`. All 200 videos, paired PSNR/SSIM/LPIPS, and source-assigned core-five VBench are complete; 55 exact-match outputs were reused and 145 newly generated. The inherited helper explicitly disabled `torch.compile`; see the warning in `docs/diffusers_reweight_components_50prompt_20260926.md`.
2. **Paused at user request: 10s/768p approximate-tail granularity, 50 prompts.** BF16 `query` vs `block`, otherwise a single shared setting. Runner: `scripts/diffusers_spark_tail_granularity_50prompt_20260927.py`. Its GPU workers were stopped before any new inference records were written; only the prepared/reused 10+10 records remain. Its explicit no-compile baseline is not comparable with the compiled historical Spark run. Do not resume or launch the following queued job before the compile-on attribution and validation are complete.
3. **Dense reference + three Spark route arms, 25 prompts — wait for jobs 1 and 2:** use **5s/768p, seed42, 20 requested steps** and the **current default Spark-H3-10pct settings**, including fanout16. Generate **exactly four arms from scratch**: `dense`, Spark `threshold`, original Spark `packed_external`, and new Spark `packed_external_no_route_qk`. **Do not run `fused`.** Among Spark arms, only route execution/selector implementation may differ; the new selector must skip the unused in-kernel route-centroid TMA read and QK GEMM while preserving the packed route result. The parent agent implemented this as an opt-in path without changing the first two live experiments; CPU static checks passed but GPU compilation/parity/performance remain unverified. The dense PSNR reference is generated **afresh in this experiment**, not from archived fanout64 jobs. The Benchmark must run its **real default `torch.compile` hook** for all arms; validate the actual wrapped-block count before launch and record it for each case in the protocol/results. After jobs 1 and 2 finish, the parent agent will first update the old route helper's hardcoded `--no-torch-compile`/fanout64 settings to the current defaults; do not edit that live dependency earlier. Report paired denoising speed after full warmup and paired **PSNR only** against the newly generated dense videos. The user has explicitly excluded SSIM, LPIPS and VBench for this third job. Do not reuse any fanout64 tail-ablation or route outputs: the parent agent moved four historical experiment/output directory pairs to `/autodl-fs/data/.trash_spark_20260927/all_fanout64_20260927/` and the old route10 pair to adjacent archive.

The 25-prompt route test follows jobs 1 and 2, a focused attribution audit
of the large old-canonical-vs-new-`full` 50-prompt PSNR gap, **and** the old
helper's safe post-run migration. The old canonical50 case1 PSNR is 22.147 dB
versus 16.783 dB for the new `full` case1 with the same dense-reference video
SHA; the old and new runs must not be described as numerically aligned until
model weights, route/compile/kernel and numeric-path differences are traced.
Before launch, compare the **three Spark** configurations
field by field (route execution/selector alone may differ), recheck effective
fanout16, and verify real compile wrapping. Test the new selector's packed
routes and attention output against original `packed_external` under identical
inputs; require numerical parity or a documented bounded tolerance. Profile a
single denoise case to verify the unused TMA route-centroid load and in-kernel
route-QK GEMM are actually absent and that speed improves. Use a single GPU
first for CuTe compilation, route-mask parity, attention-output elementwise
parity/tolerance, and small-scale timing; repair the opt-in kernel if it fails.
Run one full-denoise
smoke case for each of the four required arms; check final Spark configs,
`compile_wrapped_blocks`, and completion of one excluded full warmup per arm.
Only then launch the 25-prompt batch, without another approval.
