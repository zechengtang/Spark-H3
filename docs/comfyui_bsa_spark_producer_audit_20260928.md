# Official BSA producer → Spark-H3 adaptation audit (2026-09-28)

## Data dependency

The official `sol_attn_chunked` producer projects one QKV chunk, applies
RMSNorm/RoPE, and writes INT8 attention carriers directly. It does not retain
full BF16 Q/K/V. Its K centering vector and V quantization scale are from the
previous evaluation; the first evaluation runs two producer passes to obtain
statistics. By contrast, Spark's current `launch_spark_attn` reads full BF16
Q/K/V, computes the *current* K mean and V scale, and then quantizes. Spark's
reblock plan requires the current BF16 Q/K; global reweight additionally
requires the current BF16 K/V. Therefore, reusing the official quantized
carriers or invoking its route/exact core is not a faithful implementation
substitution. Prior stale K/V quantization would change rounding and can
change routing/output. Passing current statistics requires first materializing
data and a statistics pass, removing the one-pass advantage.

An architecture that retains current numerical semantics would project each
QKV chunk once and write full BF16 Q/K/V for reblock/reweight. The current
reblock plan then uses those full Q/K tensors to compute a **new per-head query
permutation**. Only after that plan exists can the Q carriers and 64-token
centroids be formed in the order consumed by route/exact. `prep_q` also uses
current-step K variance for its threshold. Thus even the Q side cannot be
folded wholesale into the official producer before reblock. Producing
per-token Q carriers early would require a second reorder plus a post-plan
centroid/threshold pass, while keeping the full BF16 Q write and subsequent
read. This is a different multi-kernel design, not a direct BSA graft.

## Real QKV kernel attribution

Runner: `scripts/probe_bsa_spark_reuse_ceiling_20260928.py`.
Raw: `/autodl-fs/data/h3_experiments/comfyui_bsa_spark_reuse_ceiling_20260928/gpu{2,3}.json`.
Input: captured 10s/768p real post-RoPE Q/K/V at
`/autodl-fs/data/h3_experiments/attention_path_optimization_20260917/attention_input_gpu1.pt`,
shape `[1,73565,56,128]`, BF16. Both GPUs used TopK10, block tail, reweight
enabled, `video_tokens=72576`, and sink block range `[1134,1150]`.
**This is a core-only replay:** no QKV
projection, reblock plan, reblock permutations, or direct output scatter.
`force_local_blocks=True` is fixed for the attribution. Thus the total call
time must not be quoted as production end-to-end latency; only the individual
preprocess kernels bound producer fusion opportunity.
The September 17 capture has its own 73,565-token input and is not an exact
input/shape match to the September 28 full ComfyUI 10s profile. The
784-call multiplication below only scales a measured kernel time; it is
neither a measured saving nor an attainable producer-fusion bound.

The raw per-GPU JSON now includes the capture SHA256
`8bbe2b854add2d01dd79c02b182123161962d6b43177d4a81968c187c3df6cac`,
the loaded `comfy-kitchen` CUDA extension SHA256
`d1c858ea49ef96441a2be3f88c5d1f32bd44b26831ddcdbaa8ea46dcb0a1ef41`,
its absolute path, and the SHA256 of the relevant CUDA source files. The
`comfy-kitchen` Git HEAD was `f61028a7b0f4be4beb3595c64ee919e0c345f96d`;
the worktree had modified `sol_attn.cu`, so the source hashes and loaded
binary hash, rather than the commit alone, identify the measured version.

| Measurement | GPU2 | GPU3 |
| --- | ---: | ---: |
| Spark full core median | 48.570 ms | 47.974 ms |
| `prep_q` active CUDA time | 1.255 ms | 1.254 ms |
| `prep_reduce_kv` | 1.494 ms | 1.494 ms |
| `prep_k` | 1.225 ms | 1.234 ms |
| `vquant_transpose` | 1.120 ms | 1.121 ms |
| Reweight summary quant | 1.524 ms | 1.526 ms |
| Exact kernel | 39.796 ms | 39.026 ms |

`prep_q` took about 1.25 ms. Multiplying it by 784 sparse attention calls
gives 0.98 s of *measured kernel activity*, **not** a saving estimate: the
producer runs before the reblock permutation and cannot eliminate that full
kernel without replacement work. The K/V passes likewise cannot be removed
while using current statistics. The official speed advantage comes mainly
from avoiding BF16 Q/K/V materialization, which Spark's reblock/reweight
require.

To test the minimum replacement work for early Q quantization, the runner
`scripts/probe_bsa_qcarrier_reorder_20260928.py` built a real reblock plan
from the captured Q/K, then gathered an INT8 Q carrier through its per-head
query permutation. It moved 1.040 GB (read plus write), passed sampled
index-contract checks, and took **0.803/0.802 ms median** on GPU2/3. This
excludes Q scales, post-reblock centroids and Q means, threshold, and the
extra producer quantization itself. It therefore consumes most of the 1.25 ms
`prep_q` activity before performing its remaining work. Raw results:
`qcarrier_reorder_gpu{2,3}.json` in the same audit directory. This is a
measured lower-bound substep, not an implemented end-to-end candidate.

The already screened BSA-style fused BF16 materializer combines RMSNorm/RoPE
and V copy. On one 16K projected-QKV chunk it saved only 0.166 ms (1.162 →
0.995 ms) before packed-RoPE preparation. Q/K differed in some BF16 elements
(max absolute difference 0.0625/0.03125); V was bitwise equal. With about
five chunks per call, even its optimistic 784-call saving is ~0.65 s before
overhead, and it does not preserve the existing numerical implementation.
Its raw A/B is `fused_materializer_gpu{0,1}.json` in
`/autodl-fs/data/h3_experiments/comfyui_sol_topk_sparse_step_20260928/`.

**Decision:** Do not swap the official BSA producer into Spark's default.
The direct graft changes quantization semantics and lacks the future reblock
permutation. The attempted early-Q-carrier design requires a measured
0.8 ms reorder plus unmeasured remaining work, so no faithful, validated
speed candidate exists yet. The prior full Q/K/V fusion screen is both small
and non-bitwise. No production source was changed in this audit.

Follow-up: an opt-in direct graft subsequently ran the actual original
`sol_producer_kernel`, retained its INT8 carrier writes, added BF16 Q/K/V
outputs, and passed those to the full Spark backend. It regressed complete
20-step timing on GPU2 and GPU3; see
`docs/comfyui_original_bsa_producer_spark_full_ab_20260928.md`. The direct
graft is still disabled by default.
