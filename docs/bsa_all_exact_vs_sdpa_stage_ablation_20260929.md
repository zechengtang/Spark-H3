# BSA all-exact versus ComfyUI SDPA: stage ablation

Measured on 2026-09-29 with one NVIDIA RTX PRO 6000 Blackwell, PyTorch
2.12.1+cu130, BF16 H3 attention input, 56 heads and head dimension 128.
The 10 s case replays captured real post-RoPE Q/K/V with 73,565 tokens.
The shorter case takes the first 37,897 tokens of that capture to test the
5 s sequence length; it is a length ablation, not a separate 5 s generation.

Both paths start from the **same post-RoPE Q/K/V**. Thus QKV projection,
RMSNorm, RoPE, model loading and the output projection are outside this test.
BSA calls `ck.sol_attn(q,k,v,tau=-1e9,tail=False)`; SDPA calls ComfyUI's
default `comfy.ops.scaled_dot_product_attention`. The observed SDPA CUDA
kernel is `pytorch_flash::flash_fwd_kernel` in both sizes.

CUDA event medians (12 warmed calls per row):

| Tokens | BSA all-exact | Flash SDPA raw | Flash SDPA plus output layout | Layout alone | BSA speedup over raw SDPA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 37,897 | 92.26 ms | 114.38 ms | 114.53 ms | 0.0067 ms | 1.24x |
| 73,565 | 322.54 ms | 428.26 ms | 430.81 ms | 0.0070 ms | 1.33x |

The SDPA output is already stored in a layout for which the final
transpose/contiguous conversion is a view, hence the near-zero separate
layout cost. The 2.5 ms difference between raw and converted 10 s medians
is run-order / GPU-clock variation, not a measured copy kernel: the SDPA
profile contains only one Flash kernel.

CUDA profiler stage times (one warmed call; useful for attribution, while
CUDA events above are the primary total-latency measurement):

| Tokens | BSA prepare / INT8 quantize / V transpose | BSA route | BSA INT8 exact kernel | SDPA BF16 Flash kernel |
| ---: | ---: | ---: | ---: | ---: |
| 37,897 | 2.62 ms | 0.14 ms | 88.19 ms | 112.50 ms |
| 73,565 | 5.12 ms | 0.45 ms | 313.73 ms | 431.61 ms |

The speed difference is located in the attention compute kernel. BSA's
preprocessing and route add time; they do not explain the win. The exact
kernel source explicitly uses all-INT8 MMA (`sol_attn_exact.cu`). This
comparison does **not** isolate INT8 arithmetic from BSA's distinct tiling,
memory layout, work scheduling and softmax implementation. A matched BF16
version of the BSA exact kernel would be required to attribute a numeric
fraction of the gain to INT8 alone; no such variant exists in this build.

Raw measurements:
`/autodl-fs/data/h3_experiments/bsa_all_exact_vs_sdpa_20260929/stages_10s_768p.json`
and
`/autodl-fs/data/h3_experiments/bsa_all_exact_vs_sdpa_20260929/stages_5s_length_slice.json`.
Reproduction script: `scripts/profile_bsa_all_exact_vs_sdpa_stages_20260929.py`.
