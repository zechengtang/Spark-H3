"""Isolated block8x8 skipped-tail latency without model loading."""

import json
import torch
import triton

from h3_sparse_attention.sol_numerator_virtual_q import (
    _summaries_block8, _skip_merge_chunk_block8,
)


def bench(tokens, heads, iterations=10):
    device = "cuda"
    t = tokens
    n = triton.cdiv(t, 64)
    n8 = triton.cdiv(t, 8)
    q, k, v = (torch.randn((1, t, heads, 128), device=device, dtype=torch.bfloat16)
               for _ in range(3))
    anchor = q.float().mean(1, keepdim=True).to(torch.bfloat16).contiguous()
    ak = torch.empty((1, 1, heads, n8, 128), device=device, dtype=q.dtype)
    av = torch.empty_like(ak)
    lm = torch.empty((1, 1, heads, n8), device=device, dtype=torch.float32)
    mapping = torch.zeros((n,), device=device, dtype=torch.int64)
    route = torch.zeros((1, n, heads, n), device=device, dtype=torch.uint8)
    exact = torch.randn_like(q)
    exact_lse = torch.randn((1, t, heads), device=device, dtype=torch.float32)

    def summary():
        _summaries_block8[(1, n8, heads)](
            anchor, k, v, ak, av, lm, t, heads, n8, 1,
            anchor.stride(0), anchor.stride(1), 0, 1, heads, 0,
            False, True, True, num_warps=4)

    def merge():
        _skip_merge_chunk_block8[(n, 8, heads)](
            q, mapping, ak, av, lm, route, exact, exact_lse,
            t, heads, n, n8, 1, 0, 0, num_warps=4, num_stages=1)

    summary()
    merge()
    torch.cuda.synchronize()
    return dict(tokens=t, heads=heads,
                summary_ms=triton.testing.do_bench(summary, warmup=100, rep=iterations*10),
                merge_ms=triton.testing.do_bench(merge, warmup=100, rep=iterations*10))


if __name__ == "__main__":
    print(json.dumps(bench(8192, 16, 5)), flush=True)
    print(json.dumps(bench(37897, 16, 5)), flush=True)
