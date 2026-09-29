"""Original BSA producer + BF16 handoff, same-input chunk A/B and kernel trace."""

import json
from pathlib import Path

import torch
from comfy_kitchen.backends import cuda as ck
from profile_diffusers_vs_comfy_spark_20260925 import cuda_benchmark, load_capture


def main():
    _, q, k, v = load_capture(torch)
    t, h, d = 16384, q.shape[2], q.shape[3]
    q, k, v = (x[:, :t].contiguous() for x in (q, k, v))
    qkv = torch.cat([x.reshape(t, h * d) for x in (q, k, v)], -1)
    theta = torch.arange(t, device="cuda", dtype=torch.float32)[:, None] * 0.0001
    theta = theta.expand(t, 48)
    freqs = torch.zeros((1, t, 1, 48, 2, 2), device="cuda")
    freqs[0, :, 0, :, 0, 0] = theta.cos()
    freqs[0, :, 0, :, 0, 1] = -theta.sin()
    freqs[0, :, 0, :, 1, 0] = theta.sin()
    freqs[0, :, 0, :, 1, 1] = theta.cos()
    fab = ck._packed_rope_fab(freqs, t, 96)
    qw = torch.ones(d, device="cuda", dtype=torch.bfloat16)
    kw = qw.clone()
    stale_kmean = torch.zeros((h, d), device="cuda", dtype=torch.float32)
    stale_vscale = torch.ones_like(stale_kmean)
    p = ck._C.sol_attn_plan(1, t, h)
    ws = torch.empty(p["total"], device="cuda", dtype=torch.uint8)
    out = [torch.empty_like(q) for _ in range(3)]
    ref = [torch.empty_like(q) for _ in range(3)]
    qi, ki, vi = qkv.split(h * d, -1)
    qi, ki = (x.view(1, t, h, d) for x in (qi, ki))
    wrap = ck._wrap_for_dlpack
    stream = torch.cuda.current_stream().cuda_stream

    def candidate():
        ck._C.sol_producer_begin(wrap(ws), 1, t, h, stream, 0)
        ck._C.sol_producer_chunk_materialize(
            *(wrap(x) for x in (ws, qkv, fab, qw, kw,
                                 stale_kmean, stale_vscale, *out)),
            1e-6, 96, 0, t, t, h, stream)

    def baseline():
        ck._C.rms_rope(*(wrap(x) for x in (qi, ki, freqs, qw, kw, ref[0], ref[1])),
                       1e-6, stream, True, 96)
        ref[2].copy_(vi.view(1, t, h, d))

    candidate(); baseline(); torch.cuda.synchronize()
    parity = {}
    for name, actual, expected in zip("qkv", out, ref):
        delta = (actual.float() - expected.float()).abs()
        parity[name] = {"equal_fraction": float((actual == expected).float().mean()),
                        "max_abs": float(delta.max()), "rmse": float(delta.square().mean().sqrt())}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        candidate(); torch.cuda.synchronize()
    kernels = [{"name": row.key, "cuda_ms": float(row.self_device_time_total)/1000}
               for row in prof.key_averages() if row.self_device_time_total > 0]
    result = {"shape": [1,t,h,d], "workspace_bytes": int(p["total"]),
              "parity": parity, "kernels": kernels,
              "candidate_ms": cuda_benchmark(torch,candidate,warmup=5,iterations=20),
              "baseline_ms": cuda_benchmark(torch,baseline,warmup=5,iterations=20)}
    path = Path('/autodl-fs/data/h3_experiments/comfyui_bsa_original_producer_20260928/chunk_gpu2.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps({"workspace_bytes":result["workspace_bytes"],"parity":parity,
                      "candidate_ms":result["candidate_ms"]["median_ms"],
                      "baseline_ms":result["baseline_ms"]["median_ms"],
                      "producer_kernel_seen":any("sol_producer_kernel" in x["name"] for x in kernels)}),flush=True)


if __name__ == "__main__":
    main()
