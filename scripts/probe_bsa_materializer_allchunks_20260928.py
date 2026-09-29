"""Check every 16K materializer chunk and offset using captured real QKV values."""

import json
from pathlib import Path

import torch
from comfy_kitchen.backends import cuda as ck
from profile_diffusers_vs_comfy_spark_20260925 import load_capture


def main():
    _, q, k, v = load_capture(torch)
    t, h, d = q.shape[1:]
    qout, kout, vout = (torch.empty_like(q) for _ in range(3))
    qw = torch.ones(d, dtype=torch.bfloat16, device="cuda")
    kw = qw.clone()
    wrap = ck._wrap_for_dlpack
    stream = torch.cuda.current_stream().cuda_stream
    rows = []
    for start in range(0, t, 16384):
        stop = min(start + 16384, t)
        n = stop - start
        projected = torch.cat([x[:, start:stop].reshape(n, h * d) for x in (q, k, v)], -1)
        theta = torch.arange(n, device="cuda", dtype=torch.float32)[:, None] * 0.0001
        theta = theta.expand(n, 48)
        freqs = torch.zeros((1, n, 1, 48, 2, 2), device="cuda")
        freqs[0, :, 0, :, 0, 0] = theta.cos()
        freqs[0, :, 0, :, 0, 1] = -theta.sin()
        freqs[0, :, 0, :, 1, 0] = theta.sin()
        freqs[0, :, 0, :, 1, 1] = theta.cos()
        fab = ck._packed_rope_fab(freqs, n, 96)
        ck._C.bsa_materialize_qkv_chunk(
            *(wrap(x) for x in (projected, fab, qw, kw, qout, kout, vout)),
            1e-6, 96, start, n, t, h, stream)
        qi, ki, vi = projected.split(h * d, -1)
        qi, ki = (x.view(1, n, h, d) for x in (qi, ki))
        qr, kr = torch.empty_like(qi), torch.empty_like(ki)
        ck._C.rms_rope(*(wrap(x) for x in (qi, ki, freqs, qw, kw, qr, kr)),
                       1e-6, stream, True, 96)
        torch.cuda.synchronize()
        row = {"start": start, "tokens": n}
        for name, actual, ref in (("q", qout[:, start:stop], qr),
                                  ("k", kout[:, start:stop], kr),
                                  ("v", vout[:, start:stop], vi.view(1, n, h, d))):
            delta = (actual.float() - ref.float()).abs()
            row[name] = {"equal_fraction": float((actual == ref).float().mean()),
                         "max_abs": float(delta.max()), "rmse": float(delta.square().mean().sqrt())}
        rows.append(row)
        print(json.dumps(row), flush=True)
    path = Path('/autodl-fs/data/h3_experiments/comfyui_bsa_spark_reuse_ceiling_20260928/allchunks_gpu2.json')
    path.write_text(json.dumps({"shape": [1, t, h, d], "chunks": rows}, indent=2))


if __name__ == "__main__":
    main()
