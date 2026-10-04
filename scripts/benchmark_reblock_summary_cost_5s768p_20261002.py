#!/usr/bin/env python3
"""Full-56-head shape cost probe, separate from generation-quality evidence."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("screen",REPO/"scripts/ablate_reblock_approx_5s768p_20261002.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)
sys.path.insert(0,str(screen.ROOT/"source"))


def run():
    from h3_sparse_attention.landmark_direction import landmark_direction_factors
    from h3_sparse_attention.mahalanobis_kmeans import hilbert_midpoint_sample_indices
    from h3_sparse_attention.sol_numerator_virtual_q import build_virtual_anchors,virtual_summaries
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    path = screen.ROOT/"captures"/"case01_eval04_layer25.pt"
    cap = torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    # Repeat captured heads only to populate the actual 56-head tensor shape.
    # No quality claims use these duplicated features.
    q,k,v = [cap[n].cuda().repeat(7,1,1) for n in ("q","k","v")]
    base = screen.audit.CurrentRoute()
    qp,kp,_ = base.production_partition(q,k,cap["grid"],base.cfg)
    h,n,d = q.shape
    cut = n//64*64
    qpack = q.gather(1,qp[...,None].expand(-1,-1,d))
    kpack,vpack = [x.gather(1,kp[...,None].expand(-1,-1,d)) for x in (k,v)]
    qt,kt,vt = [x[:,:cut].transpose(0,1)[None].contiguous() for x in (qpack,kpack,vpack)]
    ranges = torch.tensor([[0,cut]],device="cuda",dtype=torch.long)
    local_ranges = torch.stack((torch.arange(0,cut,64,device="cuda"),torch.arange(64,cut+1,64,device="cuda")),-1)
    a = build_virtual_anchors(qt,ranges)
    local_a = build_virtual_anchors(qt,local_ranges)
    idx = hilbert_midpoint_sample_indices(tuple(cap["grid"]),n//64,device="cuda")
    qm,km = landmark_direction_factors(q,k,idx,ridge=.001,moment="raw")
    qf,kf = torch.bmm(q,km.to(q.dtype)),torch.bmm(k,qm.to(k.dtype))
    def swap_both():
        return screen.swap_refine(qf,qp),screen.swap_refine(kf,kp)
    functions = {
        "production_reblock":lambda:base.production_partition(q,k,cap["grid"],base.cfg),
        "swap_both4_precomputed_features_reference":swap_both,
        "global_anchor_construction":lambda:build_virtual_anchors(qt,ranges),
        "local_q64_anchor_construction":lambda:build_virtual_anchors(qt,local_ranges),
        "global_summary_native":lambda:virtual_summaries(a,kt,vt),
        "local_q64_summary_native_materialized":lambda:virtual_summaries(local_a,kt,vt),
        "two_k32_summary_torch_reference":lambda:screen.two_summaries(kpack,vpack,a[0].float()),
    }
    measurements = {}
    with torch.inference_mode():
        for name,fn in functions.items():
            for _ in range(3):
                result = fn()
                del result
            torch.cuda.synchronize()
            times = []
            for _ in range(10):
                first,last = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                first.record()
                result = fn()
                last.record()
                last.synchronize()
                times.append(first.elapsed_time(last))
                del result
            measurements[name] = {"median_ms":statistics.median(times),"min_ms":min(times),"max_ms":max(times),"samples":times}
            print(name,measurements[name]["median_ms"],flush=True)
    payload = {"status":"complete","measurements":measurements,"heads":h,"video_tokens":n,
               "full_k64_blocks":cut//64,"local_anchors":local_a.shape[1],"physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),
               "runner_sha256":screen.sha(Path(__file__)),"screen_source_sha256":screen.sha(Path(screen.__file__)),
               "global_summary_bytes":h*(cut//64)*(2*d*2+4),
               "local_materialized_summary_bytes":h*(cut//64)**2*(2*d*2+4),
               "notes":["full56-head shape populated by repeating captured8-head features; quality not measured",
                        "three warmups, ten CUDA-event measurements; loading excluded",
                        "local summaries materialized at once, unlike the production bounded streamed consumer",
                        "swap excludes factor/transform work; two-K32 is unfused Torch; no end-to-end speed inference"]}
    screen.write(screen.ROOT/"cost_probe.json",payload)


if __name__ == "__main__":
    run()
