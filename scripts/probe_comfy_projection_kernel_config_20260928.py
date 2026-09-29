"""Tune the genuine 10s full-token ComfyUI projection without changing code."""

import json
from pathlib import Path
import statistics
import sys
import torch
import triton

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from profile_diffusers_vs_comfy_spark_20260925 import load_capture
from h3_sparse_attention.landmark_projection import _bthd_projection_kernel


OUTPUT=Path("/autodl-fs/data/h3_experiments/comfyui_projection_kernel_config_20260928/results.json")


def run():
    torch.set_num_threads(4)
    _,q,_,_=load_capture(torch)
    batch,tokens,heads,dim=q.shape
    factor=torch.eye(dim,device=q.device,dtype=torch.bfloat16).expand(batch*heads,-1,-1).contiguous()
    directions=torch.randn((batch*heads,15,dim),device=q.device,dtype=torch.float32)
    out=torch.empty((batch*heads,tokens,dim),device=q.device,dtype=torch.bfloat16)
    scores=torch.empty((batch*heads,tokens,15),device=q.device,dtype=torch.float32)
    records={};baseline=None
    for tile,warps,stages in ((128,8,3),(128,4,3),(128,16,3),(64,4,3),(64,8,3),
                               (128,4,2),(128,8,2)):
        def work():
            _bthd_projection_kernel[(triton.cdiv(tokens,tile),batch*heads)](
                q,factor,out,directions,scores,tokens,heads,tile,True,
                factor.stride(0),factor.stride(1),factor.stride(2),
                num_warps=warps,num_stages=stages)
        for _ in range(5):work()
        torch.cuda.synchronize()
        samples=[]
        for _ in range(16):
            begin=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            begin.record();work();end.record();end.synchronize()
            samples.append(begin.elapsed_time(end))
        if baseline is None:baseline=(out.clone(),scores.clone())
        records[f"tile{tile}_warps{warps}_stages{stages}"]=dict(
            median_ms=statistics.median(samples),samples_ms=samples,
            same_output=torch.equal(out,baseline[0]),same_scores=torch.equal(scores,baseline[1]))
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(shape=list(q.shape),records=records)
    OUTPUT.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":run()
