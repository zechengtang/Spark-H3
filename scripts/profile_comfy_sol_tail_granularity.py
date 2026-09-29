"""One captured 10s768p attention layer, identical route with two tail modes.

Not a video benchmark: error metrics are attention-output comparisons to SDPA.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from comfy_kitchen.backends import cuda as ck
from profile_diffusers_vs_comfy_spark_20260925 import load_capture, cuda_benchmark
from probe_comfy_summary_gemm_20260925 import metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reverse',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    data,q,k,v=load_capture(torch)
    sinks=[data['layout']['video_tokens']//64,(q.shape[1]+63)//64]
    calls={}
    for selection,opts in [('tau1',dict(tau=1.)),('topk10',dict(topk_ratio=.1))]:
        for granularity in ('block','query'):
            calls[f'{selection}_{granularity}']=lambda g=granularity,o=opts: ck.sol_attn(
                q,k,v,tail_granularity=g,sink_blocks=sinks,sink_q=sinks,**o)
    order=list(calls)
    if args.reverse: order.reverse()
    result=dict(shape=list(q.shape),video_tokens=int(data['layout']['video_tokens']),
                gpu=torch.cuda.get_device_name(),timings={},attention_vs_dense={},output_sha256={})
    for name in order:
        result['timings'][name]=cuda_benchmark(torch,calls[name])
        print(name,result['timings'][name]['median_ms'],flush=True)
    # Force fused flash SDPA; never silently allocate a quadratic math reference.
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        dense=torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2).contiguous()
    for name,fn in calls.items():
        out=fn()
        result['attention_vs_dense'][name]=metrics(dense,out)
        result['output_sha256'][name]=hashlib.sha256(out.view(torch.uint8).cpu().numpy()).hexdigest()
        del out
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
