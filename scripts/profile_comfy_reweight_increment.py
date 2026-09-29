"""Matched route/consumer pooled-KV vs weighted-KV ablation, single real layer."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from comfy_kitchen.backends import cuda as ck
from profile_diffusers_vs_comfy_spark_20260925 import (
    configure_environment, cuda_benchmark, load_capture, profiler_rows,
    layout_kwargs, REPOSITORY,
)
import sys
sys.path.insert(0, str(REPOSITORY))
from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
from comfyui_reblock_plan import build_comfy_reblock_permutations


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reverse',action='store_true')
    args=p.parse_args()
    configure_environment()
    torch.set_num_threads(4)
    data,q,k,v=load_capture(torch)
    layout=ComfyPackedLayout(**layout_kwargs(torch,data))
    controller=ComfySparkController(ComfySparkConfig(landmark_tree_v2_children=16))
    controller.evaluation_index=5
    qp,qi,kp,_=build_comfy_reblock_permutations(controller,q,k,layout)
    sinks=[layout.video_tokens//64,(q.shape[1]+63)//64]
    native=ck._C.spark_attn
    calls={}
    for rb in (False,True):
        for grain in ('block','query'):
            for rw in (False,True):
                def call(rb=rb,grain=grain,rw=rw):
                    def entry(*a,**kw): return native(*a,**kw,reweight=rw)
                    with patch.object(ck._C,'spark_attn',entry):
                        return ck.spark_attn(q,k,v,video_tokens=layout.video_tokens,
                            topk_ratio=.1,anchor_dtype=torch.float32,sink_blocks=sinks,sink_q=sinks,
                            tail_granularity=grain,query_permutation=qp if rb else None,
                            key_permutation=kp if rb else None)
                calls[f'{grain}/reblock={rb}/reweight={rw}']=call
    calls['reblock_plan_only']=lambda: build_comfy_reblock_permutations(controller,q,k,layout)
    for grain in ('block','query'):
        def tau_call(grain=grain):
            def entry(*a,**kw): return native(*a,**kw,reweight=False)
            with patch.object(ck._C,'spark_attn',entry):
                return ck.spark_attn(q,k,v,video_tokens=layout.video_tokens,
                    topk_ratio=0.,tau=1.,sink_blocks=sinks,sink_q=sinks,
                    tail_granularity=grain)
        calls[grain+'/tau1_matched']=tau_call
        def full_call(grain=grain):
            fresh_qp,_,fresh_kp,_=build_comfy_reblock_permutations(controller,q,k,layout)
            return ck.spark_attn(q,k,v,video_tokens=layout.video_tokens,
                topk_ratio=.1,sink_blocks=sinks,sink_q=sinks,tail_granularity=grain,
                query_permutation=fresh_qp,key_permutation=fresh_kp)
        calls[grain+'/full_with_plan']=full_call
    calls['official_sol_tau1']=lambda: ck.sol_attn(q,k,v,tau=1.,sink_blocks=sinks,sink_q=sinks)
    order=list(calls)
    if args.reverse: order.reverse()
    results=dict(shape=list(q.shape),video_tokens=layout.video_tokens,
        scope='single attention; cached plan excluded; fused reblock included when enabled',
        baseline='same one-pass selector and consumer; ordinary pooled KV summaries',
        timings={},profiles={})
    for name in order:
        results['timings'][name]=cuda_benchmark(torch,calls[name])
        print(name,results['timings'][name]['median_ms'],flush=True)
    for name in order:
        results['profiles'][name]=profiler_rows(torch,calls[name],name)
    args.output.write_text(json.dumps(results,indent=2))


if __name__=='__main__': main()
