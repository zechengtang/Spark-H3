"""Check existing CUDA RMS/RoPE direct destinations; no production changes.

Standalone post-projection microbenchmark, NOT an end-to-end speed estimate.
Run only on an otherwise idle GPU to avoid contaminating pipeline timings.
"""
import argparse
import json
import statistics
from pathlib import Path
import torch
from comfy_kitchen.backends import cuda as ck

def direct(q,k,f,qw,kw,qo,ko,epsilon,rot):
    wrap=ck._wrap_for_dlpack
    ck._C.rms_rope(*(wrap(t) for t in (q,k,f,qw,kw,qo,ko)),epsilon,
                   torch.cuda.current_stream(q.device).cuda_stream,True,rot)

def run(tokens,rot):
    h,d=56,128;inner=h*d
    torch.manual_seed(42)
    packed=torch.randn(tokens,inner*3,device='cuda',dtype=torch.bfloat16)
    f=torch.randn(1,tokens,1,rot//2,2,2,device='cuda',dtype=torch.float32)
    qw=torch.randn(d,device='cuda',dtype=torch.bfloat16);kw=torch.randn_like(qw)
    outputs=[tuple(torch.empty(1,tokens,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)) for _ in range(2)]
    # Restore the input because the old implementation modifies projected Q/K.
    scratch=packed.clone()
    def execute(new):
        oq,ok,ov=outputs[int(new)]
        for start in range(0,tokens,4096):
            stop=min(tokens,start+4096);n=stop-start
            q,k,v=scratch[start:stop].split(inner,dim=-1)
            q=q.view(1,n,h,d);k=k.view(1,n,h,d)
            if new:direct(q,k,f[:,start:stop],qw,kw,oq[:,start:stop],ok[:,start:stop],1e-6,rot)
            else:
                ck.rms_rope_split_half_(q,k,f[:,start:stop],qw,kw,epsilon=1e-6,rot_dim=rot)
                oq[:,start:stop].copy_(q);ok[:,start:stop].copy_(k)
            ov[:,start:stop].copy_(v.view(1,n,h,d))
    for new in (False,True):scratch.copy_(packed);execute(new)
    torch.cuda.synchronize()
    equal=[torch.equal(a,b) for a,b in zip(*outputs)]
    if not all(equal):raise AssertionError(f'Not bitwise equal: {tokens=} {rot=} {equal}')
    samples={False:[],True:[]}
    for i in range(16):
        for new in ((False,True) if i%2 else (True,False)):
            scratch.copy_(packed)
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();execute(new);end.record();end.synchronize()
            if i>=4:samples[new].append(begin.elapsed_time(end))
    return dict(tokens=tokens,rot=rot,bitwise_equal=equal,
                old_ms=statistics.median(samples[False]),direct_ms=statistics.median(samples[True]),
                samples={str(k):v for k,v in samples.items()})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tokens',type=int,nargs='+',default=[37887,73565])
    parser.add_argument('--rot',type=int,nargs='+',default=[128])
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    torch.set_num_threads(4)
    result=dict(records=[run(t,r) for t in args.tokens for r in args.rot])
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
