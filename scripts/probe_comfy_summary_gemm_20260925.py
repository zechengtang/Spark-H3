"""Paired, same-binary summary-builder experiment; no Diffusers imports."""
import argparse
import json
import os
from pathlib import Path

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


def metrics(ref, actual):
    # Chunk metrics so the FP32 intermediates don't occupy multiple GB.
    r, a = ref.flatten(), actual.flatten()
    stats = torch.zeros(5, device=r.device, dtype=torch.float64)
    maxerr = torch.zeros((), device=r.device)
    for i in range(0, r.numel(), 1048576):
        x, y = r[i:i+1048576].double(), a[i:i+1048576].double()
        delta = x - y
        stats += torch.stack(((delta * delta).sum(), (x*x).sum(),
                              (y*y).sum(), (x*y).sum(), (x != y).sum())).double()
        maxerr = torch.maximum(maxerr, delta.abs().max())
    err, xx, yy, xy, changed = stats.tolist()
    import math
    return dict(mse=err/r.numel(), max_abs=float(maxerr),
                relative_l2=math.sqrt(err/max(xx, 1e-30)),
                cosine=xy/max(math.sqrt(xx*yy), 1e-30),
                changed_fraction=changed/r.numel(),
                signal_to_error_db=10*math.log10(xx/err) if err else None,
                finite=bool(torch.isfinite(actual).all()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', default='0,1,1,0')
    p.add_argument('--experiment', choices=('summary', 'tail'), default='summary')
    args = p.parse_args()
    env_name = 'COMFY_SPARK_SUMMARY_GEMM' if args.experiment == 'summary' else 'COMFY_SPARK_TAIL_FIRST'
    configure_environment()
    torch.set_num_threads(4)
    data, q, k, v = load_capture(torch)
    layout = ComfyPackedLayout(**layout_kwargs(torch, data))
    ctrl = ComfySparkController(ComfySparkConfig(topk_ratio=.1, tau=1.,
              global_anchor_dtype='float32', landmark_tree_v2_children=16))
    ctrl.evaluation_index = 5
    qp, qi, kp = build_comfy_reblock_permutations(ctrl, q, k, layout)[:3]
    sinks = [layout.video_tokens//64, (q.shape[1]+63)//64]
    def call(reblock=False):
        opts = dict(query_permutation=qp, query_inverse=qi, key_permutation=kp) if reblock else {}
        return ck.spark_attn(q, k, v, video_tokens=layout.video_tokens,
                            anchor_dtype=torch.float32, topk_ratio=.1,
                            sink_blocks=sinks, sink_q=sinks, **opts)
    def baseline():
        b, t, h, d = q.shape
        ws = torch.empty(ck._C.sol_attn_plan(b,t,h,token_aug=0)['total'], device=q.device, dtype=torch.uint8)
        out = torch.empty_like(q)
        w = ck._wrap_for_dlpack
        ck._C.sol_attn(w(q),w(k),w(v),w(out),w(ws),b,t,h,d,1.,d**-.5,
                      *sinks,*sinks,torch.cuda.current_stream().cuda_stream,
                      key_bias=None,threshold=None,block_len=None,tail=False,
                      token_aug=0,topk_count=ck._topk_count(sinks[0],.1))
        return out
    def repeated_profile(fn, label):
        def repeated():
            for _ in range(8):
                out = fn()
                del out
        rows = profiler_rows(torch, repeated, label)
        for row in rows:
            row['per_call_device_ms'] = row['self_device_ms'] / 8
        return rows
    result = dict(experiment=args.experiment, shape=list(q.shape), gpu=torch.cuda.get_device_name(),
                  visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'), runs=[], accuracy={}, profiles={})
    for mode in map(int, args.modes.split(',')):
        os.environ[env_name] = str(mode)
        row = dict(mode=mode, baseline=cuda_benchmark(torch,baseline),
                   spark=cuda_benchmark(torch,call),
                   fused=cuda_benchmark(torch,lambda:call(True)))
        # Exact-only is a diagnostic timing, not a valid reweight baseline.
        # Do not emit a mislabeled incremental-reweight metric here.
        result['runs'].append(row)
        print(json.dumps(row), flush=True)
    os.environ[env_name]='0'
    result['baseline_profile'] = repeated_profile(baseline, 'native_topk10_exact_only')
    reference = [call(False),call(True)]
    for mode in sorted(set(map(int,args.modes.split(',')))):
        os.environ[env_name]=str(mode)
        result['profiles'][str(mode)] = repeated_profile(call,f'{args.experiment}_mode_{mode}')
        if mode:
            accepted = [f'float, {mode}>'] if args.experiment == 'summary' else ['__nv_bfloat16, true>']
            if mode == 1 and args.experiment == 'summary':
                accepted.append('float, true>')
            kernel_name = 'spark_summary_quant_kernel' if args.experiment == 'summary' else 'sol_exact_kernel'
            names = [row['name'] for row in result['profiles'][str(mode)]
                     if kernel_name in row['name']]
            if not any(tag in name for tag in accepted for name in names):
                raise RuntimeError(f'Experimental summary mode {mode} was not dispatched: {names}. '
                                   'This probe requires the archived experimental CUDA patch.')
        result['accuracy'][str(mode)] = {str(rb): metrics(reference[int(rb)],call(rb)) for rb in (False,True)}
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(args.output), accuracy=result['accuracy'])), flush=True)


if __name__ == '__main__':
    main()
