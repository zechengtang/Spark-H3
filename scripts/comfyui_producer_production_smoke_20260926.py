"""Exercise the production plugin without trial hooks and compare saved latents."""
import json
import os
from pathlib import Path
import subprocess
import comfyui_latest_spark_4prompt_20260926 as previous

base=previous.base
ROOT=Path('/autodl-fs/data/h3_experiments/comfyui_producer_production_smoke_20260926')
OUT=Path('/autodl-fs/data/h3_outputs/comfyui_producer_production_smoke_20260926')
REFERENCE=Path('/autodl-fs/data/h3_experiments/comfyui_producer_validate_20260926/gpu0')
PORT=8530

def same(a,b):
    import torch
    from safetensors.torch import load_file
    x,y=load_file(str(a)),load_file(str(b))
    return {name:torch.equal(x[name],y[name]) for name in x}

def run():
    os.environ['no_proxy']='127.0.0.1,localhost'
    os.environ['NO_PROXY']='127.0.0.1,localhost'
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    for d in ('input','temp','user'): (ROOT/d).mkdir(exist_ok=True)
    log=(ROOT/'server.log').open('a')
    cmd=[base.PYTHON,'main.py','--listen','127.0.0.1','--port',str(PORT),
        '--disable-auto-launch','--disable-cuda-malloc','--preview-method','none',
        '--cache-classic','--output-directory',str(OUT),
        '--input-directory',str(ROOT/'input'),'--temp-directory',str(ROOT/'temp'),
        '--user-directory',str(ROOT/'user')]
    proc=subprocess.Popen(cmd,cwd=previous.old.COMFY,stdout=log,stderr=subprocess.STDOUT,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1',
             'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4'})
    base.write_json(ROOT/'pid.json',dict(pid=proc.pid,port=PORT))
    try:
        base.wait_server(PORT)
        case=previous.cases()[0]
        result=[]
        for sec,grain in ((5,'query'),(10,'block')):
            target=ROOT/f'{sec}s_{grain}.safetensors'
            base.queue_and_wait(PORT,previous.graph(sec,f'spark_{grain}',case,ROOT/f'warm_{sec}s_{grain}.safetensors',5),'11')
            timing=base.queue_and_wait(PORT,previous.graph(sec,f'spark_{grain}',case,target),'11')
            if not timing.get('sampler_seconds') or timing['sampler_seconds']<10:
                raise RuntimeError('sampler not executed')
            ref=REFERENCE/f'{sec}s/candidate_{grain}_0.safetensors'
            equal=same(ref,target)
            if not all(equal.values()):raise AssertionError(f'Production latent mismatch: {sec}s {grain} {equal}')
            row=dict(seconds=sec,grain=grain,sampler_seconds=timing['sampler_seconds'],
                reference_path=str(ref),latent_path=str(target),sha256=base.sha256(target),bitwise_equal=equal)
            result.append(row);base.write_json(ROOT/f'{sec}s_{grain}.json',row)
            print('PRODUCTION',sec,grain,row['sampler_seconds'],equal,flush=True)
        base.write_json(ROOT/'results.json',dict(status='complete',records=result,
            production_source_hashes={str(p):base.sha256(p) for p in (
                previous.REPO/'comfyui_nodes.py',previous.REPO/'comfyui_backend.py')}))
    finally:base.stop_servers([('gpu0',proc,log)])

if __name__=='__main__':run()
