"""Unattended, isolated producer candidate screen after the baseline audit.

Every candidate has same-GPU Sol and original-Spark controls. No production
default is promoted by this screen; numerical checks and paired times are saved.
"""
import concurrent.futures as futures
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import urllib.request
import comfyui_latest_spark_4prompt_20260926 as previous

base=previous.base
ROOT=Path('/autodl-fs/data/h3_experiments/comfyui_producer_screen_20260926')
AUDIT=Path('/autodl-fs/data/h3_experiments/comfyui_pipeline_audit_v2_20260926')

def configure(port,options):
    request=urllib.request.Request(f'http://127.0.0.1:{port}/h3_trial/configure',
        data=json.dumps(options).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(request) as response:return json.load(response)

def compare(a,b):
    import torch
    from safetensors.torch import load_file
    left,right=load_file(str(a)),load_file(str(b))
    assert left.keys()==right.keys()
    result={}
    for name,x in left.items():
        y=right[name]
        if not x.is_floating_point():continue
        delta=x.float()-y.float()
        result[name]=dict(bitwise_equal=torch.equal(x,y),finite=bool(torch.isfinite(y).all()),
            rmse=float(delta.square().mean().sqrt()),max_abs=float(delta.abs().max()),
            relative_l2=float(delta.norm()/x.float().norm().clamp_min(1e-12)))
    return result

def worker(gpu):
    chunk=(4096,8192,16384,32768)[gpu];port=8490+gpu;root=ROOT/f'gpu{gpu}'
    for folder in ('input','temp','user','output'): (root/folder).mkdir(parents=True,exist_ok=True)
    log=(root/'server.log').open('a')
    cmd=[base.PYTHON,'main.py','--listen','127.0.0.1','--port',str(port),
        '--disable-auto-launch','--disable-cuda-malloc','--preview-method','none','--cache-classic']
    for folder in ('input','temp','user','output'):cmd.extend([f'--{folder}-directory',str(root/folder)])
    proc=subprocess.Popen(cmd,cwd=previous.old.COMFY,stdout=log,stderr=subprocess.STDOUT,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'H3_PRODUCER_TRIAL':'1',
             'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4'})
    base.write_json(root/'pid.json',dict(pid=proc.pid,port=port))
    try:
        base.wait_server(port)
        case=previous.cases()[0]
        order=['sol','original','candidate']
        order=order[gpu%3:]+order[:gpu%3]
        for repeat in range(2):
            for label in (order if repeat==0 else list(reversed(order))):
                target=root/f'{label}_{repeat}.safetensors';record=target.with_suffix('.json')
                if record.exists() and target.exists():continue
                configure(port,dict(enabled=label=='candidate',chunk=chunk,direct=True,views=True))
                method='sol_tau1_extra0' if label=='sol' else 'spark_query'
                base.queue_and_wait(port,previous.graph(10,method,case,root/f'warm_{label}_{repeat}.safetensors',5),'11')
                before=base.api_json(port,'/h3_trial/stats')['evaluations']
                result=base.queue_and_wait(port,previous.graph(10,method,case,target),'11')
                after=base.api_json(port,'/h3_trial/stats')['evaluations']
                result.pop('history')
                if after-before!=20 or not result.get('sampler_seconds') or result['sampler_seconds']<10:
                    raise RuntimeError(f'Incomplete/cached sampler: {after-before} evaluations')
                base.write_json(record,dict(**result,actual_evaluations=after-before,chunk=chunk,
                    gpu=gpu,repeat=repeat,label=label,latent_sha256=base.sha256(target)))
                print('SCREEN',gpu,chunk,label,repeat,result['sampler_seconds'],flush=True)
        records={label:[previous.read(root/f'{label}_{r}.json')['sampler_seconds'] for r in range(2)]
                 for label in ('sol','original','candidate')}
        errors=compare(root/'original_0.safetensors',root/'candidate_0.safetensors')
        noise=compare(root/'original_0.safetensors',root/'original_1.safetensors')
        base.write_json(root/'result.json',dict(chunk=chunk,timings=records,
            candidate_vs_original=errors,original_repeat_variation=noise,
            note='One prompt, 10s, query tail; candidate screen only, not four-prompt validation or video quality.'))
    finally:base.stop_servers([(str(gpu),proc,log)])

def run():
    import torch
    torch.set_num_threads(2)
    ROOT.mkdir(parents=True,exist_ok=True)
    base.write_json(ROOT/'status.json',dict(status='waiting_for_audit',pid=os.getpid()))
    while True:
        status=previous.read(AUDIT/'status.json') if (AUDIT/'status.json').exists() else {}
        if status.get('status')=='complete':break
        if status.get('status')=='failed':raise RuntimeError('Audit failed; not starting candidate GPU jobs')
        time.sleep(10)
    sources=[previous.REPO/'comfyui_nodes.py',previous.REPO/'comfyui_backend.py',
        previous.REPO/'scripts/comfyui_producer_trial.py',previous.old.COMFY/'custom_nodes/h3_producer_trial.py',
        previous.old.KITCHEN/'comfy_kitchen/backends/cuda/_C.abi3.so']
    hashes={str(p):base.sha256(p) for p in sources}
    protocol=ROOT/'protocol.json'
    if protocol.exists() and previous.read(protocol)['source_hashes']!=hashes:
        raise RuntimeError('Candidate source changed during resume')
    base.write_json(protocol,dict(source_hashes=hashes,case=previous.cases()[0],
        settings=previous.read(previous.ROOT/'protocol.json')['settings'],
        screening=dict(seconds=10,tail='query',chunk_by_gpu=[4096,8192,16384,32768],repeats=2)))
    base.write_json(ROOT/'status.json',dict(status='running',pid=os.getpid()))
    with futures.ThreadPoolExecutor(4) as pool:list(pool.map(worker,range(4)))
    results=[previous.read(ROOT/f'gpu{g}/result.json') for g in range(4)]
    base.write_json(ROOT/'results.json',dict(status='screen_complete',records=results))
    base.write_json(ROOT/'status.json',dict(status='complete'))

if __name__=='__main__':
    try:run()
    except Exception:
        base.write_json(ROOT/'status.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
