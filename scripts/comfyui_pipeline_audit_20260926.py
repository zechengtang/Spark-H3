"""Paired real-pipeline nested timing audit; four GPUs, existing four prompts."""
import concurrent.futures as futures
import json
import os
from pathlib import Path
import subprocess
import traceback
import comfyui_latest_spark_4prompt_20260926 as previous

base=previous.base
ROOT=Path('/autodl-fs/data/h3_experiments/comfyui_pipeline_audit_v2_20260926')

def worker(gpu):
    port=8470+gpu
    root=ROOT/f'gpu{gpu}'
    for folder in ('input','temp','user','output'): (root/folder).mkdir(parents=True,exist_ok=True)
    log=(root/'server.log').open('a')
    cmd=[base.PYTHON,'main.py','--listen','127.0.0.1','--port',str(port),
         '--disable-auto-launch','--disable-cuda-malloc','--preview-method','none','--cache-classic']
    for folder in ('input','temp','user','output'):cmd.extend([f'--{folder}-directory',str(root/folder)])
    proc=subprocess.Popen(cmd,cwd=previous.old.COMFY,stdout=log,stderr=subprocess.STDOUT,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'H3_PIPELINE_AUDIT':'1',
             'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4'})
    base.write_json(root/'pid.json',dict(pid=proc.pid,port=port))
    try:
        base.wait_server(port)
        case=previous.cases()[gpu]
        for seconds in ((10,5) if gpu<2 else (5,10)):
            methods=previous.METHODS[gpu%3:]+previous.METHODS[:gpu%3]
            for method in methods:
                dest=root/f'{seconds}s/{method}';dest.mkdir(parents=True,exist_ok=True)
                if (dest/'complete.json').exists():continue
                base.api_json(port,'/h3_audit/reset?enabled=0')
                base.queue_and_wait(port,previous.graph(seconds,method,case,dest/'warmup.safetensors',5),'11')
                for measured in ((False,True) if gpu%2==0 else (True,False)):
                    label='profiled' if measured else 'control'
                    if (dest/f'{label}_timing.json').exists() and (not measured or (dest/'ranges.json').exists()):continue
                    # Changing only the downstream save path does not invalidate
                    # ComfyUI's sampler cache. A different-step run does, without
                    # unloading the model or changing the formal seed/settings.
                    base.api_json(port,'/h3_audit/reset?enabled=0')
                    base.queue_and_wait(port,previous.graph(seconds,method,case,dest/f'pre_{label}.safetensors',5),'11')
                    base.api_json(port,f'/h3_audit/reset?enabled={int(measured)}')
                    result=base.queue_and_wait(port,previous.graph(seconds,method,case,dest/f'{label}.safetensors'),'11')
                    result.pop('history')
                    if not result.get('sampler_seconds') or result['sampler_seconds']<10:
                        raise RuntimeError('missing execution or cached sampler')
                    base.write_json(dest/f'{label}_timing.json',dict(**result,gpu=gpu,case=case['index'],method=method,seconds=seconds))
                    if measured:
                        audit=base.api_json(port,'/h3_audit/flush')
                        if audit['evaluations']!=20:raise RuntimeError(f"wrong evaluation count: {audit['evaluations']}")
                        base.write_json(dest/'ranges.json',audit)
                    print('DONE',gpu,seconds,method,label,result['sampler_seconds'],flush=True)
                base.write_json(dest/'complete.json',dict(status='complete'))
    finally:
        base.stop_servers([(str(gpu),proc,log)])

def resilient_worker(gpu):
    for attempt in range(2):
        try:return worker(gpu)
        except Exception:
            base.write_json(ROOT/f'gpu{gpu}/error_{attempt}.json',dict(traceback=traceback.format_exc()))
            if attempt:raise

def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    paths=[previous.REPO/'comfyui_nodes.py',previous.REPO/'comfyui_backend.py',
           previous.REPO/'comfyui_reblock_plan.py',previous.old.KITCHEN/'comfy_kitchen/backends/cuda/_C.abi3.so',
           previous.old.COMFY/'custom_nodes/h3_pipeline_audit.py']
    protocol=ROOT/'protocol.json'
    hashes={str(p):base.sha256(p) for p in paths}
    if protocol.exists():
        if json.loads(protocol.read_text())['source_hashes']!=hashes:raise RuntimeError('source changed; use a new audit directory')
    else:base.write_json(protocol,dict(source_hashes=hashes,cases=previous.cases(),
        settings=previous.read(previous.ROOT/'protocol.json')['settings'],
        profiling='Nested CUDA events with paired uninstrumented execution; not isolated kernel sums'))
    base.write_json(previous.ROOT/'status.json',dict(status='deferred',stage='decode',
        reason='GPU allocation to real pipeline audit; all generation artifacts preserved'))
    base.write_json(ROOT/'status.json',dict(status='running',pid=os.getpid()))
    try:
        with futures.ThreadPoolExecutor(4) as pool:list(pool.map(resilient_worker,range(4)))
    except Exception:
        base.write_json(ROOT/'status.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
    base.write_json(ROOT/'status.json',dict(status='complete'))

if __name__=='__main__':run()
