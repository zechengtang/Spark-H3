"""Break down the 5s Spark plan in one genuine 20 evaluation ComfyUI run."""
import os
from pathlib import Path
import subprocess
import comfyui_latest_spark_4prompt_20260926 as previous

base=previous.base
NAME=os.environ.get('H3_BREAKDOWN_NAME','comfyui_5s_plan_breakdown_20260926')
ROOT=Path('/autodl-fs/data/h3_experiments')/NAME
OUT=Path('/autodl-fs/data/h3_outputs')/NAME
PORT=int(os.environ.get('H3_BREAKDOWN_PORT','8541'))
GPU=os.environ.get('H3_BREAKDOWN_GPU','1')

def run():
    os.environ['no_proxy']='127.0.0.1,localhost'
    os.environ['NO_PROXY']='127.0.0.1,localhost'
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    for name in ('input','temp','user'):(ROOT/name).mkdir(exist_ok=True)
    log=(ROOT/'server.log').open('a')
    cmd=[base.PYTHON,'main.py','--listen','127.0.0.1','--port',str(PORT),
        '--disable-auto-launch','--disable-cuda-malloc','--preview-method','none','--cache-classic',
        '--output-directory',str(OUT),'--input-directory',str(ROOT/'input'),
        '--temp-directory',str(ROOT/'temp'),'--user-directory',str(ROOT/'user')]
    proc=subprocess.Popen(cmd,cwd=previous.old.COMFY,stdout=log,stderr=subprocess.STDOUT,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'H3_PIPELINE_AUDIT':'1',
             'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4'})
    base.write_json(ROOT/'pid.json',dict(pid=proc.pid,port=PORT))
    try:
        base.wait_server(PORT);case=previous.cases()[0]
        base.api_json(PORT,'/h3_audit/reset?enabled=0')
        base.queue_and_wait(PORT,previous.graph(5,'spark_block',case,ROOT/'warmup.safetensors',5),'11')
        base.api_json(PORT,'/h3_audit/reset?enabled=1')
        timing=base.queue_and_wait(PORT,previous.graph(5,'spark_block',case,ROOT/'profiled.safetensors'),'11')
        timing.pop('history')
        ranges=base.api_json(PORT,'/h3_audit/flush')
        if ranges['evaluations']!=20:raise RuntimeError('Not a full 20 evaluation run')
        base.write_json(ROOT/'timing.json',timing)
        base.write_json(ROOT/'ranges.json',ranges)
        from collections import defaultdict
        grouped=defaultdict(list)
        for row in ranges['ranges']:
            if row['name']=='reblock_plan' or row['name'].startswith('plan_'):
                grouped[row['name']].append(row)
        summary={k:dict(calls=len(v),mean_ms=sum(x['gpu_ms'] for x in v)/len(v),
            total_ms=sum(x['gpu_ms'] for x in v)) for k,v in grouped.items()}
        base.write_json(ROOT/'results.json',dict(status='complete',sampler_seconds=timing['sampler_seconds'],
            stages=summary,source_hashes={str(p):base.sha256(p) for p in (
                previous.REPO/'comfyui_nodes.py',previous.REPO/'comfyui_reblock_plan.py',
                previous.old.COMFY/'custom_nodes/h3_pipeline_audit.py')}))
        print(summary,flush=True)
    finally:base.stop_servers([(f'gpu{GPU}',proc,log)])

if __name__=='__main__':run()
