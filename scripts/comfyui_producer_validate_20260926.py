"""Automatically validate the fastest finite candidate on the original 4 prompts.

Does not modify production defaults. Saves latent differences and paired real
sampler times; candidate selection is preliminary until this validation ends.
"""
import concurrent.futures as futures
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback
import comfyui_producer_screen_20260926 as screen

previous=screen.previous;base=screen.base
ROOT=Path('/autodl-fs/data/h3_experiments/comfyui_producer_validate_20260926')
LABELS=('sol','original_block','candidate_block','original_query','candidate_query')

def worker(gpu,chunk):
    port=8510+gpu;root=ROOT/f'gpu{gpu}'
    for folder in ('input','temp','user','output'):(root/folder).mkdir(parents=True,exist_ok=True)
    log=(root/'server.log').open('a')
    cmd=[base.PYTHON,'main.py','--listen','127.0.0.1','--port',str(port),
        '--disable-auto-launch','--disable-cuda-malloc','--preview-method','none','--cache-classic']
    for folder in ('input','temp','user','output'):cmd.extend([f'--{folder}-directory',str(root/folder)])
    proc=subprocess.Popen(cmd,cwd=previous.old.COMFY,stdout=log,stderr=subprocess.STDOUT,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'H3_PRODUCER_TRIAL':'1',
             'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4'})
    base.write_json(root/'pid.json',dict(pid=proc.pid,port=port))
    try:
        base.wait_server(port);case=previous.cases()[gpu]
        for seconds in (5,10):
            dest=root/f'{seconds}s';dest.mkdir(parents=True,exist_ok=True)
            for repeat in range(2):
                shift=(gpu+repeat)%len(LABELS);order=LABELS[shift:]+LABELS[:shift]
                for label in order:
                    target=dest/f'{label}_{repeat}.safetensors';record=target.with_suffix('.json')
                    if record.exists() and target.exists():continue
                    screen.configure(port,dict(enabled=label.startswith('candidate'),chunk=chunk,direct=True,views=True))
                    method='sol_tau1_extra0' if label=='sol' else 'spark_'+label.split('_')[-1]
                    base.queue_and_wait(port,previous.graph(seconds,method,case,dest/f'warm_{label}_{repeat}.safetensors',5),'11')
                    before=base.api_json(port,'/h3_trial/stats')['evaluations']
                    result=base.queue_and_wait(port,previous.graph(seconds,method,case,target),'11')
                    after=base.api_json(port,'/h3_trial/stats')['evaluations'];result.pop('history')
                    if after-before!=20 or not result.get('sampler_seconds') or result['sampler_seconds']<10:
                        raise RuntimeError(f'Incomplete/cached sampler: {after-before}')
                    base.write_json(record,dict(**result,case=case['index'],gpu=gpu,seconds=seconds,label=label,
                        repeat=repeat,chunk=chunk,actual_evaluations=after-before,latent_sha256=base.sha256(target)))
                    print('VALIDATE',gpu,seconds,label,repeat,result['sampler_seconds'],flush=True)
            errors={grain:screen.compare(dest/f'original_{grain}_0.safetensors',dest/f'candidate_{grain}_0.safetensors')
                    for grain in ('block','query')}
            base.write_json(dest/'latent_comparison.json',errors)
    finally:base.stop_servers([(str(gpu),proc,log)])

def run():
    import torch
    torch.set_num_threads(2);ROOT.mkdir(parents=True,exist_ok=True)
    base.write_json(ROOT/'status.json',dict(status='waiting_for_screen',pid=os.getpid()))
    while True:
        status=previous.read(screen.ROOT/'status.json') if (screen.ROOT/'status.json').exists() else {}
        if status.get('status')=='complete':break
        if status.get('status')=='failed':raise RuntimeError('Screen failed; validation not started')
        time.sleep(10)
    candidates=previous.read(screen.ROOT/'results.json')['records']
    candidates=[c for c in candidates if all(m['finite'] for m in c['candidate_vs_original'].values())]
    if not candidates:raise RuntimeError('No finite candidate; refusing validation')
    chosen=max(candidates,key=lambda c:statistics.fmean(c['timings']['original'])/statistics.fmean(c['timings']['candidate']))
    chunk=chosen['chunk']
    base.write_json(ROOT/'protocol.json',dict(chunk=chunk,selection=chosen,
        source_protocol=previous.read(screen.ROOT/'protocol.json'),cases=previous.cases(),
        settings=previous.read(previous.ROOT/'protocol.json')['settings'],
        note='Candidate speed screen, then four-prompt validation. No automatic production promotion.'))
    for path,sha in previous.read(screen.ROOT/'protocol.json')['source_hashes'].items():
        if base.sha256(Path(path))!=sha:raise RuntimeError(f'Source changed: {path}')
    base.write_json(ROOT/'status.json',dict(status='running',pid=os.getpid(),chunk=chunk))
    with futures.ThreadPoolExecutor(4) as pool:list(pool.map(lambda gpu:worker(gpu,chunk),range(4)))
    summaries={}
    for seconds in (5,10):
        times={label:[previous.read(ROOT/f'gpu{g}/{seconds}s/{label}_{r}.json')['sampler_seconds']
                      for g in range(4) for r in range(2)] for label in LABELS}
        means={label:statistics.fmean(values) for label,values in times.items()}
        summaries[str(seconds)]=dict(means=means,times=times,
            speedup_vs_sol={label:means['sol']/value for label,value in means.items() if label!='sol'},
            latent_comparisons=[previous.read(ROOT/f'gpu{g}/{seconds}s/latent_comparison.json') for g in range(4)])
    base.write_json(ROOT/'results.json',dict(status='validation_complete',chunk=chunk,durations=summaries,
        production_default_changed=False,note='Latent errors are not decoded-video PSNR/VBench.'))
    base.write_json(ROOT/'status.json',dict(status='complete'))

if __name__=='__main__':
    try:run()
    except Exception:
        base.write_json(ROOT/'status.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
