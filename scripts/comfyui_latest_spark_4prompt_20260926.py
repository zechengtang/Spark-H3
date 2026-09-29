"""Resume-safe ComfyUI Sol vs latest Spark block/query, paired 2-repeat trial."""
import argparse
import concurrent.futures as futures
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import comfyui_sol_4way_benchmark_20260923 as base
import comfyui_spark_sol_10prompt_benchmark_20260924 as source
import comfyui_spark_anchor_precision_benchmark_20260925 as old

NAME='comfyui_latest_spark_4prompt_5s10s768p_20260926'
REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/autodl-fs/data/h3_experiments')/NAME
OUT=Path('/autodl-fs/data/h3_outputs')/NAME
GPUS=(0,1,2,3)
METHODS=('sol_tau1_extra0','spark_block','spark_query')
DURATIONS=(5,10)
read=old.read
write=base.write_json


def cases(): return source.cases()[:4]
def latent(seconds,method,case,repeat=0):
    if method=='dense': return old.latent_path(seconds,'dense',case)
    return ROOT/f'{seconds}s/latents/{method}/r{repeat}/case_{case["index"]:02}.safetensors'
def record(seconds,method,case,repeat=0):
    return ROOT/f'{seconds}s/records/{method}/r{repeat}/case_{case["index"]:02}.json'
def graph(seconds,method,case,target,steps=20):
    g=source.denoise_graph('spark_topk10' if method.startswith('spark_') else method,case,target,steps)
    g['4']['inputs']['length']=120 if seconds==5 else 240
    if method.startswith('spark_'):
        g['2']['inputs'].update(ablation_mode='full',video_tail_mode='dense',
            global_anchor_dtype='float32',tail_granularity=method.removeprefix('spark_'))
    return g


def prepare():
    if (ROOT/'protocol.json').exists(): return
    if ROOT.exists() or OUT.exists(): raise FileExistsError('uninitialized experiment directory')
    identities=[(c['index'],c['sample_id'],c['prompt_sha256']) for c in cases()]
    dense_manifests={5:old.BASE5_OUT/'dense/generation_manifest.json',
                     10:old.ROOT/'manifests/10s/dense.json'}
    reused={}
    for sec in DURATIONS:
        protocol=read((old.BASE5_ROOT if sec==5 else old.ROOT)/'protocol.json')
        assert [(c['index'],c['sample_id'],c['prompt_sha256']) for c in protocol['cases'][:4]]==identities
        assert protocol['settings']['steps']==20 and protocol['settings']['seed']==42
        manifest=read(dense_manifests[sec])
        rows=[x for x in manifest['records'] if x['index']<=4]
        assert len(rows)==4
        for row,c in zip(rows,cases(),strict=True):
            assert row['sample_id']==c['sample_id']
            assert base.sha256(Path(row['output_path']))==row['sha256']
            assert latent(sec,'dense',c).is_file()
            assert source.condition_path(c).is_file()
        reused[str(sec)]={'manifest':str(dense_manifests[sec]),'records':rows,
            'latents':[{'path':str(latent(sec,'dense',c)),'sha256':base.sha256(latent(sec,'dense',c))} for c in cases()]}
    ROOT.mkdir(parents=True); OUT.mkdir(parents=True)
    for gpu in GPUS:
        for label in (f'gpu{gpu}',f'decoder_gpu{gpu}'):
            for d in ('input','temp','user'): (ROOT/d/label).mkdir(parents=True)
    sources={str(p):base.sha256(p) for p in (
        REPO/'comfyui_backend.py',REPO/'comfyui_nodes.py',REPO/'comfyui_reblock_plan.py',
        old.KITCHEN/'comfy_kitchen/backends/cuda/_C.abi3.so')}
    write(ROOT/'protocol.json',dict(name=NAME,cases=cases(),methods=METHODS,
        settings=dict(steps=20,sigma_points=21,seed=42,width=1344,height=768,fps=24,
            durations=[5,10],sampler='res_multistep',scheduler='simple',repeats=2,gpus=GPUS,
            dense_evaluations=4,dense_layers=[0],warmup='excluded 5-step run per method/duration/GPU',
            timing='SamplerCustomAdvanced; includes fresh reblock plan; excludes load/conditioning/VAE/save',
            fanout=16,reuse=1,anchor_dtype='float32',topk_ratio=.1,video_tail_mode='dense',
            sol_tau=1.,sol_extra_tokens=0,turbo_lora=False),
        reused_dense=reused,source_hashes=sources,
        conditioning=[dict(path=str(source.condition_path(c)),sha256=base.sha256(source.condition_path(c))) for c in cases()]))
    for sec in DURATIONS:
        write(ROOT/f'manifests/{sec}s/dense.json',dict(status='passed',records=reused[str(sec)]['records']))


def start(gpu,decode=False):
    label=f'decoder_gpu{gpu}' if decode else f'gpu{gpu}'
    port=(8450 if decode else 8430)+gpu
    log=(ROOT/f'server_{label}.log').open('a')
    cmd=[base.PYTHON,'main.py','--listen','127.0.0.1','--port',str(port),
        '--disable-auto-launch','--disable-cuda-malloc','--preview-method','none','--cache-classic',
        '--output-directory',str(OUT),'--temp-directory',str(ROOT/'temp'/label),
        '--input-directory',str(ROOT/'input'/label),'--user-directory',str(ROOT/'user'/label)]
    if decode: cmd+=['--highvram','--disable-async-offload','--disable-dynamic-vram']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1',
         'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4'}
    proc=subprocess.Popen(cmd,cwd=old.COMFY,env=env,stdout=log,stderr=subprocess.STDOUT)
    write(ROOT/f'pid_{label}.json',dict(pid=proc.pid,port=port))
    return label,proc,log


def run_gpu(gpu):
    c=cases()[gpu]; port=8430+gpu
    for sec in DURATIONS:
        for method in METHODS:
            target=ROOT/f'{sec}s/warmup/{method}/gpu{gpu}.safetensors'
            rp=target.with_suffix('.json')
            if not rp.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                r=base.queue_and_wait(port,graph(sec,method,c,target,5),source.SAMPLER_NODE)
                r.pop('history'); write(rp,r)
        for repeat in range(2):
            shift=(gpu+repeat)%3
            order=METHODS[shift:]+METHODS[:shift]
            for method in order:
                target=latent(sec,method,c,repeat); rp=record(sec,method,c,repeat)
                if rp.exists() and target.exists() and read(rp)['latent_sha256']==base.sha256(target): continue
                target.parent.mkdir(parents=True,exist_ok=True)
                r=base.queue_and_wait(port,graph(sec,method,c,target),source.SAMPLER_NODE)
                r.pop('history')
                if not r['sampler_seconds'] or r['sampler_seconds']<10:
                    raise RuntimeError('cached/missing sampler execution: refusing invalid timing')
                write(rp,dict(**r,method=method,seconds=sec,case=c['index'],repeat=repeat,gpu=gpu,
                    steps=20,seed=42,latent_path=str(target),latent_sha256=base.sha256(target)))
                print('DENOISED',sec,method,c['index'],repeat,r['sampler_seconds'],flush=True)
    write(ROOT/f'worker_gpu{gpu}.json',dict(status='complete'))


def denoise():
    servers=[]
    try:
        for gpu in GPUS: servers.append(start(gpu))
        for gpu in GPUS:
            base.wait_server(8430+gpu)
            obj=base.api_json(8430+gpu,'/object_info')
            assert 'tail_granularity' in obj['MiniMaxH3SparkAttentionSM120']['input']['optional']
        with futures.ThreadPoolExecutor(4) as pool: list(pool.map(run_gpu,GPUS))
    finally: base.stop_servers(servers)
    summary={}
    for sec in DURATIONS:
        methods={m:[read(record(sec,m,c,r)) for c in cases() for r in range(2)] for m in METHODS}
        summary[str(sec)]={m:dict(mean_seconds=statistics.fmean(x['sampler_seconds'] for x in rows),records=rows) for m,rows in methods.items()}
        for m in METHODS[1:]:
            ratios=[a['sampler_seconds']/b['sampler_seconds'] for a,b in zip(methods[METHODS[0]],methods[m],strict=True)]
            summary[str(sec)][m]['paired_speedups']=ratios
            summary[str(sec)][m]['mean_paired_speedup']=statistics.fmean(ratios)
    write(ROOT/'denoise_summary.json',dict(status='complete',durations=summary))


def decode_gpu(gpu):
    c=cases()[gpu]; rows=[]
    for sec in DURATIONS:
        for method in METHODS:
            target=OUT/f'{sec}s/{method}/{c["index"]:02}_{c["sample_id"]}_ultrafast.mp4'
            rp=ROOT/f'{sec}s/decode/{method}/case_{c["index"]:02}.json'
            if rp.exists() and target.exists() and read(rp)['sha256']==base.sha256(target): continue
            g=old.decode_graph(sec,'spark_fp32',c)
            g['1']['inputs']['cache_path']=str(latent(sec,method,c))
            g['10']['inputs']['output_path']=str(target)
            target.parent.mkdir(parents=True,exist_ok=True)
            base.queue_and_wait(8450+gpu,g)
            info=base.inspect_video(target)
            assert info['frames']==sec*24 and info['width']==1344 and info['height']==768
            write(rp,dict(index=c['index'],sample_id=c['sample_id'],output_path=str(target),sha256=base.sha256(target),video=info))
    return rows


def decode():
    servers=[]
    try:
        for gpu in GPUS: servers.append(start(gpu,True))
        for gpu in GPUS: base.wait_server(8450+gpu)
        with futures.ThreadPoolExecutor(4) as pool: list(pool.map(decode_gpu,GPUS))
    finally: base.stop_servers(servers)
    for sec in DURATIONS:
        for method in METHODS:
            rows=[read(ROOT/f'{sec}s/decode/{method}/case_{c["index"]:02}.json') for c in cases()]
            write(ROOT/f'manifests/{sec}s/{method}.json',dict(status='passed',records=rows))


def evaluate():
    bench=REPO.parent/'MiniMax-H3-Benchmark/scripts'
    quality={}
    for sec in DURATIONS:
        quality[str(sec)]={}
        for m in METHODS:
            label=f'{sec}s_{m}'; work=ROOT/'quality'/label
            cfg=ROOT/'quality'/f'{label}_config.json'
            write(cfg,dict(work_dir=str(work),reference_manifest=str(ROOT/f'manifests/{sec}s/dense.json'),
                candidate_manifest=str(ROOT/f'manifests/{sec}s/{m}.json'),method=label,cases=[1,2,3,4],workers=4))
            subprocess.run([base.PYTHON,str(bench/'h3_quality_video_pair.py'),'run','--config',str(cfg)],
                check=True,env={**os.environ,'H3_NUM_GPUS':'4','HF_HUB_OFFLINE':'1'})
            quality[str(sec)][m]=read(work/f'{label}_quality_results.json')['summary']
    write(ROOT/'quality_results.json',dict(status='complete',summary=quality))
    # Latent errors include video and audio tensors, not a video PSNR substitute.
    import torch
    from safetensors.torch import load_file
    from probe_comfy_summary_gemm_20260925 import metrics
    torch.set_num_threads(4)
    errors=[]
    for sec in DURATIONS:
        for c in cases():
            ref=load_file(str(latent(sec,'dense',c)))
            for m in METHODS:
                a=load_file(str(latent(sec,m,c)))
                errors.append(dict(seconds=sec,case=c['index'],method=m,
                    metrics={key:metrics(ref[key],a[key]) for key in ref if ref[key].is_floating_point()}))
    write(ROOT/'latent_metrics.json',dict(status='complete',records=errors))
    shutil.copy2(REPO/'scripts/comfyui_latest_spark_vbench_20260926.py',ROOT/'vbench.py')
    subprocess.run([base.PYTHON,'-c','import vbench; vbench.prepare()'],cwd=ROOT,check=True)
    subprocess.run([base.PYTHON,str(bench/'h3_vbench_queue.py'),'run','--experiment',str(ROOT),
                    '--gpus','0','1','2','3'],check=True,env={**os.environ,'HF_HUB_OFFLINE':'1'})


def run():
    prepare()
    # Fail explicitly if the implementation changes underneath a resumed run.
    for path,sha in read(ROOT/'protocol.json')['source_hashes'].items():
        if base.sha256(Path(path))!=sha: raise RuntimeError(f'implementation changed: {path}')
    for stage,fn in [('denoise',denoise),('decode',decode),('evaluate',evaluate)]:
        marker=ROOT/f'{stage}_complete.json'
        if marker.exists(): continue
        for attempt in range(2):
            write(ROOT/'status.json',dict(status='running',stage=stage,attempt=attempt,pid=os.getpid()))
            try: fn(); break
            except Exception:
                write(ROOT/'last_error.json',dict(stage=stage,attempt=attempt,traceback=traceback.format_exc()))
                if attempt: raise
        write(marker,dict(status='complete'))
    write(ROOT/'results.json',dict(status='complete',timing=read(ROOT/'denoise_summary.json'),
        quality=read(ROOT/'quality_results.json'),latent=read(ROOT/'latent_metrics.json'),
        vbench=read(ROOT/'vbench/results.json')))
    write(ROOT/'status.json',dict(status='complete',stage='complete',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','run','launch'])
    args=p.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='launch':
        prepare()
        with (ROOT/'orchestrator.log').open('a') as log:
            child=subprocess.Popen([base.PYTHON,__file__,'run'],stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1'})
        write(ROOT/'orchestrator_pid.json',dict(pid=child.pid)); print(child.pid)
    else:
        try: run()
        except BaseException:
            write(ROOT/'status.json',dict(status='failed',traceback=traceback.format_exc(),pid=os.getpid()))
            raise
