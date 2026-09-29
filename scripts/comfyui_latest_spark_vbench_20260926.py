"""Assigned core-five scoring adapter; copied into the experiment directory."""
import json
from pathlib import Path
import statistics
import sys

EXP=Path(__file__).resolve().parent
BENCH=Path('/autodl-fs/data/h3_repos/MiniMax-H3-Benchmark')
sys.path.insert(0,str(BENCH/'scripts'))
import score_vbench20pct_768p10s as base
read,write=base.read,base.write
DIMS=('subject_consistency','background_consistency','motion_smoothness','imaging_quality','aesthetic_quality')
METHODS=('dense','sol_tau1_extra0','spark_block','spark_query')
ARMS=tuple(f'{sec}s_{m}' for sec in (5,10) for m in METHODS)


def prepare():
    assigned={c['index']:c['vbench_dimensions'] for c in read(EXP/'protocol.json')['cases']}
    videos={d:[] for d in DIMS}
    for arm in ARMS:
        sec,m=arm.split('s_',1)
        for row in read(EXP/f'manifests/{sec}s/{m}.json')['records']:
            for dim in assigned[row['index']]:
                videos[dim].append(dict(method=arm,case=row['index'],video_path=row['output_path'],video_sha256=row['sha256']))
    cache={}
    for name in ('comfyui_spark_sol_10prompt_5s768p_20260924',
                 'comfyui_spark_anchor_precision_10prompt_5s10s768p_20260925'):
        path=Path('/autodl-fs/data/h3_experiments')/name/'vbench/results.json'
        for dim,payload in read(path)['dimensions'].items():
            for row in payload['records']:
                cache[f"{dim}:{row['video_sha256']}"]=dict(score=row['score'],native_score=row['native_score'],source=str(path))
    write(EXP/'vbench/protocol.json',dict(status='prepared',dimensions=DIMS,cache=cache,
        videos_by_dimension=videos,evaluation='Source-assigned core-five only; four prompts; not official total'))


def aggregate():
    protocol=read(EXP/'vbench/protocol.json'); dims={}; scores={a:{} for a in ARMS}
    for dim in DIMS:
        result=read(EXP/f'vbench/scores/{dim}.json')
        expected={(x['method'],x['case']) for x in protocol['videos_by_dimension'][dim]}
        assert result['status']=='complete' and {(x['method'],x['case']) for x in result['records']}==expected
        dims[dim]=result
        for arm in ARMS:
            scores[arm][dim]=100*statistics.fmean(x['score'] for x in result['records'] if x['method']==arm)
    write(EXP/'vbench/results.json',dict(status='complete',dimensions=dims,scores_percent_assigned_prompts=scores,official_total=False))
