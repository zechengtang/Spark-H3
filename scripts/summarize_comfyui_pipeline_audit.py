"""Summarize nested real-execution ranges without adding overlapping timers."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

def summarize(path):
    payload=json.loads(path.read_text())
    rows=payload['ranges'];groups=defaultdict(list)
    for row in rows:groups[row['name']].append(row)
    roots=[r for r in rows if r['parent'] is None]
    exclusive=sum(r['exclusive_ms'] for r in rows)
    inclusive=sum(r['gpu_ms'] for r in roots)
    # This checks bookkeeping, not whether CUDA gaps are active kernel work.
    assert abs(exclusive-inclusive)<max(.1,inclusive*1e-6)
    timing=json.loads((path.parent/'profiled_timing.json').read_text())
    control_path=path.parent/'control_timing.json'
    control=json.loads(control_path.read_text()) if control_path.exists() else None
    stages={name:dict(calls=len(g),inclusive_total_ms=sum(r['gpu_ms'] for r in g),
        exclusive_total_ms=sum(r['exclusive_ms'] for r in g),mean_ms=statistics.fmean(r['gpu_ms'] for r in g),
        min_exclusive_ms=min(r['exclusive_ms'] for r in g)) for name,g in groups.items()}
    steps=[]
    for step in range(payload['evaluations']):
        selected=[r for r in rows if r['step']==step]
        d=defaultdict(float)
        for r in selected:d[r['name']]+=r['exclusive_ms']
        steps.append(dict(step=step,exclusive_ms=dict(d)))
    return dict(path=str(path),method=timing['method'],seconds=timing['seconds'],gpu=timing['gpu'],
        case=timing['case'],profiled_sampler_s=timing['sampler_seconds'],
        control_sampler_s=control['sampler_seconds'] if control else None,
        profiling_overhead_fraction=timing['sampler_seconds']/control['sampler_seconds']-1 if control else None,
        model_ms=inclusive,exclusive_sum_ms=exclusive,
        sampler_outside_model_ms=timing['sampler_seconds']*1000-inclusive,stages=stages,steps=steps)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    result=[summarize(path) for path in sorted(a.root.glob('gpu*/*s/*/ranges.json'))]
    print(json.dumps(dict(records=result),indent=2))
