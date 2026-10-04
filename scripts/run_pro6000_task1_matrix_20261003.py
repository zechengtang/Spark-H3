#!/usr/bin/env python3
"""Four-card PRO6000 task-1 matrix plus explicit dynamic-cache A/B."""
from __future__ import annotations
import contextlib, dataclasses, json, os, statistics, subprocess, sys, time
from pathlib import Path
import pytorch_spark_tail_ablation_25prompt_5s768p_20260924 as base

NAME="pro6000_task1_matrix_20261003"
ROOT=Path("/autodl-fs/data/h3_experiments")/NAME
OUT=Path("/autodl-fs/data/h3_outputs")/NAME
SAMPLES=base.BENCH/"vbench_core5_percent_subsets/20pct/samples.json"
SOURCE=Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")
CASES=(2,13,31,33); GPUS=(0,1,2,3)
DURATIONS={"5s":120,"10s":240,"14p4s":345}
ARMS=("legacy_threshold","fused_threshold","fused_packed_external","fused_packed_external_no_route_qk")

def cases():
    return base.pipeline().load_cases(SAMPLES,list(CASES),expected_indices=tuple(range(1,51)))

def args():
    return base.pipeline().build_parser().parse_args([
        "denoise","--samples",str(SAMPLES),"--output",str(OUT),"--method","dense",
        "--case-indices",*map(str,CASES),"--steps","20","--frames","240",
        "--height","768","--width","1344","--workers","1"])

def config(arm,steps=20):
    from h3_sparse_attention import H3SparseAttentionConfig
    kw=dict(warmup_percent=20.0 if steps==20 else 33.0,sol_dense_layers=1,
            sol_route_topk_ratio=.1,sol_log_density=False)
    if arm!="legacy_threshold": kw["landmark_tree_v2_midpoint_direction_mode"]="fused"
    kw["sol_route_topk_execution"]={
        "legacy_threshold":"threshold","fused_threshold":"threshold",
        "fused_packed_external":"packed_external",
        "fused_packed_external_no_route_qk":"packed_external_no_route_qk"}[arm]
    return H3SparseAttentionConfig.spark(steps,**kw)

def write(path,value): base.write(path,value)

def prepare():
    if ROOT.exists() or OUT.exists(): raise FileExistsError(f"refusing overwrite {ROOT} or {OUT}")
    (ROOT/"records").mkdir(parents=True); OUT.mkdir(parents=True)
    for name in ("conditioning_cache","conditioning_manifest.json"):
        (OUT/name).symlink_to(SOURCE/name,target_is_directory=name.endswith("cache"))
    selected=cases(); base.pipeline().configure_denoise_workflow(args(),selected)
    write(ROOT/"protocol.json",dict(status="running",name=NAME,cases=selected,gpus=GPUS,
        durations=DURATIONS,arms=ARMS,steps=20,actual_evaluations=19,seed=42,
        warmup="discarded 3 requested steps: exactly one dense plus one sparse eval",
        cache_ab="10s prompt A compiled then prompt B reuse vs forced _FUSED_COMPILED clear; order balanced",
        configs={a:dataclasses.asdict(config(a)) for a in ARMS},
        revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=base.IMPL,text=True).strip(),
        runner_sha256=base.sha(Path(__file__).resolve())))

def run(pipe,p,state,frames,steps,arm=None):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention
    from h3_sparse_attention.sol_numerator_virtual_q import fused_compile_cache_stats
    value=p.clone_state(state); value.values["prompt_embeds"]=value.values["prompt_embeds"].cuda()
    ctx=contextlib.nullcontext(None) if arm is None else install_h3_sparse_attention(pipe.transformer,config(arm,steps))
    before=fused_compile_cache_stats()
    with ctx as plugin,torch.inference_mode():
        if plugin: plugin.reset()
        torch.cuda.synchronize(); start=time.perf_counter()
        out=pipe(state=value,num_frames=frames,height=768,width=1344,num_inference_steps=steps,
                 generator=torch.Generator(device="cpu").manual_seed(42),output=["latents","audio_latents"])
        torch.cuda.synchronize(); seconds=time.perf_counter()-start
        summary=None if plugin is None else plugin.summary()
    assert all(torch.isfinite(out[k]).all().item() for k in ("latents","audio_latents"))
    if summary:
        assert summary["completed_evaluations"]==steps-1,summary
        assert summary["dense_evaluations"]==(4 if steps==20 else 1),summary
    after=fused_compile_cache_stats(); del out,value
    return dict(seconds=seconds,attention_summary=summary,compile_before=before,compile_after=after,
                new_compile_calls=after["compile_calls"]-before["compile_calls"],
                new_compile_seconds=after["compile_seconds"]-before["compile_seconds"])

def worker(rank):
    import torch
    import h3_sparse_attention.sol_numerator_virtual_q as fused
    rank=int(rank); torch.set_num_threads(4); p=base.pipeline(); selected=cases()
    workflow,states=p.configure_denoise_workflow(args(),selected)
    pipe,manager,acceleration,placement=p.load_denoiser(args(),workflow)
    assigned=rank; target=1 if rank==0 else rank
    try:
        # Explicit cache reuse versus forced miss.  Both run prompt B/assigned prompt.
        fused._FUSED_COMPILED.clear(); fused._FUSED_COMPILE_CALLS=0; fused._FUSED_COMPILE_SECONDS=0.0
        warm=run(pipe,p,states[0],240,3,"legacy_threshold")
        order=("reuse","cold") if rank%2==0 else ("cold","reuse")
        for mode in order:
            if mode=="cold": fused._FUSED_COMPILED.clear()
            row=run(pipe,p,states[target],240,20,"legacy_threshold")
            write(ROOT/"records"/f"gpu{rank}_cache_{mode}.json",dict(kind="cache_ab",gpu=rank,
                  mode=mode,case=selected[target]["index"],placement=placement,**row))
        write(ROOT/"records"/f"gpu{rank}_cache_warmup.json",dict(kind="warmup",gpu=rank,**warm))

        # Main Dense/Spark matrix; one independent prompt per GPU.
        for label,frames in DURATIONS.items():
            warm=run(pipe,p,states[assigned],frames,3,"legacy_threshold")
            write(ROOT/"records"/f"gpu{rank}_{label}_baseline_warmup.json",dict(kind="warmup",**warm))
            for method,arm in (("dense",None),("spark_10pct","legacy_threshold")):
                row=run(pipe,p,states[assigned],frames,20,arm)
                write(ROOT/"records"/f"gpu{rank}_{label}_{method}.json",dict(kind="matrix",gpu=rank,
                    case=selected[assigned]["index"],duration=label,frames=frames,method=method,**row))

        # 10s independent kernel ablation; baseline arm above is reused.
        for arm in ARMS[1:]:
            warm=run(pipe,p,states[assigned],240,3,arm)
            row=run(pipe,p,states[assigned],240,20,arm)
            write(ROOT/"records"/f"gpu{rank}_10s_{arm}.json",dict(kind="ablation",gpu=rank,
                case=selected[assigned]["index"],arm=arm,**row))
            write(ROOT/"records"/f"gpu{rank}_10s_{arm}_warmup.json",dict(kind="warmup",**warm))
        write(ROOT/f"worker_gpu{rank}.json",{"status":"complete"})
    finally:
        acceleration.remove(); del pipe,manager; p.release_cpu_arenas()

def summarize():
    rows=[json.loads(x.read_text()) for x in (ROOT/"records").glob("*.json")]
    result={"status":"complete","matrix":{},"ablation":{},"cache_ab":{}}
    for label in DURATIONS:
        result["matrix"][label]={}
        for method in ("dense","spark_10pct"):
            vals=[r["seconds"] for r in rows if r.get("kind")=="matrix" and r["duration"]==label and r["method"]==method]
            result["matrix"][label][method]={"values":vals,"mean_seconds":statistics.mean(vals)}
        d=result["matrix"][label]; d["speedup_x"]=d["dense"]["mean_seconds"]/d["spark_10pct"]["mean_seconds"]
    for arm in ARMS:
        vals=([r["seconds"] for r in rows if r.get("kind")=="matrix" and r["duration"]=="10s" and r["method"]=="spark_10pct"]
              if arm==ARMS[0] else [r["seconds"] for r in rows if r.get("kind")=="ablation" and r["arm"]==arm])
        result["ablation"][arm]={"values":vals,"mean_seconds":statistics.mean(vals)}
    for mode in ("reuse","cold"):
        selected=[r for r in rows if r.get("kind")=="cache_ab" and r["mode"]==mode]
        result["cache_ab"][mode]={"seconds":[r["seconds"] for r in selected],
            "mean_seconds":statistics.mean(r["seconds"] for r in selected),
            "compile_seconds":[r["new_compile_seconds"] for r in selected],
            "compile_calls":[r["new_compile_calls"] for r in selected]}
    result["cache_ab"]["saving_seconds"]=result["cache_ab"]["cold"]["mean_seconds"]-result["cache_ab"]["reuse"]["mean_seconds"]
    write(ROOT/"results.json",result); print(json.dumps(result,indent=2),flush=True)

def launch():
    prepare(); jobs=[]
    for rank,gpu in enumerate(GPUS):
        log=(ROOT/f"gpu{gpu}.log").open("a")
        proc=subprocess.Popen([str(base.PYTHON),str(Path(__file__).resolve()),"worker",str(rank)],
             env={**os.environ,**base.ENV,"CUDA_VISIBLE_DEVICES":str(gpu)},stdout=log,stderr=subprocess.STDOUT)
        jobs.append((proc,log))
    codes=[p.wait() for p,_ in jobs]
    for _,log in jobs: log.close()
    if any(codes): raise RuntimeError(codes)
    summarize()

if __name__=="__main__":
    cmd=sys.argv[1] if len(sys.argv)>1 else "run"
    if cmd=="run": launch()
    elif cmd=="worker": worker(sys.argv[2])
    elif cmd=="summarize": summarize()
    else: raise ValueError(cmd)
