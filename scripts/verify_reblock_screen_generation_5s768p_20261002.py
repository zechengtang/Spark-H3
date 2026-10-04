#!/usr/bin/env python3
"""Three-prompt full-generation follow-up for the best local-swap layout.

This deliberately unfused adapter measures whether the QKV screen transfers
to generated RGB. Its added compute is reference-code overhead, not a speed
claim for an optimized packing kernel. The production package stays untouched.
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reblock_screen",REPO/"scripts/ablate_reblock_approx_5s768p_20261002.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)
ROOT = screen.ROOT/"generation_check"
CASES = (1,15,44)


def prepare():
    result = json.loads((screen.ROOT/"results.json").read_text())
    baseline = result["layout_means"]["baseline"]["output_relative_l2"]
    candidates = ("swap_q4","swap_k4","swap_both4")
    best = min(candidates,key=lambda n:result["layout_means"][n]["output_relative_l2"])
    if result["layout_means"][best]["output_relative_l2"] >= baseline:
        raise RuntimeError("No local-swap candidate improves the screen; no generation follow-up warranted")
    ROOT.mkdir(exist_ok=True)
    for gpu in range(2):
        path = ROOT/f"workflow_{gpu}"
        path.mkdir(exist_ok=True)
        for name in ("conditioning_cache","conditioning_manifest.json"):
            if not (path/name).exists():
                (path/name).symlink_to(screen.audit.CACHE/name)
    screen.write(ROOT/"protocol.json",{
        "status":"running","cases":CASES,"candidate":best,"arms":["dense","baseline",best],
        "seed":42,"resolution":[1344,768],"requested_frames":120,"steps":20,
        "warmup":"current 4 dense evaluations + layer0; one full baseline generation excluded per worker",
        "selection":"best mean QKV output relative L2 among three capacity-preserving swap arms, not held-out selection",
        "scope":"small in-sample full-generation follow-up; not 10s validation or a VBench ranking",
        "timing":"resident synchronized denoise; one measurement; candidate adapter unfused and recomputes M2",
        "runner_sha256":screen.sha(Path(__file__)),"adapter_sha256":screen.sha(Path(screen.__file__)),
        "frozen_package":str(screen.ROOT/"source"),"started_unix":time.time()})
    shutil.copy2(Path(__file__),ROOT/"runner_source.py")


def make_swap_adapter(name,original):
    from h3_sparse_attention.landmark_direction import landmark_direction_factors
    from h3_sparse_attention.mahalanobis_kmeans import hilbert_midpoint_sample_indices
    def adapter(controller,qb,kb,layout):
        assert controller.config.sol_virtual_query_levels_up == 99
        result = list(original(controller,qb,kb,layout))
        b,n,h,d = qb.shape
        count = layout.video_tokens
        q,k = [x[:,:count].permute(0,2,1,3).reshape(b*h,count,d) for x in (qb,kb)]
        indices = hilbert_midpoint_sample_indices(tuple(layout.grid),count//64,device=q.device)
        qm,km = landmark_direction_factors(q,k,indices,ridge=.001,moment="raw")
        for side,source,metric,offset in (("q",q,km,0),("k",k,qm,2)):
            if name == "swap_q4" and side == "k" or name == "swap_k4" and side == "q":
                continue
            feature = torch.bmm(source,metric.to(source.dtype))
            p,_ = screen.swap_refine(feature,result[offset].reshape(b*h,count))
            inverse = torch.empty_like(p)
            inverse.scatter_(1,p,torch.arange(count,device=p.device).expand_as(p))
            result[offset],result[offset+1] = p.reshape(b,h,count),inverse.reshape(b,h,count)
            result[4+(side=="k")] = dict(result[4+(side=="k")],experimental_neighbor_swap_passes=4)
        return tuple(result)
    return adapter


def generate_worker(gpu):
    sys.path.insert(0,str(screen.ROOT/"source"))
    sys.path.insert(0,str(screen.audit.BENCH/"scripts"))
    import _impl_bootstrap
    import minimax_h3_vbench_4gpu_pipeline as pipeline
    import h3_sparse_attention.spark_integration as integration
    from h3_sparse_attention import install_h3_sparse_attention
    torch.set_num_threads(4)
    selected = json.loads((ROOT/"protocol.json").read_text())["candidate"]
    assigned = CASES[gpu::2]
    args = pipeline.build_parser().parse_args([
        "denoise","--samples",str(screen.audit.SAMPLES),"--output",str(ROOT/f"workflow_{gpu}"),
        "--method","dense","--case-indices",*map(str,assigned),"--steps","20", "--frames","120",
        "--height","768","--width","1344","--workers","1"])
    cases = pipeline.load_cases(screen.audit.SAMPLES,list(assigned),expected_indices=tuple(range(1,51)))
    workflow,states = pipeline.configure_denoise_workflow(args,cases)
    pipe,manager,acceleration,placement = pipeline.load_denoiser(args,workflow)
    original = integration._landmark_tree_v2_qk_block_permutations
    def infer(state,arm):
        state = pipeline.clone_state(state)
        state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
        context = pipeline.DensePlugin() if arm == "dense" else install_h3_sparse_attention(pipe.transformer,screen.config())
        integration._landmark_tree_v2_qk_block_permutations = make_swap_adapter(arm,original) if arm == selected else original
        try:
            with context,torch.inference_mode():
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = pipe(state=state,num_frames=120,height=768,width=1344,num_inference_steps=20,
                              generator=torch.Generator(device="cpu").manual_seed(42),output=["latents","audio_latents"])
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
            return output,elapsed
        finally:
            integration._landmark_tree_v2_qk_block_permutations = original
    try:
        warm,_ = infer(states[0],"baseline")
        del warm
        for case,state in zip(cases,states):
            for arm in ("dense","baseline",selected):
                path = ROOT/"latents"/f"{arm}_case{case['index']:02d}.pt"
                if path.exists():
                    continue
                output,elapsed = infer(state,arm)
                assert all(torch.isfinite(output[n]).all() for n in ("latents","audio_latents"))
                path.parent.mkdir(exist_ok=True)
                torch.save({n:output[n].cpu() for n in ("latents","audio_latents")},path)
                screen.write(path.with_suffix(".json"),{"case":case,"arm":arm,"seconds":elapsed,
                    "sha256":screen.sha(path),"finite":True,"physical_gpu":gpu,"placement":placement})
                del output
                print("GENERATED",case["index"],arm,elapsed,flush=True)
    finally:
        integration._landmark_tree_v2_qk_block_permutations = original
        acceleration.remove()


def decode_worker(gpu):
    import numpy as np
    import torch.nn.functional as F
    sys.path.insert(0,str(screen.ROOT/"source"))
    sys.path.insert(0,str(screen.audit.BENCH/"scripts"))
    import _impl_bootstrap
    import minimax_h3_vbench_4gpu_pipeline as pipeline
    torch.set_num_threads(4)
    selected = json.loads((ROOT/"protocol.json").read_text())["candidate"]
    args = pipeline.build_parser().parse_args(["decode","--output",str(ROOT),"--method","dense"])
    pipe,manager,acceleration = pipeline.load_decoder(args)
    coordinate = torch.arange(11,device="cuda",dtype=torch.float32)-5
    kernel1 = torch.exp(-coordinate.square()/4.5)
    kernel1 /= kernel1.sum()
    kernel = (kernel1[:,None]*kernel1[None,:]).expand(3,1,11,11).contiguous()
    def decode(arm,case):
        path = ROOT/"latents"/f"{arm}_case{case:02d}.pt"
        assert screen.sha(path) == json.loads(path.with_suffix(".json").read_text())["sha256"]
        payload = torch.load(path,map_location="cpu",weights_only=False)
        with torch.inference_mode():
            result = pipe(latents=payload["latents"].cuda(),audio_latents=payload["audio_latents"].cuda(),
                          output_type="np",output=["videos","audio","sampling_rate"])
        video = np.ascontiguousarray((result["videos"][0][:120]*255).round().astype(np.uint8))
        assert video.shape == (120,768,1344,3)
        return video
    try:
        for case in CASES[gpu::2]:
            ref = decode("dense",case)
            (ROOT/"rgb").mkdir(exist_ok=True)
            np.save(ROOT/"rgb"/f"dense_case{case:02d}.npy",ref)
            for arm in ("baseline",selected):
                pixels = decode(arm,case)
                np.save(ROOT/"rgb"/f"{arm}_case{case:02d}.npy",pixels)
                sse,count,ssims = 0.,0,[]
                with torch.inference_mode():
                    for start in range(0,120,8):
                        a,b = [torch.from_numpy(x[start:start+8]).cuda().float().permute(0,3,1,2)/255 for x in (ref,pixels)]
                        sse += float((a-b).square().sum())
                        count += a.numel()
                        ma,mb = F.conv2d(a,kernel,groups=3),F.conv2d(b,kernel,groups=3)
                        va,vb = F.conv2d(a*a,kernel,groups=3)-ma*ma,F.conv2d(b*b,kernel,groups=3)-mb*mb
                        cov = F.conv2d(a*b,kernel,groups=3)-ma*mb
                        ssims.extend((((2*ma*mb+.0001)*(2*cov+.0009))/((ma*ma+mb*mb+.0001)*(va+vb+.0009))).flatten(1).mean(1).cpu().tolist())
                screen.write(ROOT/"quality"/f"{arm}_case{case:02d}.json",{
                    "case":case,"arm":arm,"frames":120,"psnr_db":-10*math.log10(sse/count),
                    "ssim":sum(ssims)/len(ssims),"rgb_sha256":hashlib_sha(pixels),
                    "reference_rgb_sha256":hashlib_sha(ref),"rgb_path":str(ROOT/"rgb"/f"{arm}_case{case:02d}.npy")})
                del pixels
                print("SCORED",case,arm,flush=True)
            del ref
    finally:
        acceleration.remove()


def hashlib_sha(array):
    import hashlib
    return hashlib.sha256(memoryview(array)).hexdigest()


def launch(stage):
    (ROOT/"logs").mkdir(exist_ok=True)
    def run(gpu):
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),H3_IMPL_REPO=str(screen.ROOT/"source"),
                   OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",PYTHONUNBUFFERED="1")
        with (ROOT/"logs"/f"{stage}_{gpu}.log").open("a") as log:
            p = subprocess.Popen([screen.PYTHON,str(Path(__file__).resolve()),stage+"_worker",str(gpu)],
                                 env=env,stdout=log,stderr=subprocess.STDOUT)
            screen.write(ROOT/f"process_{stage}_{gpu}.json",{"pid":p.pid,"gpu":gpu})
            return p.wait()
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        codes = list(pool.map(run,range(2)))
    if any(codes):
        raise RuntimeError(f"{stage} return codes: {codes}")


def aggregate():
    p = json.loads((ROOT/"protocol.json").read_text())
    rows = []
    for case in CASES:
        for arm in ("baseline",p["candidate"]):
            row = json.loads((ROOT/"quality"/f"{arm}_case{case:02d}.json").read_text())
            row["denoise_seconds"] = json.loads((ROOT/"latents"/f"{arm}_case{case:02d}.json").read_text())["seconds"]
            rows.append(row)
    means = {arm:{m:sum(r[m] for r in rows if r["arm"]==arm)/len(CASES)
                  for m in ("psnr_db","ssim","denoise_seconds")} for arm in ("baseline",p["candidate"])}
    screen.write(ROOT/"results.json",{"status":"complete","rows":rows,"means":means,"protocol":p})
    p.update(status="complete",completed_unix=time.time())
    screen.write(ROOT/"protocol.json",p)
    print(json.dumps(means,indent=2),flush=True)


def run():
    prepare()
    launch("generate")
    launch("decode")
    aggregate()


if __name__ == "__main__":
    name = sys.argv[1]
    if name.endswith("_worker"):
        globals()[name](int(sys.argv[2]))
    else:
        globals()[name]()
