#!/usr/bin/env python3
"""Paired 5s768p Dense-QKV screen: layouts, their own oracles, and summaries.

Production modules are read-only. Capture and analysis are separate stages so
the model and the experimental layouts never compete for GPU memory. All
layouts evaluate the same original query-token anchors, including their
containing Q64 blocks; all heads/steps/layers/prompts are paired.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
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
sys.path.insert(0, str(REPO))
NAME = "reblock_approx_10prompt_5s768p_20261002"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
CASES = (1, 6, 11, 15, 21, 27, 32, 38, 44, 50)
EVALUATIONS = (4, 7, 11, 15, 18)
LAYERS = (1, 12, 24, 25, 36, 49)
HEADS = (0, 7, 15, 23, 31, 39, 47, 55)
ANCHORS = 8
LAYOUTS = (
    "baseline", "flat", "q_only", "k_only", "raw_cosine",
    "metric_euclidean", "unit_moment", "hellinger_q32", "hellinger_q128",
    "swap_q4", "swap_k4", "swap_both4",
)
APPROX = (
    "uniform", "global_native", "anchor_scale05", "anchor_scale15",
    "local_q64", "two_summary_k32", "oracle_mass", "oracle_value", "oracle_both",
)
PYTHON = "/root/miniconda3/bin/python"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = module(REPO / "scripts/verify_shared_k64_oracle_10prompt_20261002.py", "expanded_audit")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config():
    from h3_sparse_attention import H3SparseAttentionConfig
    return H3SparseAttentionConfig.spark(
        20, warmup_percent=20, sol_dense_layers=1,
        sol_force_local_blocks=False, sol_log_density=False,
        landmark_tree_v2_children=16, landmark_tree_v2_initial_order="flat")


def prepare():
    if ROOT.exists():
        return
    ROOT.mkdir(parents=True)
    source_dir = ROOT / "source"
    source_dir.mkdir()
    shutil.copytree(REPO / "h3_sparse_attention", source_dir / "h3_sparse_attention",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(Path(__file__), source_dir / Path(__file__).name)
    shutil.copy2(REPO / "scripts/verify_shared_k64_oracle_10prompt_20261002.py",
                 source_dir / "verify_shared_k64_oracle_10prompt_20261002.py")
    for gpu in range(2):
        out = ROOT / f"workflow_{gpu}"
        out.mkdir()
        for name in ("conditioning_cache", "conditioning_manifest.json"):
            (out / name).symlink_to(audit.CACHE / name)
    write(ROOT / "protocol.json", {
        "status": "prepared", "created_unix": time.time(),
        "cases": CASES, "evaluations": EVALUATIONS, "layers": LAYERS,
        "heads": HEADS, "model_total_heads": 56, "anchors_per_head": ANCHORS,
        "seed": 42, "resolution": [1344, 768], "requested_frames": 120,
        "seconds": 5, "requested_steps": 20, "actual_evaluations": 19,
        "expected_captures": len(CASES)*len(EVALUATIONS)*len(LAYERS),
        "layout_arms": LAYOUTS, "approximation_arms": APPROX,
        "trajectory": "unchanged compiled Diffusers Dense, BF16, seed42",
        "sampling": "identical original query-token IDs for every layout; uniformly random from baseline active queries without replacement",
        "budget": "round(0.10 * complete_video_K64_blocks) plus identical exact video tail/context",
        "oracle": "each layout's own mean normalized attention mass over its Q64 rows; no finer exact tiles",
        "comparison": "paired per prompt; prompt-cluster bootstrap, not head/token pseudo-replication",
        "limits": "5s is a screening proxy, not evidence of 10s transfer; one seed; mathematical attention diagnostic, not end-to-end speed",
        "packing": "four passes of capacity-preserving adjacent K64/Q64 swaps (max16 per pair); tree already has exact leaf capacities",
        "tail_control": "production protects largest transformed-norm remainder tokens; freeze baseline Q/K excluded IDs for every geometry, including matched-tail flat controls",
        "approximation": "fixed baseline layout and native mask; full video/context normalization; production native summary kernel",
        "approximation_cost": "K32 adds a second summary per omitted K64; Q64-local anchors are a costly upper-bound probe; oracles are nondeployable",
        "route_config": dataclasses.asdict(config()),
        "source_sha256": {str(p.relative_to(source_dir)): sha(p)
                           for p in source_dir.rglob("*.py")},
    })


class CaptureCollector(audit.Collector):
    def record(self, layer, attn, hidden, rotary):
        if self.evaluation not in EVALUATIONS or layer not in LAYERS:
            return
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        if attn.fused_projections:
            rawq, rawk, rawv = attn.to_qkv(hidden).chunk(3, dim=-1)
        else:
            rawq, rawk, rawv = attn.to_q(hidden), attn.to_k(hidden), attn.to_v(hidden)
        q = attn.norm_q(rawq.unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(rawk.unflatten(-1, (attn.heads, -1)))
        v = rawv.unflatten(-1, (attn.heads, -1))
        if rotary is not None:
            q, k = _apply_rotary_emb(q, *rotary), _apply_rotary_emb(k, *rotary)
        h = torch.tensor(HEADS, device=hidden.device)
        q, k, v = [x[0].index_select(1, h).permute(1, 0, 2) for x in (q, k, v)]
        context = torch.ones(hidden.shape[1], device=hidden.device, dtype=torch.bool)
        context[self.video_rows] = False
        context = context.nonzero(as_tuple=False).flatten()
        payload = {
            "case": self.case, "evaluation": self.evaluation, "layer": layer,
            "grid": self.grid, "heads": HEADS, "trajectory": "dense_reference",
            "q": q.index_select(1, self.video_rows).contiguous().cpu(),
            "k": k.index_select(1, self.video_rows).contiguous().cpu(),
            "v": v.index_select(1, self.video_rows).contiguous().cpu(),
            "context_k": k.index_select(1, context).contiguous().cpu(),
            "context_v": v.index_select(1, context).contiguous().cpu(),
        }
        path = ROOT / "captures" / f"case{self.case:02d}_eval{self.evaluation:02d}_layer{layer:02d}.pt"
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        write(path.with_suffix(".json"), {
            "path": str(path), "sha256": sha(path), "grid": self.grid,
            "video_tokens": payload["q"].shape[1], "context_tokens": context.numel(),
            "case": self.case, "evaluation": self.evaluation, "layer": layer,
        })
        self.units += 1
        print("CAPTURE", path.name, flush=True)


def capture_worker(gpu):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    audit.ROOT, audit.HEADS = ROOT, HEADS
    audit.LAYERS = LAYERS
    sys.path.insert(0, str(audit.BENCH / "scripts"))
    import _impl_bootstrap
    import minimax_h3_vbench_4gpu_pipeline as pipeline
    assigned = CASES[gpu::2]
    args = pipeline.build_parser().parse_args([
        "denoise", "--samples", str(audit.SAMPLES), "--output", str(ROOT / f"workflow_{gpu}"),
        "--method", "dense", "--case-indices", *map(str, assigned),
        "--steps", "20", "--frames", "120", "--height", "768", "--width", "1344", "--workers", "1"])
    cases = pipeline.load_cases(audit.SAMPLES, list(assigned), expected_indices=tuple(range(1, 51)))
    workflow, states = pipeline.configure_denoise_workflow(args, cases)
    pipe, manager, acceleration, placement = pipeline.load_denoiser(args, workflow)
    base = audit.CurrentRoute()
    try:
        for case, state in zip(cases, states):
            done = ROOT / "capture_cases" / f"case{case['index']:02d}.json"
            if done.exists():
                continue
            collector = CaptureCollector(pipe.transformer, case["index"], base)
            collector.install()
            state = pipeline.clone_state(state)
            state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
            start = time.perf_counter()
            try:
                with torch.inference_mode():
                    output = pipe(state=state, num_frames=120, height=768, width=1344,
                                  num_inference_steps=20, generator=torch.Generator(device="cpu").manual_seed(42),
                                  output=["latents", "audio_latents"])
                assert collector.evaluation == 18 and collector.units == len(EVALUATIONS)*len(LAYERS)
                assert all(torch.isfinite(output[n]).all().item() for n in ("latents", "audio_latents"))
                write(done, {"status": "complete", "case": case, "captures": collector.units,
                             "seconds": time.perf_counter()-start, "finite": True, "placement": placement})
                del output
            finally:
                collector.remove()
            del state, collector
            print("CAPTURE_CASE_COMPLETE", case["index"], flush=True)
    finally:
        acceleration.remove()


def swap_refine(features, permutation, passes=4, max_swaps=16):
    """Preserve exact capacities; swaps strictly reduce fixed-centroid cosine cost.

    Centroids are means of unit feature vectors, so re-estimating their unit
    directions cannot increase the spherical-kmeans objective after a swap.
    Adjacent block pairs alternate parity. The excluded tail never moves.
    """
    heads, tokens, dim = features.shape
    blocks = tokens//64
    p = permutation.clone()
    unit = torch.nn.functional.normalize(features.float(), dim=-1)
    objective = []
    total_swaps = 0
    for iteration in range(passes):
        ids = p[:, :blocks*64].reshape(heads, blocks, 64)
        x = unit.gather(1, ids.reshape(heads, -1)[..., None].expand(-1, -1, dim))
        x = x.reshape(heads, blocks, 64, dim)
        centers = torch.nn.functional.normalize(x.mean(2), dim=-1)
        objective.append(float((1-(x*centers[:, :, None]).sum(-1)).mean()))
        left_ids = torch.arange(iteration%2, blocks-1, 2, device=features.device)
        if not len(left_ids):
            continue
        right_ids = left_ids+1
        left, right = x[:, left_ids], x[:, right_ids]
        delta = centers[:, right_ids]-centers[:, left_ids]
        gl = torch.einsum("hptd,hpd->hpt", left, delta)
        gr = -torch.einsum("hptd,hpd->hpt", right, delta)
        vl, il = gl.topk(min(max_swaps, 64), -1)
        vr, ir = gr.topk(min(max_swaps, 64), -1)
        valid = vl+vr > 1e-6
        total_swaps += int(valid.sum())
        a, b = ids[:, left_ids].clone(), ids[:, right_ids].clone()
        ta, tb = a.gather(-1, il), b.gather(-1, ir)
        a.scatter_(-1, il, torch.where(valid, tb, ta))
        b.scatter_(-1, ir, torch.where(valid, ta, tb))
        ids[:, left_ids], ids[:, right_ids] = a, b
    ids = p[:, :blocks*64].reshape(heads, blocks, 64)
    x = unit.gather(1, ids.reshape(heads, -1)[..., None].expand(-1, -1, dim)).reshape(heads, blocks, 64, dim)
    centers = torch.nn.functional.normalize(x.mean(2), dim=-1)
    objective.append(float((1-(x*centers[:, :, None]).sum(-1)).mean()))
    assert all(b <= a+1e-6 for a,b in zip(objective, objective[1:])), objective
    return p, {"objective": objective, "token_swaps": total_swaps}


class LayoutBuilder:
    def __init__(self):
        self.base = audit.CurrentRoute()
        self.plans = {}

    def tree(self, features, grid, distance="cosine", excluded=None):
        from h3_sparse_attention.landmark_tree_v2 import PreparedLandmarkTreeV2Permutation
        key = (features.shape, tuple(grid), distance)
        plan = self.plans.get(key)
        if plan is None:
            plan = PreparedLandmarkTreeV2Permutation(
                batch=features.shape[0], tokens=features.shape[1], dim=features.shape[2],
                grid_shape=tuple(grid), device=features.device, distance=distance,
                max_children=16, fanout_mode="power_of_two_fanout",
                landmark_mode="midpoint", landmark_count=32, midpoint_direction_mode="legacy",
                initial_order="flat", order_mode="parent_order", group_size=1)
            self.plans[key] = plan
        features = features.to(torch.bfloat16).contiguous()
        if excluded is not None and excluded.shape[1]:
            # Only excluded tokens are modified, then discarded by initial
            # partitioning. Their large norm fixes tail IDs without altering
            # any active token's clustering feature or the Q/K attention data.
            features = features.clone()
            h = torch.arange(features.shape[0], device=features.device)[:,None]
            # Early layers can have much larger transformed feature norms
            # than the layer25 pilot. Bound every active vector's norm by
            # sqrt(D)*max_abs, rather than using an arbitrary fixed constant.
            strength = features.abs().amax((1,2)).float()*(2*math.sqrt(features.shape[-1]))+1
            assert torch.isfinite(strength).all()
            features[h,excluded] = 0
            features[h,excluded,0] = strength[:,None].to(features.dtype)
        p, _ = plan.run(features)
        return p.clone()

    def build(self, q, k, grid):
        from h3_sparse_attention.landmark_direction import landmark_direction_factors
        from h3_sparse_attention.mahalanobis_kmeans import hilbert_midpoint_sample_indices
        n = q.shape[1]
        identity = torch.arange(n, device=q.device).expand(q.shape[0], -1)
        qp, kp, metadata = self.base.production_partition(q, k, grid, self.base.cfg)
        cut = n//64*64
        qe,ke = qp[:,cut:],kp[:,cut:]
        def flat_with_tail(excluded):
            active = torch.ones_like(identity,dtype=torch.bool).scatter_(1,excluded,False)
            return torch.cat((identity[active].reshape(q.shape[0],cut),excluded),-1)
        qflat,kflat = flat_with_tail(qe),flat_with_tail(ke)
        indices = hilbert_midpoint_sample_indices(tuple(grid), n//64, device=q.device)
        qm, km = landmark_direction_factors(q, k, indices, ridge=.001, moment="raw")
        qf, kf = torch.bmm(q, km.to(q.dtype)), torch.bmm(k, qm.to(k.dtype))
        # Independent feature adapter must exactly reproduce the production partition.
        parity = self.tree(torch.cat((kf,qf)), grid)
        assert torch.equal(parity[:q.shape[0]], kp) and torch.equal(parity[q.shape[0]:], qp), "production layout parity failed"
        excluded = torch.cat((ke,qe))
        layouts = {"baseline": (qp,kp), "flat": (qflat,kflat),
                   "q_only": (qp,kflat), "k_only": (qflat,kp)}
        p = self.tree(torch.cat((k,q)), grid, excluded=excluded)
        layouts["raw_cosine"] = (p[q.shape[0]:],p[:q.shape[0]])
        p = self.tree(torch.cat((kf,qf)), grid, distance="euclidean", excluded=excluded)
        layouts["metric_euclidean"] = (p[q.shape[0]:],p[:q.shape[0]])
        qm2, km2 = landmark_direction_factors(q, k, indices, ridge=.001, moment="unit")
        p = self.tree(torch.cat((torch.bmm(k,qm2.to(k.dtype)),torch.bmm(q,km2.to(q.dtype)))), grid, excluded=excluded)
        layouts["unit_moment"] = (p[q.shape[0]:],p[:q.shape[0]])
        for count in (32,128):
            ids = hilbert_midpoint_sample_indices(tuple(grid), count, device=q.device)
            logits = torch.bmm(q,k[:,ids].transpose(1,2)).float()*q.shape[-1]**-.5
            feature = logits.softmax(-1).sqrt().to(torch.bfloat16)
            layouts[f"hellinger_q{count}"] = (self.tree(feature,grid,distance="euclidean",excluded=qe),kp)
        qs, qdiag = swap_refine(qf, qp)
        ks, kdiag = swap_refine(kf, kp)
        layouts.update(swap_q4=(qs,kp),swap_k4=(qp,ks),swap_both4=(qs,ks))
        for a,b in layouts.values():
            assert torch.equal(a.sort(-1).values,identity) and torch.equal(b.sort(-1).values,identity)
            assert torch.equal(a[:,cut:],qe) and torch.equal(b[:,cut:],ke)
        return layouts, {"production":metadata,"query_swap":qdiag,"key_swap":kdiag}


def summaries(q, k, v, qids=None, scale=1., components="full"):
    """Use the production tensorcore summary and anchor construction kernels."""
    from h3_sparse_attention.sol_numerator_virtual_q import build_virtual_anchors, virtual_summaries
    qt = q.transpose(0,1)[None].contiguous()
    cut = q.shape[1]//64*64
    if qids is None:
        ranges = torch.tensor([[0,cut]],device=q.device,dtype=torch.long)
        anchor = build_virtual_anchors(qt,ranges)
    else:
        # qids: one Q64 ID per anchor and head; compute per-head local means.
        qb = q[:,:cut].reshape(q.shape[0],cut//64,64,q.shape[-1])
        hi = torch.arange(q.shape[0],device=q.device)[:,None]
        anchor = qb[hi,qids].float().mean(-2).to(q.dtype).permute(1,0,2)[None].contiguous()
    if scale != 1.:
        anchor = (anchor.float()*scale).to(q.dtype)
    kt, vt = [x[:,:cut].transpose(0,1)[None].contiguous() for x in (k,v)]
    ak,av,lm = virtual_summaries(anchor,kt,vt,reweight_components=components)
    # [parents, heads, complete K64, dim] and [parents, heads, K64].
    return ak[0].float(),av[0].float(),lm[0],anchor[0].float()


def two_summaries(k,v,anchor):
    h,t,d = k.shape
    count = t//64
    kb,vb = [x[:,:count*64].float().reshape(h,count,2,32,d) for x in (k,v)]
    a = anchor[0]
    logits = torch.einsum("hcsrd,hd->hcsr",kb,a)*d**-.5
    weights = logits.softmax(-1)
    kc = torch.einsum("hcsr,hcsrd->hcsd",weights,kb).to(k.dtype).float()
    vc = torch.einsum("hcsr,hcsrd->hcsd",weights,vb).to(v.dtype).float()
    bias = torch.logsumexp(logits,-1)-torch.einsum("hcsd,hd->hcs",kc,a)*d**-.5
    return kc,vc,bias


def relative_error(output,reference):
    return (output-reference).norm(dim=-1)/reference.norm(dim=-1).clamp_min(1e-12)


def mix_summary(logmass,values,exact_logmass,exact_conditional):
    """Normalize in log space, including severely peaked early-layer heads.

    ``logmass`` and ``exact_logmass`` are relative to the true Dense Z.
    Directly exponentiating them can overflow or underflow even when the
    mixed attention's normalized output is perfectly finite.
    """
    total = torch.logaddexp(exact_logmass,logmass.logsumexp(-1))
    p = (logmass-total[:,None]).exp()
    approximate = p@values if values.ndim == 2 else torch.einsum("ac,acd->ad",p,values)
    out = approximate+(exact_logmass-total).exp()[:,None]*exact_conditional
    # Absolute mass ratios can exceed even FP64. Keep a documented finite
    # display cap, and also return the uncapped absolute log-ratio error.
    return out,total.clamp(max=80).expm1().abs(),total.abs()


def evaluate_capture(cap, builder):
    start = time.perf_counter()
    q0,k0,v0,ck,cv = [cap[n].cuda() for n in ("q","k","v","context_k","context_v")]
    assert q0.dtype == torch.bfloat16 and q0.shape == k0.shape == v0.shape
    heads,n,d = q0.shape
    complete,cut = n//64,n//64*64
    quota = max(1,round(.1*complete))
    # One seeded, paired anchor sample per head and capture, independent of layouts.
    generator = torch.Generator(device="cpu").manual_seed(42+cap["case"]*10000+cap["evaluation"]*100+cap["layer"])
    anchor_positions = torch.stack([torch.randperm(cut,generator=generator)[:ANCHORS] for _ in range(heads)]).cuda()
    torch.cuda.synchronize()
    build_start = time.perf_counter()
    layouts,metadata = builder.build(q0,k0,cap["grid"])
    anchor_ids = layouts["baseline"][0].gather(1,anchor_positions)
    torch.cuda.synchronize()
    build_seconds = time.perf_counter()-build_start
    rows = []
    for name,(qp,kp) in layouts.items():
        q = q0.gather(1,qp[...,None].expand(-1,-1,d))
        k,v = [x.gather(1,kp[...,None].expand(-1,-1,d)) for x in (k0,v0)]
        inverse = torch.empty_like(qp)
        inverse.scatter_(1,qp,torch.arange(n,device=q.device).expand_as(qp))
        packed_anchor = inverse.gather(1,anchor_ids)
        qids,within = packed_anchor//64,packed_anchor%64
        native_summary = summaries(q,k,v)
        extra = {}
        if name == "baseline":
            extra = {
                "uniform":summaries(q,k,v,components="none"),
                "anchor_scale05":summaries(q,k,v,scale=.5),
                "anchor_scale15":summaries(q,k,v,scale=1.5),
                "local_q64":summaries(q,k,v,qids=qids),
                "two_summary_k32":two_summaries(k,v,native_summary[3]),
            }
        for head in range(heads):
            query = q[head,:cut].reshape(complete,64,d)[qids[head]].float()
            query_flat = query.reshape(-1,d)
            keys = torch.cat((k[head].float(),ck[head].float()))
            vals = torch.cat((v[head].float(),cv[head].float()))
            logits = query_flat@keys.T*d**-.5
            rowmax = logits.max(-1).values
            # Keep the small log denominator separate from huge early-layer
            # logits, avoiding cancellation in partial-oracle reconstruction.
            logz = (logits-rowmax[:,None]).logsumexp(-1)
            prob = logits.softmax(-1)
            mass = prob[:,:cut].reshape(ANCHORS,64,complete,64).sum(-1)
            mean_mass = mass.mean(1)
            score = query.mean(1)@k[head,:cut].float().reshape(complete,64,d).mean(1).T*d**-.5
            chosen = score.topk(quota,-1).indices
            oracle = mean_mass.topk(quota,-1).indices
            mask = torch.zeros_like(mean_mass,dtype=torch.bool).scatter_(1,chosen,True)
            block_native = mean_mass.gather(1,chosen).sum(-1)
            block_oracle = mean_mass.gather(1,oracle).sum(-1)
            # Same original query token per anchor across all layouts.
            ri = torch.arange(ANCHORS,device=q.device)*64+within[head]
            qa = query_flat[ri]
            pa = prob[ri]
            ma = mass[torch.arange(ANCHORS,device=q.device),within[head]]
            reference = pa@vals
            exact_mask = mask.repeat_interleave(64,-1)
            exact_p = pa[:,:cut]*exact_mask
            exact_mass = exact_p.sum(-1)+pa[:,cut:].sum(-1)
            full_exact_mask = torch.cat((exact_mask,torch.ones(ANCHORS,keys.shape[0]-cut,device=q.device,dtype=torch.bool)),-1)
            exact_logits = (logits[ri]-rowmax[ri,None]).masked_fill(~full_exact_mask,-float("inf"))
            exact_logmass = exact_logits.logsumexp(-1)-logz[ri]
            exact_conditional = exact_logits.softmax(-1)@vals
            ak,av,lm,_ = native_summary
            al = (qa@ak[0,head].T*d**-.5-rowmax[ri,None])+lm[0,head][None]-logz[ri,None]
            al = al.masked_fill(mask,-float("inf"))
            native_out,native_mass_error,native_log_error = mix_summary(al,av[0,head],exact_logmass,exact_conditional)
            oracle_anchor = ma.gather(1,oracle).sum(-1)+pa[:,cut:].sum(-1)
            # Per-row K64 oracle is only a K-layout diagnostic, never an exact-tile candidate.
            row_oracle = ma.topk(quota,-1).values.sum(-1)+pa[:,cut:].sum(-1)
            output_errors = {"global_native":relative_error(native_out,reference)}
            mass_errors = {"global_native":native_mass_error}
            log_errors = {"global_native":native_log_error}
            if name == "baseline":
                for method,item in extra.items():
                    if method == "two_summary_k32":
                        ac,vc,bc = item
                        local_logits = (torch.einsum("ad,csd->acs",qa,ac[head])*d**-.5-rowmax[ri,None,None])+bc[head][None]-logz[ri,None,None]
                        local_logits = local_logits.masked_fill(mask[...,None],-float("inf")).flatten(1)
                        summary_values = vc[head].reshape(complete*2,d)
                    else:
                        ac,vc,bc,_ = item
                        if method == "local_q64":
                            local_logits = (torch.einsum("ad,acd->ac",qa,ac[:,head])*d**-.5-rowmax[ri,None])+bc[:,head]-logz[ri,None]
                            summary_values = vc[:,head]
                        else:
                            local_logits = (qa@ac[0,head].T*d**-.5-rowmax[ri,None])+bc[0,head][None]-logz[ri,None]
                            summary_values = vc[0,head]
                        local_logits = local_logits.masked_fill(mask,-float("inf"))
                    out,mass_error,log_error = mix_summary(local_logits,summary_values,exact_logmass,exact_conditional)
                    output_errors[method] = relative_error(out,reference)
                    mass_errors[method],log_errors[method] = mass_error,log_error
                true_p = ma.masked_fill(mask,0)
                token_probs = pa[:,:cut].reshape(ANCHORS,complete,64)
                true_value = torch.einsum("act,ctd->acd",token_probs,v[head,:cut].float().reshape(complete,64,d))/ma[...,None].clamp_min(1e-30)
                for method,logp,vc in (
                    ("oracle_mass",true_p.log(),av[0,head][None].expand(ANCHORS,-1,-1)),
                    ("oracle_value",al,true_value),
                    ("oracle_both",true_p.log(),true_value),
                ):
                    out,mass_error,log_error = mix_summary(logp,vc,exact_logmass,exact_conditional)
                    output_errors[method] = relative_error(out,reference)
                    mass_errors[method],log_errors[method] = mass_error,log_error
                assert float(output_errors["oracle_both"].max()) < 2e-4
            numbers = torch.stack((block_native,block_oracle,block_native/block_oracle,
                                   exact_mass,oracle_anchor,row_oracle,output_errors["global_native"]),-1).cpu().tolist()
            for a,values in enumerate(numbers):
                row = {"layout":name,"head":HEADS[head],"anchor":a,
                       "original_query_token":int(anchor_ids[head,a]),"query_block":int(qids[head,a]),
                       **dict(zip(("block_native_mass","block_oracle_mass","block_efficiency",
                                   "paired_query_native_mass","paired_query_block_oracle_mass",
                                   "paired_query_row_oracle_mass","output_relative_l2"),values))}
                if name == "baseline":
                    row["approximation_relative_l2"] = {m:float(e[a]) for m,e in output_errors.items()}
                    row["approximation_total_mass_abs_error"] = {m:float(e[a]) for m,e in mass_errors.items()}
                    row["approximation_log_total_mass_error"] = {m:float(e[a]) for m,e in log_errors.items()}
                assert all(math.isfinite(x) for x in values), row
                rows.append(row)
    torch.cuda.synchronize()
    return {"case":cap["case"],"evaluation":cap["evaluation"],"layer":cap["layer"],
            "video_tokens":n,"context_tokens":ck.shape[1],"complete_blocks":complete,"quota":quota,
            "layout_build_seconds":build_seconds,"seconds":time.perf_counter()-start,
            "metadata":metadata,"rows":rows}


def analyze_worker(gpu):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    shutil.copy2(Path(__file__),ROOT/f"effective_analysis_source_{gpu}.py")
    write(ROOT/f"analysis_source_{gpu}.json",{"runner_sha256":sha(Path(__file__)),
          "note":"adaptive tail protection and stable log-domain mixing; prior partial units retained separately, all final units recomputed from this source",
          "torch_version":torch.__version__,"cuda_version":torch.version.cuda,
          "device":torch.cuda.get_device_name(),"started_unix":time.time()})
    builder = LayoutBuilder()
    for case in CASES[gpu::2]:
        for evaluation in EVALUATIONS:
            for layer in LAYERS:
                stem = f"case{case:02d}_eval{evaluation:02d}_layer{layer:02d}"
                target = ROOT/"units"/f"{stem}.json"
                if target.exists():
                    continue
                path = ROOT/"captures"/f"{stem}.pt"
                manifest = json.loads(path.with_suffix(".json").read_text())
                assert sha(path) == manifest["sha256"], path
                cap = torch.load(path,map_location="cpu",mmap=True,weights_only=False)
                with torch.inference_mode():
                    result = evaluate_capture(cap,builder)
                result["analysis_source_sha256"] = sha(Path(__file__))
                write(target,result)
                print("ANALYZED",stem,round(result["seconds"],2),flush=True)
                del cap,result


def launch(stage):
    (ROOT/"logs").mkdir(exist_ok=True)
    def run(gpu):
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),H3_IMPL_REPO=str(ROOT/"source"),
                   OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",PYTHONUNBUFFERED="1")
        with (ROOT/"logs"/f"{stage}_{gpu}.log").open("a") as log:
            p = subprocess.Popen([PYTHON,str(Path(__file__).resolve()),stage+"_worker",str(gpu)],
                                 env=env,stdout=log,stderr=subprocess.STDOUT)
            write(ROOT/f"process_{stage}_{gpu}.json",{"pid":p.pid,"stage":stage,"gpu":gpu})
            return p.wait()
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        codes = list(pool.map(run,range(2)))
    if any(codes):
        raise RuntimeError(f"{stage} worker return codes: {codes}")


def aggregate():
    import numpy as np
    files = sorted((ROOT/"units").glob("*.json"))
    expected = len(CASES)*len(EVALUATIONS)*len(LAYERS)
    assert len(files) == expected,(len(files),expected)
    docs = [json.loads(p.read_text()) for p in files]
    sources = {u["analysis_source_sha256"] for u in docs}
    assert len(sources) == 1,sources
    rows = [{"case":u["case"],"evaluation":u["evaluation"],"layer":u["layer"],**r} for u in docs for r in u["rows"]]
    assert len(rows) == expected*len(LAYOUTS)*len(HEADS)*ANCHORS
    metrics = ("block_native_mass","block_oracle_mass","block_efficiency","paired_query_native_mass",
               "paired_query_block_oracle_mass","paired_query_row_oracle_mass","output_relative_l2")
    def average(rs):
        return {m:sum(r[m] for r in rs)/len(rs) for m in metrics}
    layouts = {name:average([r for r in rows if r["layout"] == name]) for name in LAYOUTS}
    per_case = {str(case):{name:average([r for r in rows if r["case"] == case and r["layout"] == name])
                          for name in LAYOUTS} for case in CASES}
    rng = np.random.default_rng(42)
    resample = rng.integers(0,len(CASES),size=(10000,len(CASES)))
    deltas = {}
    for name in LAYOUTS:
        deltas[name] = {}
        for metric in metrics:
            values = np.array([per_case[str(c)][name][metric]-per_case[str(c)]["baseline"][metric] for c in CASES])
            deltas[name][metric] = {"mean":float(values.mean()),"paired_prompt_bootstrap_95ci":np.percentile(values[resample].mean(1),[2.5,97.5]).tolist(),
                                   "positive_prompts":int((values>0).sum()),"negative_prompts":int((values<0).sum())}
    baseline = [r for r in rows if r["layout"] == "baseline"]
    approx = {}
    for name in APPROX:
        per_prompt = np.array([np.mean([r["approximation_relative_l2"][name] for r in baseline if r["case"] == c]) for c in CASES])
        reference = np.array([np.mean([r["approximation_relative_l2"]["global_native"] for r in baseline if r["case"] == c]) for c in CASES])
        delta = per_prompt-reference
        approx[name] = {"output_relative_l2":float(per_prompt.mean()),"paired_delta_vs_native":float(delta.mean()),
                        "paired_prompt_bootstrap_delta_95ci":np.percentile(delta[resample].mean(1),[2.5,97.5]).tolist(),
                        "improved_prompts":int((delta<0).sum()),
                        "log_total_mass_abs_error":float(np.mean([r["approximation_log_total_mass_error"][name] for r in baseline])),
                        "total_mass_abs_error":float(np.mean([r["approximation_total_mass_abs_error"][name] for r in baseline]))}
    groups = {axis:{str(key):{name:average([r for r in rows if r[axis] == key and r["layout"] == name])
                              for name in LAYOUTS} for key in sorted({r[axis] for r in rows})}
              for axis in ("evaluation","layer","head")}
    result = {"status":"complete","captures":len(files),"rows":len(rows),"analysis_source_sha256":next(iter(sources)),"layout_means":layouts,
              "layout_paired_deltas":deltas,"per_prompt":per_case,"groups":groups,
              "approximation_means":approx,"runtime_seconds":sum(u["seconds"] for u in docs),
              "note":"common original query anchors; full-key normalized metrics; different-layout Q64 means contain different query populations",
              "layout_build_seconds_mean":sum(u["layout_build_seconds"] for u in docs)/len(docs)}
    write(ROOT/"results.json",result)
    protocol = json.loads((ROOT/"protocol.json").read_text())
    protocol.update(status="screen_complete",completed_unix=time.time())
    write(ROOT/"protocol.json",protocol)
    print(json.dumps({"layout_means":layouts,"approximation_means":approx},indent=2),flush=True)


def run():
    prepare()
    launch("capture")
    launch("analyze")
    aggregate()


def analyze():
    launch("analyze")
    aggregate()


def progress():
    print(json.dumps({"capture_cases":len(list((ROOT/"capture_cases").glob("*.json"))),
                      "captures":len(list((ROOT/"captures").glob("*.pt"))),
                      "analyzed":len(list((ROOT/"units").glob("*.json"))),"expected_captures":300}))


def pilot():
    """Validate every arm on one real, already-completed capture."""
    sys.path.insert(0,str(ROOT/"source"))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    path = ROOT/"captures"/"case01_eval04_layer25.pt"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert sha(path) == metadata["sha256"]
    cap = torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    with torch.inference_mode():
        result = evaluate_capture(cap,LayoutBuilder())
    write(ROOT/"pilot.json",result)
    for name in LAYOUTS:
        rows = [r for r in result["rows"] if r["layout"] == name]
        print(name,{m:sum(r[m] for r in rows)/len(rows) for m in
                    ("block_native_mass","block_oracle_mass","paired_query_native_mass","output_relative_l2")},flush=True)
    print("PILOT_SECONDS",result["seconds"],flush=True)


def archive_watch():
    """Incrementally persist a task-owned RAM cache, then restore durable paths.

    This optional IO-only helper does not participate in numerical analysis.
    Never delete the RAM data; retain its symlink as an additional copy until
    the durable hashes have been independently checked.
    """
    logical = ROOT/"captures"
    if not logical.is_symlink():
        raise RuntimeError("archive_watch requires the task-owned memory cache symlink")
    memory = logical.resolve()
    if memory.parent != Path("/dev/shm") or not memory.name.startswith("h3_reblock_approx_20261002."):
        raise RuntimeError(f"unexpected memory cache: {memory}")
    durable = ROOT/"captures_disk_backup"
    durable.mkdir(exist_ok=True)
    while True:
        subprocess.run(["rsync","-a","--exclude=*.tmp*",str(memory)+"/",str(durable)+"/"],check=True)
        count = len(list(durable.glob("*.pt")))
        write(ROOT/"archive_status.json",{"status":"copying","durable_captures":count,
                                         "memory":str(memory),"updated_unix":time.time()})
        print("ARCHIVED",count,flush=True)
        if (ROOT/"results.json").exists():
            break
        time.sleep(30)
    expected = len(CASES)*len(EVALUATIONS)*len(LAYERS)
    manifests = list(durable.glob("*.json"))
    assert len(manifests) == expected,len(manifests)
    for manifest in manifests:
        metadata = json.loads(manifest.read_text())
        assert sha(manifest.with_suffix(".pt")) == metadata["sha256"],manifest
    logical.rename(ROOT/"captures_memory_link")
    durable.rename(logical)
    write(ROOT/"archive_status.json",{"status":"complete","durable_captures":expected,
          "hashes_verified":expected,"memory_copy_retained":str(memory),"completed_unix":time.time()})
    print("DURABLE_ARCHIVE_COMPLETE",expected,flush=True)


def smoke():
    """Cheap GPU invariants, including production parity and oracle reconstruction."""
    torch.manual_seed(42)
    q,k,v = [torch.randn(2,272,128,device="cuda",dtype=torch.bfloat16) for _ in range(3)]
    cap = dict(case=1,evaluation=4,layer=25,grid=(1,4,68),heads=(0,7),q=q.cpu(),k=k.cpu(),v=v.cpu(),
               context_k=torch.randn(2,13,128,dtype=torch.bfloat16),context_v=torch.randn(2,13,128,dtype=torch.bfloat16))
    # Use fewer anchors for the synthetic tiny grid.
    global ANCHORS,HEADS
    saved = ANCHORS,HEADS
    ANCHORS,HEADS = 3,(0,7)
    try:
        with torch.inference_mode():
            result = evaluate_capture(cap,LayoutBuilder())
        assert len(result["rows"]) == len(LAYOUTS)*2*3
        print("SMOKE_PASSED",len(result["rows"]),flush=True)
    finally:
        ANCHORS,HEADS = saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command",choices=("prepare","run","analyze","capture_worker","analyze_worker","aggregate","progress","smoke","archive_watch","pilot"))
    parser.add_argument("gpu",nargs="?",type=int)
    args = parser.parse_args()
    if args.command.endswith("_worker"):
        # Load the immutable package snapshot, not a subsequently edited worktree.
        sys.path.insert(0,str(ROOT/"source"))
        globals()[args.command](args.gpu)
    else:
        globals()[args.command]()
