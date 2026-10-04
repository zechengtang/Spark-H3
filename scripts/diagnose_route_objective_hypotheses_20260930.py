#!/usr/bin/env python3
"""One-hour diagnostic of why global-anchor key routing loses to mean routing.

Replays the existing synchronized Dense Q/K/V captures and compares component
ablations, anchor interpolation, production-shared summaries, and two exact
mass routes against attention-mass and exact-vs-approx output-residual oracles.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch


HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
SOURCE = REPO / "scripts/diagnose_global_anchor_route_oracles_20260930.py"
spec = importlib.util.spec_from_file_location("route_oracle_base", SOURCE)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

NAME = "route_objective_hypotheses_2prompt_20260930"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
REPORT = REPO / "reports" / NAME
METHODS = (
    "native_mean",
    "current_key_full",
    "current_key_weights_only",
    "current_key_bias_only",
    "key_stored_bias",
    "production_shared_full",
    "production_shared_weights_only",
    "anchor_025",
    "anchor_050",
    "anchor_075",
    "exact_lme_mean_query",
    "exact_pair_lme",
)
ORACLES = base.ORACLES


def theoretical_summary(blocks, anchor, values=None, *, stored_bias=False):
    scale = base.HEAD_DIM**-0.5
    logits = torch.einsum("hctd,hd->hct", blocks, anchor) * scale
    weights = logits.softmax(-1)
    centroid_fp32 = torch.einsum("hct,hctd->hcd", weights, blocks)
    centroid = centroid_fp32.to(torch.bfloat16)
    shift_key = centroid.float() if stored_bias else centroid_fp32
    bias = torch.logsumexp(logits, -1) - math.log(base.BLOCK)
    bias -= torch.einsum("hd,hcd->hc", anchor, shift_key) * scale
    value_summary = None
    if values is not None:
        value_summary = torch.einsum("hct,hctd->hcd", weights, values).to(torch.bfloat16)
    return centroid, bias, value_summary


@torch.inference_mode()
def process_unit(cap, layer: int, cfg, prompt: int, evaluation: int):
    record = cap["records"][layer]
    q0, k0, v0 = [record[name].cuda(non_blocking=False) for name in ("q", "k", "v")]
    heads, tokens, dim = q0.shape
    assert q0.shape == k0.shape == v0.shape and heads == 4 and dim == base.HEAD_DIM
    qp, kp, partition_meta = base.production_partition(q0, k0, cap["grid"], cfg)
    q = q0.gather(1, qp[..., None].expand(-1, -1, dim))
    k = k0.gather(1, kp[..., None].expand(-1, -1, dim))
    v = v0.gather(1, kp[..., None].expand(-1, -1, dim))
    complete, scale = tokens // base.BLOCK, base.HEAD_DIM**-0.5
    cut, quota = complete * base.BLOCK, max(1, round(base.TOPK_RATIO * complete))
    sample_blocks = torch.linspace(
        0, complete - 1, base.SAMPLED_QUERY_BLOCKS, device="cuda"
    ).round().long().unique().tolist()

    qf, kf, vf = q.float(), k.float(), v.float()
    qblocks = qf[:, :cut].reshape(heads, complete, base.BLOCK, dim)
    kblocks = kf[:, :cut].reshape(heads, complete, base.BLOCK, dim)
    vblocks = vf[:, :cut].reshape(heads, complete, base.BLOCK, dim)
    q_global = qf.mean(1)
    qmean = qblocks.mean(2).to(torch.bfloat16)
    kmean = kblocks.mean(2).to(torch.bfloat16)
    mean_scores = torch.einsum("hqd,hkd->hqk", qmean, kmean).float() * scale

    kw, kb, _ = theoretical_summary(kblocks, q_global)
    _, kb_stored, _ = theoretical_summary(kblocks, q_global, stored_bias=True)
    weighted_scores = torch.einsum("hqd,hkd->hqk", qmean, kw).float() * scale
    method_scores = {
        "native_mean": mean_scores,
        "current_key_full": weighted_scores + kb[:, None, :],
        "current_key_weights_only": weighted_scores,
        "current_key_bias_only": mean_scores + kb[:, None, :],
        "key_stored_bias": weighted_scores + kb_stored[:, None, :],
    }
    for alpha, name in ((0.25, "anchor_025"), (0.50, "anchor_050"), (0.75, "anchor_075")):
        ka, ba, _ = theoretical_summary(kblocks, q_global * alpha, stored_bias=True)
        method_scores[name] = (
            torch.einsum("hqd,hkd->hqk", qmean, ka).float() * scale
            + ba[:, None, :]
        )

    # Build the exact summaries consumed by the frozen approximation branch.
    from h3_sparse_attention.sol_numerator_virtual_q import build_virtual_anchors, virtual_summaries
    q_bthd = q.transpose(0, 1)[None].contiguous()
    k_bthd = k.transpose(0, 1)[None].contiguous()
    v_bthd = v.transpose(0, 1)[None].contiguous()
    global_range = torch.tensor([[0, tokens]], device="cuda", dtype=torch.int64)
    production_anchor = build_virtual_anchors(q_bthd, global_range)
    production_k, production_v, production_lm = virtual_summaries(
        production_anchor, k_bthd, v_bthd
    )
    production_k = production_k[0, 0, :, :complete]
    production_v = production_v[0, 0, :, :complete]
    production_lm = production_lm[0, 0, :, :complete]
    shared_weight = torch.einsum(
        "hqd,hkd->hqk", qmean, production_k
    ).float() * scale
    method_scores["production_shared_weights_only"] = shared_weight
    # production LM is log-sum-exp; log(64) is common and irrelevant to Top-K.
    method_scores["production_shared_full"] = (
        shared_weight + production_lm[:, None, :] - math.log(base.BLOCK)
    )

    rows = []
    score_diagnostics = []
    log_pairs = math.log(base.BLOCK * base.BLOCK)
    for head in range(heads):
        keys = kblocks[head].reshape(cut, dim)
        values = vblocks[head]
        for query_block in sample_blocks:
            query = qblocks[head, query_block]
            logits = query @ keys.T * scale
            row_max = logits.amax(-1, keepdim=True)
            exponentials = (logits - row_max).exp().reshape(
                base.BLOCK, complete, base.BLOCK
            )
            exact_mass_unscaled = exponentials.sum(-1)
            exact_numerator_unscaled = torch.einsum("qkt,ktd->qkd", exponentials, values)
            total_mass = exact_mass_unscaled.sum(-1)
            total_numerator = exact_numerator_unscaled.sum(1)
            exact_output = total_numerator / total_mass[:, None]
            attention_mass = (exact_mass_unscaled / total_mass[:, None]).mean(0)

            approximate_logmass = (
                query @ production_k[head].float().T * scale
                + production_lm[head][None, :]
            )
            approximate_mass = (approximate_logmass - row_max).exp()
            approximate_numerator = approximate_mass[..., None] * production_v[head].float()[None]
            alternate_denominator = total_mass[:, None] - exact_mass_unscaled + approximate_mass
            alternate_numerator = total_numerator[:, None, :] - exact_numerator_unscaled + approximate_numerator
            alternate_output = alternate_numerator / alternate_denominator[..., None]
            output_residual = (alternate_output - exact_output[:, None]).norm(dim=-1).mean(0)

            exact_pair = torch.logsumexp(
                logits.reshape(base.BLOCK, complete, base.BLOCK), dim=(0, 2)
            ) - log_pairs
            qbar = query.mean(0)
            exact_meanq = torch.logsumexp(
                torch.einsum("ktd,d->kt", kblocks[head], qbar) * scale, dim=-1
            ) - math.log(base.BLOCK)
            local_scores = {
                name: scores[head, query_block] for name, scores in method_scores.items()
            }
            local_scores["exact_lme_mean_query"] = exact_meanq
            local_scores["exact_pair_lme"] = exact_pair
            oracle_values = {
                "attention_mass": attention_mass,
                "single_block_output_residual": output_residual,
                "pair_logmeanexp": exact_pair,
            }
            selected = {}
            row = dict(
                prompt=prompt,
                evaluation=evaluation,
                layer=layer,
                head=int(cap["heads"][head]),
                query_block=query_block,
                candidate_blocks=complete,
                quota=quota,
                methods={},
            )
            for name, score in local_scores.items():
                metrics = {}
                for oracle_name, oracle in oracle_values.items():
                    recall, efficiency, ids = base.topk_metrics(score, oracle, quota)
                    selected[name] = ids
                    metrics[oracle_name] = dict(
                        topk_recall=recall,
                        topk_value_efficiency=efficiency,
                        spearman=base.spearman(score, oracle),
                    )
                delta = score - exact_pair
                metrics["pair_logmeanexp_error"] = dict(
                    mae=float(delta.abs().mean()),
                    rmse=float(delta.square().mean().sqrt()),
                )
                metrics["selected_attention_mass_fraction"] = float(attention_mass[selected[name]].sum())
                row["methods"][name] = metrics
            row["mean_overlap"] = {
                name: float(torch.isin(selected["native_mean"], ids).float().mean())
                for name, ids in selected.items() if name != "native_mean"
            }
            row["bias_diagnostics"] = {
                "current_bias_mean": float(kb[head].mean()),
                "current_bias_min": float(kb[head].min()),
                "current_bias_max": float(kb[head].max()),
                "bias_vs_mass_spearman": base.spearman(kb[head], attention_mass),
                "bias_vs_residual_spearman": base.spearman(kb[head], output_residual),
            }
            current = local_scores["current_key_full"]
            shared = local_scores["production_shared_full"]
            stored = local_scores["key_stored_bias"]
            score_diagnostics.append(dict(
                current_vs_shared_rmse=float((current-shared).square().mean().sqrt()),
                current_vs_shared_spearman=base.spearman(current, shared),
                current_vs_stored_bias_rmse=float((current-stored).square().mean().sqrt()),
            ))
            rows.append(row)

    anchor_ratio = q_global.norm(dim=-1) / qf.norm(dim=-1).mean(1)
    metadata = dict(
        prompt=prompt,
        evaluation=evaluation,
        layer=layer,
        heads=list(cap["heads"]),
        video_tokens=tokens,
        complete_blocks=complete,
        quota=quota,
        sample_query_blocks=sample_blocks,
        partition=partition_meta,
        anchor_norm_ratio=[float(x) for x in anchor_ratio],
        score_diagnostics=score_diagnostics,
    )
    return rows, metadata


def mean(values):
    return statistics.mean(values) if values else float("nan")


def aggregate() -> None:
    documents = [json.loads((ROOT / f"worker_{p}.json").read_text()) for p in base.PROMPTS]
    rows = [row for document in documents for row in document["rows"]]
    summary = {}
    for method in METHODS:
        item = {}
        for oracle in ORACLES:
            item[oracle] = {
                metric: mean([row["methods"][method][oracle][metric] for row in rows])
                for metric in ("topk_recall", "topk_value_efficiency", "spearman")
            }
        item["pair_logmeanexp_error"] = {
            metric: mean([row["methods"][method]["pair_logmeanexp_error"][metric] for row in rows])
            for metric in ("mae", "rmse")
        }
        item["selected_attention_mass_fraction"] = mean([
            row["methods"][method]["selected_attention_mass_fraction"] for row in rows
        ])
        item["mean_topk_overlap"] = 1.0 if method == "native_mean" else mean([
            row["mean_overlap"][method] for row in rows
        ])
        summary[method] = item
    units = [unit for document in documents for unit in document["units"]]
    diagnostics = {
        "anchor_norm_ratio": mean([x for unit in units for x in unit["anchor_norm_ratio"]]),
        "current_vs_shared_score_rmse": mean([
            d["current_vs_shared_rmse"] for unit in units for d in unit["score_diagnostics"]
        ]),
        "current_vs_shared_score_spearman": mean([
            d["current_vs_shared_spearman"] for unit in units for d in unit["score_diagnostics"]
        ]),
        "current_vs_stored_bias_score_rmse": mean([
            d["current_vs_stored_bias_rmse"] for unit in units for d in unit["score_diagnostics"]
        ]),
        "bias_vs_mass_spearman": mean([row["bias_diagnostics"]["bias_vs_mass_spearman"] for row in rows]),
        "bias_vs_residual_spearman": mean([row["bias_diagnostics"]["bias_vs_residual_spearman"] for row in rows]),
        "bias_mean": mean([row["bias_diagnostics"]["current_bias_mean"] for row in rows]),
    }
    result = dict(
        status="complete",
        summary=summary,
        diagnostics=diagnostics,
        rows=rows,
        units=units,
        elapsed_seconds=sum(document["elapsed_seconds"] for document in documents),
        scope=dict(
            prompts=list(base.PROMPTS), evaluations=list(base.EVALUATIONS),
            layers=list(base.LAYERS), heads=[0,1,2,3],
            sampled_query_blocks_per_head=base.SAMPLED_QUERY_BLOCKS,
            row_count=len(rows), trajectory="synchronized Dense-reference",
        ),
    )
    base.write(ROOT / "results.json", result)
    base.write(REPORT / "results.json", result)
    lines = [
        "# Route-objective hypothesis diagnostic", "",
        "Two synchronized Dense trajectories; evaluations 4/11/18, layers 12/24/36/48, heads 0–3, eight query blocks per head.", "",
        "| Route | Mass recall | Residual recall | Residual efficiency | Pair-LME recall | Pair-LME RMSE | Mean overlap |", 
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item=summary[method]
        lines.append(
            f"| {method} | {item['attention_mass']['topk_recall']:.4f} | "
            f"{item['single_block_output_residual']['topk_recall']:.4f} | "
            f"{item['single_block_output_residual']['topk_value_efficiency']:.4f} | "
            f"{item['pair_logmeanexp']['topk_recall']:.4f} | "
            f"{item['pair_logmeanexp_error']['rmse']:.4f} | {item['mean_topk_overlap']:.4f} |"
        )
    lines += ["", "## Diagnostics", "", "```json", json.dumps(diagnostics, indent=2), "```"]
    (ROOT / "REPORT.md").write_text("\n".join(lines)+"\n")
    (REPORT / "REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"summary":summary,"diagnostics":diagnostics},indent=2),flush=True)


def prepare() -> None:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    (ROOT / "logs").mkdir(parents=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    base.install_snapshot()
    base.write(ROOT / "manifest.json", dict(
        status="prepared", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        script=str(HERE), script_sha256=base.sha(HERE), snapshot=str(base.SNAP),
        captures=str(base.CAPTURES), prompts=list(base.PROMPTS),
        evaluations=list(base.EVALUATIONS), layers=list(base.LAYERS),
        methods=list(METHODS), oracles=list(ORACLES), config=asdict(base.config()),
    ))


def worker(prompt: int) -> None:
    base.ROOT = ROOT
    base.process_unit = process_unit
    base.worker(prompt)


def run() -> None:
    prepare()
    devices=os.environ.get("CUDA_VISIBLE_DEVICES","0,1").split(",")
    def launch(item):
        prompt,device=item
        with (ROOT/"logs"/f"worker_{prompt}.log").open("w") as log:
            subprocess.run([sys.executable,str(HERE),"worker",str(prompt)],
                env={**os.environ,"CUDA_VISIBLE_DEVICES":device},stdout=log,
                stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(launch,zip(base.PROMPTS,devices[:2])))
    aggregate()


if __name__ == "__main__":
    command=sys.argv[1] if len(sys.argv)>1 else "run"
    if command=="worker":
        base.install_snapshot(); worker(int(sys.argv[2]))
    else:
        globals()[command]()
