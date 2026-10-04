#!/usr/bin/env python3
"""Real-Q/K attention-mass recall for adjacent-subblock route variants."""

from __future__ import annotations

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
spec = importlib.util.spec_from_file_location("subblock_recall_base", SOURCE)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

NAME = os.environ.get(
    "H3_SUBBLOCK_RECALL_NAME", "subblock_route_recall_2prompt_20260930"
)
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
REPORT = REPO / "reports" / NAME
SNAP = Path(
    "/autodl-fs/data/h3_experiments/"
    "subblock_route_table34_50prompt_20260930/snapshot"
)
METHODS = (
    "native_q64_k64_mean",
    "q64_k32_lme",
    "q32_k64_lme",
    "q32_k32_lme",
    "q32_k32_max",
    "q32_k32_normmass",
)


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _install():
    base.SNAP = SNAP
    base.install_snapshot()


def _scores(qblocks, kblocks):
    """Return HxQxK natural-log route scores from BF16 stored means."""
    scale = base.HEAD_DIM ** -0.5
    q64 = qblocks.mean(2).to(torch.bfloat16)
    k64 = kblocks.mean(2).to(torch.bfloat16)
    q32 = qblocks.reshape(*qblocks.shape[:2], 2, 32, base.HEAD_DIM).mean(3).to(torch.bfloat16)
    k32 = kblocks.reshape(*kblocks.shape[:2], 2, 32, base.HEAD_DIM).mean(3).to(torch.bfloat16)
    native = torch.einsum("hqd,hkd->hqk", q64, k64).float() * scale
    q64k32 = torch.einsum("hqd,hkcd->hqkc", q64, k32).float() * scale
    q32k64 = torch.einsum("hqcd,hkd->hqkc", q32, k64).float() * scale
    q32k32 = torch.einsum("hqad,hkbd->hqkab", q32, k32).float() * scale
    log_key_mass = torch.logsumexp(q32k32, -1)
    log_probability = log_key_mass - torch.logsumexp(
        log_key_mass, dim=2, keepdim=True
    )
    return {
        "native_q64_k64_mean": native,
        "q64_k32_lme": torch.logsumexp(q64k32, -1) - math.log(2),
        "q32_k64_lme": torch.logsumexp(q32k64, -1) - math.log(2),
        "q32_k32_lme": torch.logsumexp(q32k32, (-1, -2)) - math.log(4),
        "q32_k32_max": q32k32.amax((-1, -2)),
        "q32_k32_normmass": torch.logsumexp(log_probability, -1) - math.log(2),
    }


@torch.inference_mode()
def process_unit(cap, layer, cfg, prompt, evaluation):
    record = cap["records"][layer]
    q0, k0 = [record[name].cuda() for name in ("q", "k")]
    heads, tokens, dim = q0.shape
    assert heads == 4 and dim == base.HEAD_DIM and tokens == cap["video_tokens"]
    qp, kp, partition = base.production_partition(q0, k0, cap["grid"], cfg)
    q = q0.gather(1, qp[..., None].expand(-1, -1, dim)).float()
    k = k0.gather(1, kp[..., None].expand(-1, -1, dim)).float()
    complete = tokens // base.BLOCK
    cut = complete * base.BLOCK
    quota = max(1, round(base.TOPK_RATIO * complete))
    qblocks = q[:, :cut].reshape(heads, complete, base.BLOCK, dim)
    kblocks = k[:, :cut].reshape(heads, complete, base.BLOCK, dim)
    scores = _scores(qblocks, kblocks)
    sampled = torch.linspace(0, complete - 1, base.SAMPLED_QUERY_BLOCKS,
                             device="cuda").round().long().unique().tolist()
    rows = []
    for head in range(heads):
        keys = kblocks[head].reshape(cut, dim)
        for query_block in sampled:
            query = qblocks[head, query_block]
            logits = query @ keys.T * (dim ** -0.5)
            probability = logits.softmax(-1).reshape(base.BLOCK, complete, base.BLOCK)
            mass = probability.sum(-1).mean(0)
            oracle = mass.topk(quota).indices
            oracle_mass = mass[oracle].sum()
            methods = {}
            for method in METHODS:
                score = scores[method][head, query_block]
                selected = score.topk(quota).indices
                selected_mask = torch.zeros_like(mass, dtype=torch.bool)
                selected_mask[selected] = True
                methods[method] = dict(
                    topk_recall=float(selected_mask[oracle].float().mean()),
                    mass_efficiency=float(mass[selected].sum() / oracle_mass),
                    selected_attention_mass=float(mass[selected].sum()),
                    spearman=base.spearman(score, mass),
                )
            rows.append(dict(
                prompt=prompt, evaluation=evaluation, layer=layer,
                head=int(cap["heads"][head]), query_block=query_block,
                candidate_blocks=complete, quota=quota, methods=methods,
            ))
    return rows, dict(prompt=prompt, evaluation=evaluation, layer=layer,
                      complete_blocks=complete, quota=quota, partition=partition)


def worker(prompt):
    _install()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = base.config()
    rows, units = [], []
    started = time.perf_counter()
    capture_index = json.loads((base.CAPTURES / f"capture_{prompt:02}.json").read_text())
    by_eval = {item["evaluation"]: item for item in capture_index["files"]}
    for evaluation in base.EVALUATIONS:
        item = by_eval[evaluation]
        if base.sha(item["path"]) != item["sha256"]:
            raise RuntimeError(f"capture hash mismatch: {item['path']}")
        cap = torch.load(item["path"], map_location="cpu", mmap=True, weights_only=False)
        for layer in base.LAYERS:
            unit_rows, unit = process_unit(cap, layer, cfg, prompt, evaluation)
            rows.extend(unit_rows); units.append(unit)
            print("UNIT", prompt, evaluation, layer, len(rows), flush=True)
            torch.cuda.empty_cache()
        del cap
    write(ROOT / f"worker_{prompt}.json", dict(
        status="complete", prompt=prompt, elapsed_seconds=time.perf_counter()-started,
        rows=rows, units=units,
    ))


def aggregate():
    docs = [json.loads((ROOT / f"worker_{p}.json").read_text()) for p in base.PROMPTS]
    rows = [row for doc in docs for row in doc["rows"]]
    summary = {
        method: {
            metric: statistics.mean(row["methods"][method][metric] for row in rows)
            for metric in ("topk_recall", "mass_efficiency",
                           "selected_attention_mass", "spearman")
        }
        for method in METHODS
    }
    result = dict(
        status="complete", summary=summary, rows=rows,
        scope=dict(prompts=list(base.PROMPTS), evaluations=list(base.EVALUATIONS),
                   layers=list(base.LAYERS), heads=[0, 1, 2, 3],
                   sampled_query_blocks_per_head=base.SAMPLED_QUERY_BLOCKS,
                   topk_ratio=base.TOPK_RATIO, trajectory="Dense-reference",
                   reblock="Table3/4 fanout16 flat midpoint32"),
        elapsed_gpu_seconds=sum(doc["elapsed_seconds"] for doc in docs),
    )
    write(ROOT / "results.json", result)
    write(REPORT / "results.json", result)
    lines = [
        "# Adjacent-subblock route attention-mass recall", "",
        "| Route | TopK recall | Mass efficiency | Selected mass | Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary[method]
        lines.append(f"| {method} | {item['topk_recall']:.4f} | "
                     f"{item['mass_efficiency']:.4f} | "
                     f"{item['selected_attention_mass']:.4f} | "
                     f"{item['spearman']:.4f} |")
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def prepare():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    (ROOT / "logs").mkdir(parents=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    write(ROOT / "protocol.json", dict(
        status="prepared", snapshot=str(SNAP), prompts=list(base.PROMPTS),
        evaluations=list(base.EVALUATIONS), layers=list(base.LAYERS),
        methods=list(METHODS), topk_ratio=base.TOPK_RATIO,
    ))


def run():
    jobs = []
    for gpu, prompt in enumerate(base.PROMPTS):
        log = (ROOT / "logs" / f"worker_{prompt}.log").open("a")
        process = subprocess.Popen(
            [sys.executable, str(HERE), "worker", str(prompt)],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((process, log))
    codes = [process.wait() for process, _ in jobs]
    for _, log in jobs: log.close()
    if any(codes): raise RuntimeError(codes)
    aggregate()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "worker": worker(int(sys.argv[2]))
    else: globals()[command]()
