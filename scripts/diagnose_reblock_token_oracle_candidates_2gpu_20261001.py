#!/usr/bin/env python3
"""Two-GPU fixed-QKV study of Spark routing and progressively looser oracles.

This script is deliberately a diagnostic, not a production attention path.  It
replays the same frozen fanout-16 reblocking used by the Table 3/4 experiments
and separates three often-confused upper bounds at a fixed 10% budget:

1. ``shared_k64_oracle``: one K64 mask shared by all 64 rows of a Q64 block;
   this is the attainable upper bound for the current execution interface.
2. ``row_k64_oracle``: a different set of K64 blocks for every query row;
   token-pair work is unchanged, but the current kernel cannot express it.
3. ``row_token_oracle``: keep ``K_blocks * 64`` individual K tokens per query
   row (equivalently, mask approximately the bottom 90%); this exactly matches
   the current rounded block-route token-pair budget but ignores K64 execution
   granularity.

It also screens route scores which still emit a Q64-by-K64 mask.  No model is
loaded and no generations are written.  Invoke ``run`` only after GPUs are
available; ``describe`` is CPU-only and safe while another benchmark is active.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
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
BASE_PATH = REPO / "scripts/diagnose_global_anchor_route_oracles_20260930.py"
BASE_SPEC = importlib.util.spec_from_file_location("reblock_oracle_base", BASE_PATH)
base = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(base)

NAME = "reblock_token_oracle_candidates_2prompt_20261001"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
REPORT = REPO / "reports" / NAME
METHODS = (
    "native_mean",
    "q32_k32_normalized_mass",
    "q16_k64_normalized_mass",
    "qtoken_k64_normalized_mass",
    "diag_cumulant_lambda025",
    "diag_cumulant_lambda050",
    "diag_cumulant_lambda100",
    "exact_lme_mean_query",
)
ORACLES = (
    "shared_k64_oracle",
    "row_k64_oracle",
    "row_token_oracle",
)
SCORE_PAIR_MULTIPLIER = {
    "native_mean": 1,
    "q32_k32_normalized_mass": 4,
    "q16_k64_normalized_mass": 4,
    "qtoken_k64_normalized_mass": 64,
    # Mean plus three factorized diagonal-variance products.
    "diag_cumulant_lambda025": 4,
    "diag_cumulant_lambda050": 4,
    "diag_cumulant_lambda100": 4,
    # One mean query against all 64 tokens of every K64 parent.
    "exact_lme_mean_query": 64,
}


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe() -> None:
    payload = {
        "name": NAME,
        "gpu_work_started": False,
        "captures": str(base.CAPTURES),
        "prompts": list(base.PROMPTS),
        "evaluations": list(base.EVALUATIONS),
        "layers": list(base.LAYERS),
        "sampled_query_blocks_per_head": base.SAMPLED_QUERY_BLOCKS,
        "topk_ratio": base.TOPK_RATIO,
        "methods": list(METHODS),
        "oracles": list(ORACLES),
        "score_pair_multiplier_vs_native_mean": SCORE_PAIR_MULTIPLIER,
        "row_token_oracle_definition": (
            "per query row, keep round(0.10 * complete_K64_blocks) * 64 keys "
            "with the largest dense logits; mask approximately the bottom 90%; "
            "this matches the rounded production block budget; context and the "
            "incomplete 48-token video tail are absent from the frozen capture"
        ),
    }
    print(json.dumps(payload, indent=2))


def ranks(value: torch.Tensor) -> torch.Tensor:
    order = value.argsort()
    result = torch.empty_like(order, dtype=torch.float32)
    result[order] = torch.arange(value.numel(), device=value.device, dtype=torch.float32)
    return result


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    x, y = ranks(left), ranks(right)
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum() / denominator.clamp_min(1e-20))


def output_metrics(output: torch.Tensor, reference: torch.Tensor) -> dict:
    residual = output - reference
    relative = residual.norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-20)
    cosine = torch.nn.functional.cosine_similarity(output, reference, dim=-1)
    return {
        "output_relative_l2": float(relative.mean()),
        "output_cosine": float(cosine.mean()),
    }


def masked_output(
    probabilities: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return renormalized exact output and retained mass for a bool QxK mask."""
    retained = (probabilities * mask).sum(dim=-1)
    numerator = (probabilities * mask) @ values
    return numerator / retained[:, None].clamp_min(1e-20), retained


def candidate_scores(qblock: torch.Tensor, kblocks: torch.Tensor) -> dict:
    """Build Q64-to-physical-K64 scores without changing execution granularity."""
    block, dim = qblock.shape
    candidates = kblocks.shape[0]
    assert block == base.BLOCK == 64 and kblocks.shape[1:] == (64, dim)
    scale = dim ** -0.5
    qmean = qblock.mean(0)
    kmean = kblocks.mean(1)
    native = (kmean @ qmean) * scale

    q32 = qblock.reshape(2, 32, dim).mean(1)
    k32 = kblocks.reshape(candidates, 2, 32, dim).mean(2)
    child = torch.einsum("ad,kbd->kab", q32, k32) * scale
    log_k_mass = torch.logsumexp(child, dim=-1)
    q32_probability = torch.softmax(log_k_mass.transpose(0, 1), dim=-1)
    q32k32 = q32_probability.mean(0)

    q16 = qblock.reshape(4, 16, dim).mean(1)
    q16_logits = q16 @ kmean.T * scale
    q16k64 = q16_logits.softmax(-1).mean(0)

    qtoken_logits = qblock @ kmean.T * scale
    qtokenk64 = qtoken_logits.softmax(-1).mean(0)

    qvar = qblock.var(dim=0, correction=0)
    kvar = kblocks.var(dim=1, correction=0)
    # Diagonal approximation of Var(q^T k / sqrt(d)) for independent tokens:
    # mu_q^T C_k mu_q + mu_k^T C_q mu_k + tr(C_q C_k), divided by d.
    variance = (
        kvar @ qmean.square()
        + kmean.square() @ qvar
        + kvar @ qvar
    ) / dim
    variance = variance.clamp_min(0)

    exact_lme = torch.logsumexp(qmean @ kblocks.transpose(1, 2) * scale, dim=-1)
    exact_lme = exact_lme - math.log(base.BLOCK)
    return {
        "native_mean": native,
        "q32_k32_normalized_mass": q32k32,
        "q16_k64_normalized_mass": q16k64,
        "qtoken_k64_normalized_mass": qtokenk64,
        "diag_cumulant_lambda025": native + 0.5 * 0.25 * variance,
        "diag_cumulant_lambda050": native + 0.5 * 0.50 * variance,
        "diag_cumulant_lambda100": native + 0.5 * variance,
        "exact_lme_mean_query": exact_lme,
    }


@torch.inference_mode()
def process_unit(cap, layer: int, cfg, prompt: int, evaluation: int):
    record = cap["records"][layer]
    q0, k0, v0 = [record[name].cuda(non_blocking=False) for name in ("q", "k", "v")]
    heads, tokens, dim = q0.shape
    assert q0.shape == k0.shape == v0.shape
    assert heads == 4 and dim == base.HEAD_DIM and tokens == cap["video_tokens"]
    qp, kp, partition = base.production_partition(q0, k0, cap["grid"], cfg)
    q = q0.gather(1, qp[..., None].expand(-1, -1, dim)).float()
    k = k0.gather(1, kp[..., None].expand(-1, -1, dim)).float()
    v = v0.gather(1, kp[..., None].expand(-1, -1, dim)).float()

    complete = tokens // base.BLOCK
    cut = complete * base.BLOCK
    block_quota = max(1, round(base.TOPK_RATIO * complete))
    # Match the production route's rounded K64 budget exactly.  Using ceil(10%
    # of tokens) would quietly give the token oracle a few extra token pairs.
    token_quota = block_quota * base.BLOCK
    qblocks = q[:, :cut].reshape(heads, complete, base.BLOCK, dim)
    kblocks = k[:, :cut].reshape(heads, complete, base.BLOCK, dim)
    sampled = torch.linspace(
        0, complete - 1, base.SAMPLED_QUERY_BLOCKS, device="cuda"
    ).round().long().unique().tolist()
    assert len(sampled) == base.SAMPLED_QUERY_BLOCKS

    rows = []
    for head in range(heads):
        keys = kblocks[head].reshape(cut, dim)
        values = v[head, :cut]
        for query_block in sampled:
            query = qblocks[head, query_block]
            logits = query @ keys.T * (dim ** -0.5)
            probability = logits.softmax(-1)
            dense_output = probability @ values
            row_block_mass = probability.reshape(base.BLOCK, complete, base.BLOCK).sum(-1)
            shared_mass = row_block_mass.mean(0)
            shared_ids = shared_mass.topk(block_quota).indices

            row_block_ids = row_block_mass.topk(block_quota, dim=-1).indices
            row_block_mask = torch.zeros_like(row_block_mass, dtype=torch.bool)
            row_block_mask.scatter_(1, row_block_ids, True)
            row_block_token_mask = row_block_mask[:, :, None].expand(
                -1, -1, base.BLOCK
            ).reshape(base.BLOCK, cut)

            row_token_ids = logits.topk(token_quota, dim=-1).indices
            row_token_mask = torch.zeros_like(logits, dtype=torch.bool)
            row_token_mask.scatter_(1, row_token_ids, True)
            row_token_touched = torch.zeros_like(row_block_mask)
            row_token_touched.scatter_(1, row_token_ids // base.BLOCK, True)
            q64_union_blocks = row_token_touched.any(0).float().mean()

            oracle_masks = {}
            shared_token_mask = torch.zeros_like(logits, dtype=torch.bool)
            shared_token_mask.view(base.BLOCK, complete, base.BLOCK)[:, shared_ids] = True
            oracle_masks["shared_k64_oracle"] = shared_token_mask
            oracle_masks["row_k64_oracle"] = row_block_token_mask
            oracle_masks["row_token_oracle"] = row_token_mask
            oracle_metrics = {}
            for oracle, mask in oracle_masks.items():
                output, retained = masked_output(probability, values, mask)
                oracle_metrics[oracle] = {
                    "retained_attention_mass": float(retained.mean()),
                    **output_metrics(output, dense_output),
                }
            oracle_metrics["row_token_oracle"].update(
                mean_k64_blocks_touched_per_row=float(row_token_touched.float().sum(-1).mean()),
                mean_k64_fraction_touched_per_row=float(row_token_touched.float().mean()),
                q64_union_k64_fraction=float(q64_union_blocks),
            )

            scores = candidate_scores(query, kblocks[head])
            methods = {}
            for method, score in scores.items():
                selected = score.topk(block_quota).indices
                selected_mass = shared_mass[selected].sum()
                mask = torch.zeros_like(shared_mass, dtype=torch.bool)
                mask[selected] = True
                token_mask = mask[:, None].expand(-1, base.BLOCK).reshape(1, cut)
                token_mask = token_mask.expand(base.BLOCK, -1)
                output, retained = masked_output(probability, values, token_mask)
                methods[method] = {
                    "shared_oracle_block_recall": float(mask[shared_ids].float().mean()),
                    "shared_oracle_mass_efficiency": float(
                        selected_mass / shared_mass[shared_ids].sum().clamp_min(1e-20)
                    ),
                    "selected_attention_mass": float(selected_mass),
                    "mass_spearman": spearman(score, shared_mass),
                    "native_overlap": None,
                    **output_metrics(output, dense_output),
                    "mean_row_retained_attention_mass": float(retained.mean()),
                }
            native_ids = scores["native_mean"].topk(block_quota).indices
            for method, score in scores.items():
                selected = score.topk(block_quota).indices
                methods[method]["native_overlap"] = float(
                    torch.isin(selected, native_ids).float().mean()
                )

            rows.append({
                "prompt": prompt,
                "evaluation": evaluation,
                "layer": layer,
                "head": int(cap["heads"][head]),
                "query_block": query_block,
                "candidate_blocks": complete,
                "candidate_tokens": cut,
                "block_quota": block_quota,
                "token_quota": token_quota,
                "methods": methods,
                "oracles": oracle_metrics,
            })
    return rows, {
        "prompt": prompt,
        "evaluation": evaluation,
        "layer": layer,
        "video_tokens": tokens,
        "complete_blocks": complete,
        "tail_tokens_excluded": tokens - cut,
        "block_quota": block_quota,
        "token_quota": token_quota,
        "partition": partition,
    }


def worker(prompt: int) -> None:
    base.install_snapshot()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = base.config()
    rows, units = [], []
    started = time.perf_counter()
    index = json.loads((base.CAPTURES / f"capture_{prompt:02}.json").read_text())
    by_evaluation = {item["evaluation"]: item for item in index["files"]}
    for evaluation in base.EVALUATIONS:
        item = by_evaluation[evaluation]
        if sha(item["path"]) != item["sha256"]:
            raise RuntimeError(f"capture hash mismatch: {item['path']}")
        cap = torch.load(item["path"], map_location="cpu", mmap=True, weights_only=False)
        assert cap["trajectory"] == "dense_reference" and cap["seed"] == 42
        for layer in base.LAYERS:
            unit_rows, unit = process_unit(cap, layer, cfg, prompt, evaluation)
            rows.extend(unit_rows)
            units.append(unit)
            print("UNIT", prompt, evaluation, layer, len(rows), flush=True)
            torch.cuda.empty_cache()
        del cap
    write(ROOT / f"worker_{prompt}.json", {
        "status": "complete",
        "prompt": prompt,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "units": units,
    })


def mean(values) -> float:
    return statistics.mean(values)


def aggregate() -> None:
    documents = [json.loads((ROOT / f"worker_{p}.json").read_text()) for p in base.PROMPTS]
    rows = [row for document in documents for row in document["rows"]]
    expected = (
        len(base.PROMPTS) * len(base.EVALUATIONS) * len(base.LAYERS)
        * 4 * base.SAMPLED_QUERY_BLOCKS
    )
    assert len(rows) == expected
    method_metrics = (
        "shared_oracle_block_recall",
        "shared_oracle_mass_efficiency",
        "selected_attention_mass",
        "mass_spearman",
        "native_overlap",
        "mean_row_retained_attention_mass",
        "output_relative_l2",
        "output_cosine",
    )
    oracle_metrics = (
        "retained_attention_mass", "output_relative_l2", "output_cosine"
    )
    summary = {
        method: {
            metric: mean([row["methods"][method][metric] for row in rows])
            for metric in method_metrics
        }
        for method in METHODS
    }
    oracle_summary = {
        oracle: {
            metric: mean([row["oracles"][oracle][metric] for row in rows])
            for metric in oracle_metrics
        }
        for oracle in ORACLES
    }
    oracle_summary["row_token_oracle"].update({
        metric: mean([row["oracles"]["row_token_oracle"][metric] for row in rows])
        for metric in (
            "mean_k64_blocks_touched_per_row",
            "mean_k64_fraction_touched_per_row",
            "q64_union_k64_fraction",
        )
    })
    result = {
        "status": "complete",
        "summary": summary,
        "oracle_summary": oracle_summary,
        "score_pair_multiplier_vs_native_mean": SCORE_PAIR_MULTIPLIER,
        "rows": rows,
        "units": [unit for document in documents for unit in document["units"]],
        "scope": {
            "prompts": list(base.PROMPTS),
            "evaluations": list(base.EVALUATIONS),
            "layers": list(base.LAYERS),
            "heads": [0, 1, 2, 3],
            "sampled_query_blocks_per_head": base.SAMPLED_QUERY_BLOCKS,
            "trajectory": "synchronized Dense-reference",
            "reblock": "frozen Table3/4 fanout16 flat midpoint32",
            "normalization": "video-only because captures omit always-dense context",
            "top90_interpretation": (
                "mask approximately the bottom 90% per query row; retain "
                "block_quota*64 tokens so token-pair work matches production"
            ),
            "hard_mask_warning": (
                "output errors renormalize exact selected values and do not include "
                "Spark reweight approximation for skipped blocks"
            ),
        },
        "elapsed_gpu_seconds": sum(document["elapsed_seconds"] for document in documents),
    }
    write(ROOT / "results.json", result)
    write(REPORT / "results.json", result)

    lines = [
        "# Reblock route and token-oracle diagnostic", "",
        "Frozen Dense Q/K/V, current fanout-16 reblocking, and a 10% exact budget. "
        "All candidate routes still emit one K64 mask per Q64 parent.", "",
        "## Route candidates", "",
        "| Route | Score pairs vs mean | Block recall | Mass efficiency | Selected mass | Hard-mask rel-L2 |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary[method]
        lines.append(
            f"| {method} | {SCORE_PAIR_MULTIPLIER[method]}x | "
            f"{item['shared_oracle_block_recall']:.4f} | "
            f"{item['shared_oracle_mass_efficiency']:.4f} | "
            f"{item['selected_attention_mass']:.4f} | "
            f"{item['output_relative_l2']:.4f} |"
        )
    lines += ["", "## Oracle ladder", "",
              "| Oracle | Retained mass | Hard-mask rel-L2 | Output cosine |",
              "|---|---:|---:|---:|"]
    for oracle in ORACLES:
        item = oracle_summary[oracle]
        lines.append(
            f"| {oracle} | {item['retained_attention_mass']:.4f} | "
            f"{item['output_relative_l2']:.4f} | {item['output_cosine']:.4f} |"
        )
    token = oracle_summary["row_token_oracle"]
    lines += ["", "The row-token oracle means: mask the bottom 90% and retain the top "
              "10% video K tokens independently for every query row, using exactly the "
              "same rounded token-pair budget as the K64 route. It is not directly "
              "executable by the current K64 kernel.", "",
              f"It touches {token['mean_k64_fraction_touched_per_row']:.2%} of physical "
              f"K64 parents per row and {token['q64_union_k64_fraction']:.2%} after "
              "unioning the 64 rows of a Q64 parent."]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": summary, "oracle_summary": oracle_summary}, indent=2))


def prepare() -> None:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    ROOT.mkdir(parents=True)
    (ROOT / "logs").mkdir()
    REPORT.mkdir(parents=True, exist_ok=True)
    base.install_snapshot()
    manifest = {
        "status": "prepared",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(HERE),
        "script_sha256": sha(HERE),
        "base_script": str(BASE_PATH),
        "base_script_sha256": sha(BASE_PATH),
        "snapshot": str(base.SNAP),
        "capture_root": str(base.CAPTURES),
        "capture_manifest_sha256": sha(base.CAPTURES / "manifest.json"),
        "methods": list(METHODS),
        "oracles": list(ORACLES),
        "score_pair_multiplier_vs_native_mean": SCORE_PAIR_MULTIPLIER,
        "gpu_work_started": False,
    }
    write(ROOT / "manifest.json", manifest)
    write(REPORT / "manifest.json", manifest)


def run() -> None:
    prepare()
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(devices) < 2:
        raise RuntimeError("diagnostic requires two visible GPUs")

    def launch(item):
        prompt, device = item
        with (ROOT / "logs" / f"worker_{prompt}.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(HERE), "worker", str(prompt)],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(launch, zip(base.PROMPTS, devices[:2])))
    aggregate()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "describe"
    if command == "worker":
        worker(int(sys.argv[2]))
    else:
        globals()[command]()
