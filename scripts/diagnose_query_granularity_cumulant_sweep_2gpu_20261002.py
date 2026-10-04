#!/usr/bin/env python3
"""Follow-up Q-granularity and diagonal-cumulant sweep on frozen Dense Q/K/V."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
SOURCE = REPO / "scripts/diagnose_reblock_token_oracle_candidates_2gpu_20261001.py"
SPEC = importlib.util.spec_from_file_location("token_oracle_study", SOURCE)
study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(study)

NAME = "query_granularity_cumulant_sweep_2prompt_20261002"
METHODS = (
    "native_mean",
    "q32_k32_normalized_mass",
    "q16_k64_normalized_mass",
    "q8_k64_normalized_mass",
    "q4_k64_normalized_mass",
    "qtoken_k64_normalized_mass",
    "diag_cumulant_lambda100",
    "diag_cumulant_lambda150",
    "diag_cumulant_lambda200",
    "diag_cumulant_lambda300",
    "diag_cumulant_lambda400",
    "exact_lme_mean_query",
)
PAIR_MULTIPLIER = {
    "native_mean": 1,
    "q32_k32_normalized_mass": 4,
    "q16_k64_normalized_mass": 4,
    "q8_k64_normalized_mass": 8,
    "q4_k64_normalized_mass": 16,
    "qtoken_k64_normalized_mass": 64,
    "diag_cumulant_lambda100": 4,
    "diag_cumulant_lambda150": 4,
    "diag_cumulant_lambda200": 4,
    "diag_cumulant_lambda300": 4,
    "diag_cumulant_lambda400": 4,
    "exact_lme_mean_query": 64,
}
original_candidate_scores = study.candidate_scores


def candidate_scores(qblock: torch.Tensor, kblocks: torch.Tensor) -> dict:
    scores = original_candidate_scores(qblock, kblocks)
    dim = qblock.shape[-1]
    scale = dim ** -0.5
    kmean = kblocks.mean(1)
    for query_rows in (8, 4):
        queries = qblock.reshape(64 // query_rows, query_rows, dim).mean(1)
        probabilities = (queries @ kmean.T * scale).softmax(-1)
        scores[f"q{query_rows}_k64_normalized_mass"] = probabilities.mean(0)

    qmean = qblock.mean(0)
    qvar = qblock.var(dim=0, correction=0)
    kvar = kblocks.var(dim=1, correction=0)
    variance = (
        kvar @ qmean.square()
        + kmean.square() @ qvar
        + kvar @ qvar
    ) / dim
    variance = variance.clamp_min(0)
    native = scores["native_mean"]
    for suffix, value in (("150", 1.5), ("200", 2.0), ("300", 3.0), ("400", 4.0)):
        scores[f"diag_cumulant_lambda{suffix}"] = native + 0.5 * value * variance
    return scores


def configure() -> None:
    study.HERE = HERE
    study.NAME = NAME
    study.ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
    study.REPORT = REPO / "reports" / NAME
    study.METHODS = METHODS
    study.SCORE_PAIR_MULTIPLIER = PAIR_MULTIPLIER
    study.candidate_scores = candidate_scores


if __name__ == "__main__":
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "describe"
    if command == "worker":
        study.worker(int(sys.argv[2]))
    else:
        getattr(study, command)()
