#!/usr/bin/env python3
"""Table-4-aligned TopK10 Spark: 8x8 Q/K refinement only in skipped tail."""

from pathlib import Path
import sys

import diffusers_spark_topk30_25prompt_20260928 as experiment


def config(method):
    from h3_sparse_attention import H3SparseAttentionConfig

    if method != "spark_block8x8":
        raise ValueError(method)
    return H3SparseAttentionConfig.spark(
        20, sol_route_topk_ratio=0.1, sol_log_density=False,
        sol_video_tail_mode="dense", sol_tail_granularity="block8x8",
        sol_global_anchor_dtype="bfloat16",
        sol_reweight_summary_math="tensorcore", sol_reweight_logmass_key="stored",
        sol_reweight_components="full",
    )


experiment.NAME = "diffusers_spark_block8x8_25prompt_10s768p_20260928"
experiment.ROOT = Path("/autodl-fs/data/h3_experiments") / experiment.NAME
experiment.OUT = Path("/autodl-fs/data/h3_outputs") / experiment.NAME
experiment.METHODS = ("spark_block8x8",)
experiment.SCRIPT_PATH = Path(__file__).resolve()
experiment.config = config
experiment.configure()


def run():
    experiment.run()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command in ("generate_worker", "decode_worker"):
        getattr(experiment.shared, command)(int(sys.argv[2]))
    elif command == "run":
        run()
    elif command == "vbench":
        experiment.vbench()
    elif command in ("prepare", "aggregate", "manifests", "quality"):
        getattr(experiment, command)()
    else:
        raise ValueError(command)
