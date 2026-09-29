#!/usr/bin/env python3
"""One-prompt full-path profile for the current native ComfyUI Spark producer."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import time

import comfyui_sol_4way_benchmark_20260923 as base
import comfyui_spark_sol_10prompt_benchmark_20260924 as source


NAME = "comfyui_spark_fullpath_profile_1prompt_10s768p_20260925"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
PORT = 8430
PROFILE_RE = re.compile(r"reblock CUDA profile: (\{.*\})")


def graph(case, target, steps):
    value = source.denoise_graph("spark_topk10", case, target, steps)
    value["2"]["inputs"].update(
        ablation_mode="full",
        video_tail_mode="dense",
        global_anchor_dtype="float32",
    )
    value["4"]["inputs"].update(width=1344, height=768, length=240)
    return value


def profile_from_log(path: Path):
    deadline = time.time() + 90
    while time.time() < deadline:
        matches = PROFILE_RE.findall(path.read_text(errors="replace")) if path.exists() else []
        if matches:
            return ast.literal_eval(matches[-1])
        time.sleep(0.25)
    raise TimeoutError("Spark profile was not emitted")


def main():
    if (ROOT / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite completed result in {ROOT}")
    for path in (
        ROOT / "user" / "gpu0",
        ROOT / "temp" / "gpu0",
        ROOT / "input" / "gpu0",
        ROOT / "latents",
        OUT,
    ):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["SPARK_PROFILE_REBLOCK"] = "1"
    base.ROOT, base.OUT = ROOT, OUT
    case = source.cases()[0]
    server = None
    try:
        process, log = base.start_server("gpu0", 0, PORT, OUT)
        server = [("gpu0", process, log)]
        base.wait_server(PORT)
        warm = ROOT / "latents" / "warmup.safetensors"
        base.queue_and_wait(PORT, graph(case, warm, 5), source.SAMPLER_NODE)
        target = ROOT / "latents" / "measured.safetensors"
        result = base.queue_and_wait(PORT, graph(case, target, 20), source.SAMPLER_NODE)
        profile = profile_from_log(ROOT / "server_gpu0.log")
        base.write_json(ROOT / "result.json", {
            "status": "complete",
            "case": case,
            "sampler_seconds": result["sampler_seconds"],
            "profile": profile,
            "latent_path": str(target),
            "latent_sha256": base.sha256(target),
        })
        print(json.dumps({"sampler_seconds": result["sampler_seconds"], "profile": profile}, indent=2))
    finally:
        if server is not None:
            base.stop_servers(server)


if __name__ == "__main__":
    main()
