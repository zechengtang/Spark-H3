"""Paired ComfyUI Sol tau=1 versus official fixed-TopK10 sparse-step profile."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import traceback

import comfyui_latest_spark_4prompt_20260926 as trial
from summarize_comfyui_pipeline_audit import summarize


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_sol_topk_sparse_step_20260928")
ARCHIVED = Path("/autodl-fs/data/h3_experiments/comfyui_latest_spark_4prompt_5s10s768p_20260926/protocol.json")
METHODS = ("sol_tau1", "sol_topk10", "spark_topk10_native")
trial.source.cases = lambda: trial.read(ARCHIVED)["cases"]
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"


def graph(case, target, method, steps=20):
    if method == "spark_topk10_native":
        value = trial.graph(10, "spark_block", case, target, steps)
        value["2"]["inputs"].update(ablation_mode="topk_only", tail_granularity="block")
        return value
    value = trial.graph(10, "sol_tau1_extra0", case, target, steps)
    if method == "sol_topk10":
        value["2"]["inputs"].update(selection="sla", **{"selection.keep_percent": 10.0})
        value["2"]["inputs"].pop("selection.tau", None)
    return value


def worker(gpu):
    port = 8480 + gpu
    base = trial.base
    root = ROOT / f"gpu{gpu}"
    for folder in ("input", "temp", "user", "output"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    log = (root / "server.log").open("a")
    cmd = [base.PYTHON, "main.py", "--listen", "127.0.0.1", "--port", str(port),
           "--disable-auto-launch", "--disable-cuda-malloc", "--preview-method", "none",
           "--cache-classic"]
    for folder in ("input", "temp", "user", "output"):
        cmd.extend([f"--{folder}-directory", str(root / folder)])
    proc = subprocess.Popen(
        cmd, cwd=trial.old.COMFY, stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "H3_PIPELINE_AUDIT": "1",
             "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"},
    )
    base.write_json(root / "pid.json", {"pid": proc.pid, "port": port})
    try:
        base.wait_server(port, timeout=600)
        case = trial.cases()[0]
        methods = METHODS if gpu == 0 else tuple(reversed(METHODS))
        for method in methods:
            dest = root / method
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "ranges.json").exists():
                continue
            base.api_json(port, "/h3_audit/reset?enabled=0")
            base.queue_and_wait(port, graph(case, dest / "warmup.safetensors", method, 5), "11")
            base.api_json(port, "/h3_audit/reset?enabled=1")
            result = base.queue_and_wait(port, graph(case, dest / "measured.safetensors", method), "11")
            result.pop("history", None)
            if not result.get("sampler_seconds") or result["sampler_seconds"] < 10:
                raise RuntimeError(f"cached or missing sampler: {method}")
            audit = base.api_json(port, "/h3_audit/flush")
            if audit["evaluations"] != 20:
                raise RuntimeError(f"expected 20 model evaluations, got {audit['evaluations']}")
            base.write_json(dest / "timing.json", result)
            base.write_json(dest / "ranges.json", audit)
            print("DONE", gpu, method, result["sampler_seconds"], flush=True)
    finally:
        base.stop_servers([(str(gpu), proc, log)])


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    base = trial.base
    source = trial.old.KITCHEN / "comfy_kitchen/backends/cuda/_C.abi3.so"
    base.write_json(ROOT / "protocol.json", {
        "methods": METHODS, "gpus": [0, 1], "case": trial.cases()[0],
        "settings": {"seconds": 10, "resolution": [1344, 768], "steps": 20,
                     "seed": 42, "dense_evaluations": 4, "sparse_evaluations": 16,
                     "topk_ratio": 0.1, "sol_tau": 1.0, "extra_tokens": 0,
                     "warmup": "excluded 5-step denoise per method per GPU"},
        "source_sha256": base.sha256(source),
    })
    base.write_json(ROOT / "status.json", {"status": "running", "pid": os.getpid()})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, (0, 1)))
        paths = sorted(ROOT.glob("gpu*/*/ranges.json"))
        for path in paths:
            timing = trial.read(path.parent / "timing.json")
            base.write_json(path.parent / "profiled_timing.json", {
                **timing, "gpu": int(path.parts[-3].removeprefix("gpu")),
                "seconds": 10, "method": path.parent.name,
                "case": trial.cases()[0]["index"],
            })
        records = [summarize(path) for path in paths]
        base.write_json(ROOT / "summary.json", {"status": "complete", "records": records})
        base.write_json(ROOT / "status.json", {"status": "complete"})
    except Exception:
        base.write_json(ROOT / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
