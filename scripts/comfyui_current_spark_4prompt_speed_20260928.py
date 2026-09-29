"""Current-code four-prompt ComfyUI Sol/Spark speed check on GPU0-1."""

import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import sys

import comfyui_latest_spark_4prompt_20260926 as trial


trial.NAME = "comfyui_current_spark_4prompt_speed_20260928"
trial.ROOT = Path("/autodl-fs/data/h3_experiments") / trial.NAME
trial.OUT = Path("/autodl-fs/data/h3_outputs") / trial.NAME
trial.GPUS = (0, 1)
trial.METHODS = ("sol_tau1_extra0", "spark_block", "spark_query")
trial.DURATIONS = (5, 10)
ROOT = trial.ROOT
ARCHIVED = Path("/autodl-fs/data/h3_experiments/comfyui_latest_spark_4prompt_5s10s768p_20260926/protocol.json")
trial.source.cases = lambda: trial.read(ARCHIVED)["cases"]
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"


def worker(gpu):
    cases = trial.cases()[gpu::2]
    port = 8430 + gpu
    for seconds in trial.DURATIONS:
        for case in cases:
            for method in trial.METHODS:
                warmup = ROOT / f"{seconds}s/warmup/{method}/case_{case['index']:02}.safetensors"
                warmup.parent.mkdir(parents=True, exist_ok=True)
                trial.base.queue_and_wait(port, trial.graph(seconds, method, case, warmup, 5), trial.source.SAMPLER_NODE)
            for method in trial.METHODS:
                target = trial.latent(seconds, method, case)
                record_path = trial.record(seconds, method, case)
                if record_path.is_file() and target.is_file():
                    existing = trial.read(record_path)
                    if existing.get("latent_sha256") == trial.base.sha256(target):
                        continue
                target.parent.mkdir(parents=True, exist_ok=True)
                response = trial.base.queue_and_wait(port, trial.graph(seconds, method, case, target), trial.source.SAMPLER_NODE)
                response.pop("history", None)
                if not response.get("sampler_seconds") or response["sampler_seconds"] < 10:
                    raise RuntimeError(f"cached/missing sampler execution: {seconds}s {method} {case['index']}")
                trial.write(record_path, dict(**response, method=method, seconds=seconds,
                    case=case["index"], repeat=0, gpu=gpu, steps=20, seed=42,
                    latent_path=str(target), latent_sha256=trial.base.sha256(target)))
                print("DENOISED", seconds, method, case["index"], response["sampler_seconds"], flush=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    trial.OUT.mkdir(parents=True, exist_ok=True)
    for gpu in trial.GPUS:
        for name in ("input", "temp", "user"):
            (ROOT / name / f"gpu{gpu}").mkdir(parents=True, exist_ok=True)
    sources = [trial.REPO / "comfyui_backend.py", trial.REPO / "comfyui_nodes.py",
               trial.REPO / "comfyui_reblock_plan.py",
               trial.old.KITCHEN / "comfy_kitchen/backends/cuda/_C.abi3.so"]
    hashes = {str(path): trial.base.sha256(path) for path in sources}
    protocol_path = ROOT / "protocol.json"
    if protocol_path.exists():
        if trial.read(protocol_path)["source_hashes"] != hashes:
            raise RuntimeError("implementation changed during speed experiment")
    else:
        trial.write(protocol_path, dict(name=trial.NAME, cases=trial.cases(),
            methods=trial.METHODS, durations=trial.DURATIONS, source_hashes=hashes,
            conditioning=[dict(path=str(trial.source.condition_path(case)),
                sha256=trial.base.sha256(trial.source.condition_path(case))) for case in trial.cases()],
            settings=dict(steps=20, seed=42, resolution="1344x768", repeats=1,
                gpus=trial.GPUS, topk_ratio=.1, fanout=16, global_anchor_dtype="float32",
                video_tail_mode="dense", warmup="excluded 5-step run per method/case",
                scope="sampler speed only; no decode, PSNR, or VBench")))
    trial.write(ROOT / "scope.json", dict(status="running", gpus=trial.GPUS,
        cases=[case["index"] for case in trial.cases()], methods=trial.METHODS,
        durations=trial.DURATIONS, repetitions=1,
        scope="current ComfyUI 20-step sampler speed only; no decode or scoring"))
    servers = []
    try:
        for gpu in trial.GPUS:
            servers.append(trial.start(gpu))
        for gpu in trial.GPUS:
            trial.base.wait_server(8430 + gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, trial.GPUS))
    finally:
        trial.base.stop_servers(servers)
    summary = {}
    for seconds in trial.DURATIONS:
        rows = {method: [trial.read(trial.record(seconds, method, case))
                         for case in trial.cases()] for method in trial.METHODS}
        baseline = {row["case"]: row["sampler_seconds"] for row in rows["sol_tau1_extra0"]}
        summary[str(seconds)] = {method: dict(
            mean_seconds=statistics.fmean(row["sampler_seconds"] for row in values),
            paired_speedups=[baseline[row["case"]] / row["sampler_seconds"] for row in values],
            records=values) for method, values in rows.items()}
    trial.write(ROOT / "results.json", dict(status="complete", summary=summary))
    print(json.dumps({seconds: {method: round(values["mean_seconds"], 3)
        for method, values in methods.items()} for seconds, methods in summary.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
