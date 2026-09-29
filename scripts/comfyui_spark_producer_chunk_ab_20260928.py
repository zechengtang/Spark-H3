"""One-prompt 10s Spark A/B for chunked QKV producer size on GPU0-1."""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import comfyui_latest_spark_4prompt_20260926 as trial
from summarize_comfyui_pipeline_audit import summarize


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_spark_producer_chunk_ab_20260928")
ARCHIVED = Path("/autodl-fs/data/h3_experiments/comfyui_latest_spark_4prompt_5s10s768p_20260926/protocol.json")
CASE = trial.read(ARCHIVED)["cases"][0]
METHOD = "spark_block"
CHUNKS = {0: 16384, 1: 32768}
trial.ROOT = ROOT
trial.OUT = Path("/autodl-fs/data/h3_outputs") / ROOT.name
trial.source.cases = lambda: trial.read(ARCHIVED)["cases"]
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"


def measure(gpu):
    port = 8430 + gpu
    folder = ROOT / f"gpu{gpu}"
    warmup = folder / "warmup.safetensors"
    trial.base.api_json(port, "/h3_audit/reset?enabled=0")
    trial.base.queue_and_wait(port, trial.graph(10, METHOD, CASE, warmup, 5), trial.source.SAMPLER_NODE)
    trial.base.api_json(port, "/h3_audit/reset?enabled=1")
    target = folder / "measured.safetensors"
    result = trial.base.queue_and_wait(port, trial.graph(10, METHOD, CASE, target), trial.source.SAMPLER_NODE)
    result.pop("history", None)
    if not result.get("sampler_seconds") or result["sampler_seconds"] < 10:
        raise RuntimeError(f"missing 20-step sampler execution on GPU{gpu}")
    trial.write(folder / "profiled_timing.json", dict(
        **result, gpu=gpu, seconds=10, method=METHOD, case=CASE["index"]))
    audit = trial.base.api_json(port, "/h3_audit/flush")
    if audit["evaluations"] != 20:
        raise RuntimeError(f"expected 20 evaluations on GPU{gpu}")
    trial.write(folder / "ranges.json", audit)
    print("DONE", gpu, CHUNKS[gpu], result["sampler_seconds"], flush=True)


def main():
    trial.OUT.mkdir(parents=True, exist_ok=True)
    for gpu in CHUNKS:
        for name in ("input", "temp", "user"):
            (ROOT / name / f"gpu{gpu}").mkdir(parents=True, exist_ok=True)
    trial.write(ROOT / "protocol.json", dict(
        status="running", case=CASE, duration=10, method=METHOD,
        chunks=CHUNKS, seed=42, steps=20,
        warmup="excluded 5-step run per GPU; one measured 20-step run",
        scope="sampler and nested CUDA-event profile; no decode"))
    servers = []
    try:
        for gpu, chunk in CHUNKS.items():
            os.environ["H3_SPARK_PRODUCER_CHUNK"] = str(chunk)
            os.environ["H3_PIPELINE_AUDIT"] = "1"
            servers.append(trial.start(gpu))
        for gpu in CHUNKS:
            port = 8430 + gpu
            trial.base.wait_server(port, timeout=600)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(measure, CHUNKS))
    finally:
        trial.base.stop_servers(servers)
    rows = {str(gpu): summarize(ROOT / f"gpu{gpu}" / "ranges.json") for gpu in CHUNKS}
    trial.write(ROOT / "results.json", dict(status="complete", rows=rows))
    print(json.dumps({gpu: row["profiled_sampler_s"] for gpu, row in rows.items()}), flush=True)


if __name__ == "__main__":
    main()
