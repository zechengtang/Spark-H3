"""Paired full-denoise QKV producer chunk screen on two GPUs."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import traceback

import comfyui_latest_spark_4prompt_20260926 as trial


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_producer_chunk_paired_20260928")
ARCHIVED = Path("/autodl-fs/data/h3_experiments/comfyui_latest_spark_4prompt_5s10s768p_20260926/protocol.json")
CHUNKS = (16384, 24576, 32768)
trial.source.cases = lambda: trial.read(ARCHIVED)["cases"]
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"


def worker(gpu):
    base = trial.base
    case = trial.cases()[0]
    order = CHUNKS if gpu == 0 else tuple(reversed(CHUNKS))
    for chunk in order:
        root = ROOT / f"gpu{gpu}" / str(chunk)
        if (root / "record.json").exists():
            continue
        port = 8520 + gpu
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
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                 "H3_SPARK_PRODUCER_CHUNK": str(chunk), "H3_SPARK_DIRECT_OUTPUT": "1",
                 "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"},
        )
        try:
            base.wait_server(port, timeout=600)
            base.queue_and_wait(port, trial.graph(10, "spark_block", case, root / "warmup.safetensors", 5), "11")
            target = root / "measured.safetensors"
            result = base.queue_and_wait(port, trial.graph(10, "spark_block", case, target), "11")
            result.pop("history", None)
            if not result.get("sampler_seconds") or result["sampler_seconds"] < 10:
                raise RuntimeError(f"cached or missing sampler: GPU{gpu}, chunk {chunk}")
            base.write_json(root / "record.json", {
                "gpu": gpu, "chunk": chunk, "case": case["index"],
                "sampler_seconds": result["sampler_seconds"],
                "latent_sha256": base.sha256(target),
            })
            print("DONE", gpu, chunk, result["sampler_seconds"], flush=True)
        finally:
            base.stop_servers([(f"gpu{gpu}-{chunk}", proc, log)])


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    base = trial.base
    base.write_json(ROOT / "protocol.json", {
        "case": trial.cases()[0], "gpus": [0, 1], "chunks": CHUNKS,
        "order": {"0": CHUNKS, "1": list(reversed(CHUNKS))},
        "settings": {"seconds": 10, "resolution": [1344, 768], "steps": 20,
                     "seed": 42, "warmup": "excluded 5-step denoise per chunk/GPU",
                     "spark": "block, full reblock and global reweight, direct output"},
        "source_sha256": {
            str(path): base.sha256(path)
            for path in (trial.REPO / "comfyui_backend.py", trial.REPO / "comfyui_nodes.py",
                         trial.old.KITCHEN / "comfy_kitchen/backends/cuda/_C.abi3.so")
        },
    })
    base.write_json(ROOT / "status.json", {"status": "running", "pid": os.getpid()})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, (0, 1)))
        rows = [trial.read(path) for path in sorted(ROOT.glob("gpu*/*/record.json"))]
        if len(rows) != 2 * len(CHUNKS):
            raise RuntimeError(f"expected {2 * len(CHUNKS)} records, got {len(rows)}")
        for gpu in (0, 1):
            if len({row["latent_sha256"] for row in rows if row["gpu"] == gpu}) != 1:
                raise RuntimeError(f"GPU{gpu} chunk variants differ in latent output")
        base.write_json(ROOT / "summary.json", {"status": "complete", "records": rows})
        base.write_json(ROOT / "status.json", {"status": "complete"})
    except Exception:
        base.write_json(ROOT / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
