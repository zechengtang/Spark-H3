"""Paired current-code ComfyUI full-path profile on one prompt per duration."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import traceback

import comfyui_latest_spark_4prompt_20260926 as trial
from summarize_comfyui_pipeline_audit import summarize


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_current_fullpath_profile_20260928")
ARCHIVED = Path("/autodl-fs/data/h3_experiments/comfyui_latest_spark_4prompt_5s10s768p_20260926/protocol.json")
METHODS = ("sol_tau1_extra0", "spark_block", "spark_query")
RUNS = ((0, 10), (1, 5))
trial.source.cases = lambda: trial.read(ARCHIVED)["cases"]
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"


def worker(gpu: int, seconds: int) -> None:
    port = 8470 + gpu
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
        for method in (METHODS if gpu == 0 else reversed(METHODS)):
            dest = root / f"{seconds}s" / method
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "complete.json").exists():
                continue
            base.api_json(port, "/h3_audit/reset?enabled=0")
            base.queue_and_wait(port, trial.graph(seconds, method, case, dest / "warmup.safetensors", 5), "11")
            for measured in ((False, True) if gpu == 0 else (True, False)):
                label = "profiled" if measured else "control"
                if (dest / f"{label}_timing.json").exists() and (not measured or (dest / "ranges.json").exists()):
                    continue
                # The different-step run invalidates ComfyUI's sampler cache.
                base.api_json(port, "/h3_audit/reset?enabled=0")
                base.queue_and_wait(port, trial.graph(seconds, method, case, dest / f"pre_{label}.safetensors", 5), "11")
                base.api_json(port, f"/h3_audit/reset?enabled={int(measured)}")
                result = base.queue_and_wait(port, trial.graph(seconds, method, case, dest / f"{label}.safetensors"), "11")
                result.pop("history", None)
                if not result.get("sampler_seconds") or result["sampler_seconds"] < 10:
                    raise RuntimeError(f"Cached or missing sampler: {method} {label}")
                base.write_json(dest / f"{label}_timing.json", {
                    **result, "gpu": gpu, "seconds": seconds, "method": method,
                    "case": case["index"],
                })
                if measured:
                    audit = base.api_json(port, "/h3_audit/flush")
                    if audit["evaluations"] != 20:
                        raise RuntimeError(f"Expected 20 model evaluations, got {audit['evaluations']}")
                    base.write_json(dest / "ranges.json", audit)
                print("DONE", gpu, seconds, method, label, result["sampler_seconds"], flush=True)
            base.write_json(dest / "complete.json", {"status": "complete"})
    finally:
        base.stop_servers([(str(gpu), proc, log)])


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = [
        trial.REPO / "comfyui_backend.py", trial.REPO / "comfyui_nodes.py",
        trial.REPO / "comfyui_reblock_plan.py",
        trial.old.KITCHEN / "comfy_kitchen/backends/cuda/_C.abi3.so",
        trial.old.COMFY / "custom_nodes/h3_pipeline_audit.py",
    ]
    hashes = {str(p): trial.base.sha256(p) for p in sources}
    protocol_path = ROOT / "protocol.json"
    if protocol_path.exists():
        if trial.read(protocol_path)["source_hashes"] != hashes:
            raise RuntimeError("Implementation changed; use a new profile directory")
    else:
        trial.base.write_json(protocol_path, {
            "source_hashes": hashes, "case": trial.cases()[0], "runs": RUNS,
            "methods": METHODS, "settings": trial.read(ARCHIVED)["settings"],
            "midpoint_direction_mode": "legacy", "scope": "control + nested CUDA-event profile; no decode",
        })
    trial.base.write_json(ROOT / "status.json", {"status": "running", "pid": os.getpid()})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda pair: worker(*pair), RUNS))
        records = [summarize(path) for path in sorted(ROOT.glob("gpu*/*s/*/ranges.json"))]
        trial.base.write_json(ROOT / "summary.json", {"status": "complete", "records": records})
        trial.base.write_json(ROOT / "status.json", {"status": "complete"})
    except Exception:
        trial.base.write_json(ROOT / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
