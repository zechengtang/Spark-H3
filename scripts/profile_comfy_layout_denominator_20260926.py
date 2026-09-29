#!/usr/bin/env python3
"""Fresh one-prompt ComfyUI denoise denominators for layout-cost fractions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import traceback

import comfyui_sol_4way_benchmark_20260923 as base
import comfyui_latest_spark_4prompt_20260926 as latest


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_layout_conversion_profile_20260926")
PORT = 8510


def main():
    # This host may export an HTTP proxy; local ComfyUI API calls must bypass it.
    for key in ("NO_PROXY", "no_proxy"):
        entries = os.environ.get(key, "")
        os.environ[key] = f"{entries},127.0.0.1,localhost" if entries else "127.0.0.1,localhost"
    for name in ("input", "temp", "user", "output", "latents"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    case = latest.cases()[0]
    log = (ROOT / "server.log").open("a")
    cmd = [base.PYTHON, "main.py", "--listen", "127.0.0.1", "--port", str(PORT),
           "--disable-auto-launch", "--disable-cuda-malloc", "--preview-method", "none",
           "--cache-classic", "--output-directory", str(ROOT / "output"),
           "--temp-directory", str(ROOT / "temp"),
           "--input-directory", str(ROOT / "input"),
           "--user-directory", str(ROOT / "user")]
    process = subprocess.Popen(cmd, cwd=latest.old.COMFY, stdout=log,
                               stderr=subprocess.STDOUT,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "0",
                                    "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                                    "OMP_NUM_THREADS": "4"})
    base.write_json(ROOT / "server_pid.json", dict(pid=process.pid, port=PORT))
    try:
        base.wait_server(PORT)
        rows = []
        for seconds in (5, 10):
            warm = ROOT / "latents" / f"{seconds}s_warmup.safetensors"
            base.queue_and_wait(PORT, latest.graph(seconds, "spark_block", case, warm, 5),
                                latest.source.SAMPLER_NODE)
            target = ROOT / "latents" / f"{seconds}s_measured.safetensors"
            result = base.queue_and_wait(PORT, latest.graph(seconds, "spark_block", case, target, 20),
                                         latest.source.SAMPLER_NODE)
            if result["sampler_seconds"] < 10:
                raise RuntimeError("ComfyUI reused cached sampler result")
            rows.append(dict(seconds=seconds, sampler_seconds=result["sampler_seconds"],
                             latent=str(target), latent_sha256=base.sha256(target)))
            print("DENOISE", seconds, result["sampler_seconds"], flush=True)
        base.write_json(ROOT / "denominator.json", dict(
            status="complete", method="current Spark block, all-exact sink", case=case,
            seed=42, requested_denoise_steps=20, evaluations=20,
            sparse_calls=16 * 49, warmup="one excluded 5-step run per duration",
            timing="SamplerCustomAdvanced; excludes model load, text conditioning, VAE and save",
            records=rows))
    except Exception:
        base.write_json(ROOT / "denominator_error.json", dict(traceback=traceback.format_exc()))
        raise
    finally:
        base.stop_servers([("layout_profile", process, log)])


if __name__ == "__main__":
    main()
