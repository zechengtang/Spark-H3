"""Resume VDN8 latent decoding with local MP4 staging to avoid shared-FS stalls."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import shutil
import subprocess
import time

import comfyui_sol_4way_benchmark_20260923 as api
import comfyui_vdn8_four_model_72_run_20260929 as trial


LOCAL = Path("/tmp/h3_vdn8_20260929_outputs")
PORT_BASE = 8710


def start_server(gpu):
    model = trial.MODELS[gpu]
    root = LOCAL / model / "comfy"
    for kind in ("input", "temp", "user", "output"):
        (root / kind).mkdir(parents=True, exist_ok=True)
    log = (trial.ROOT / model / "local_decode_server.log").open("a")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
           "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    command = [trial.PYTHON, "main.py", "--listen", "127.0.0.1",
               "--port", str(PORT_BASE + gpu), "--disable-auto-launch",
               "--disable-cuda-malloc", "--preview-method", "none",
               "--output-directory", str(root / "output"),
               "--temp-directory", str(root / "temp"),
               "--input-directory", str(root / "input"),
               "--user-directory", str(root / "user")]
    process = subprocess.Popen(command, cwd=trial.COMFY, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    return process, log


def decode_model(gpu, limit=None):
    model = trial.MODELS[gpu]
    port = PORT_BASE + gpu
    process, log = start_server(gpu)
    completed = 0
    try:
        api.wait_server(port)
        print("LOCAL SERVER READY", model, flush=True)
        for case in trial.cases():
            for method in trial.METHODS:
                stem = f"{case['label']}_{method}"
                root = trial.ROOT / model
                record_path = root / "videos" / f"{stem}.json"
                dest = trial.OUT / model / f"{stem}.mp4"
                local = LOCAL / model / "videos" / f"{stem}.mp4"
                latent = root / "latents" / f"{stem}.safetensors"
                if record_path.exists() and dest.exists() and trial.read(record_path).get("status") == "complete":
                    print("REUSE VIDEO", model, stem, flush=True)
                    continue
                if dest.exists() and not record_path.exists():
                    video = api.inspect_video(dest)
                    if (video["frames"], video["width"], video["height"]) != (
                            case["frames"], case["width"], case["height"]):
                        raise RuntimeError(f"incorrect existing video shape: {dest}: {video}")
                    trial.write(record_path, dict(status="complete", model=model,
                                method=method, case=case["label"], output_path=str(dest),
                                sha256=api.sha256(dest), video=video,
                                recovered_existing=True))
                    print("RECOVERED VIDEO", model, stem, flush=True)
                    continue
                local.parent.mkdir(parents=True, exist_ok=True)
                if not local.exists():
                    started = time.perf_counter()
                    result = trial.execute(port, trial.decode_graph(latent, local, case))
                    result.pop("history", None)
                    print("LOCAL DECODED", model, stem,
                          round(time.perf_counter() - started, 2), flush=True)
                else:
                    result = dict(prompt_id=None, sampler_seconds=None,
                                  workflow_seconds=None, reused_local=True)
                video = api.inspect_video(local)
                if (video["frames"], video["width"], video["height"]) != (
                        case["frames"], case["width"], case["height"]):
                    raise RuntimeError(f"incorrect video shape: {local}: {video}")
                digest = api.sha256(local)
                dest.parent.mkdir(parents=True, exist_ok=True)
                temporary = dest.with_name(f".{dest.name}.copy-{os.getpid()}")
                shutil.copyfile(local, temporary)
                temporary.replace(dest)
                if dest.stat().st_size != local.stat().st_size:
                    raise RuntimeError(f"copy size mismatch: {dest}")
                trial.write(record_path, dict(status="complete", model=model,
                            method=method, case=case["label"], output_path=str(dest),
                            local_path=str(local), sha256=digest, video=video, **result))
                completed += 1
                print("PUBLISHED", model, stem, flush=True)
                if limit is not None and completed >= limit:
                    return
        trial.write(root / "generation_complete.json", dict(status="complete", model=model,
                    cases=3, methods=6, videos=18))
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=range(4))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.gpu is not None:
        decode_model(args.gpu, args.limit)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(decode_model, gpu) for gpu in range(4)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        print("ALL VIDEOS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
