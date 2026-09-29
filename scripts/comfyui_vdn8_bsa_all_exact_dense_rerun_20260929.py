"""Rerun 4x3 VDN8 dense references with BSA all-key-block sinks on GPU 2 and 3."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import comfyui_sol_4way_benchmark_20260923 as api
import comfyui_vdn8_four_model_72_run_20260929 as prior
from comfyui_vdn8_decode_local_20260929 import LOCAL as PRIOR_LOCAL


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_vdn8_bsa_exact_scheduled_20260929")
OUT = Path("/autodl-fs/data/h3_outputs/comfyui_vdn8_bsa_exact_scheduled_20260929")
LOCAL = Path("/tmp/h3_vdn8_bsa_exact_scheduled_20260929")
PORT = {1: 8921, 2: 8922, 3: 8923}
ASSIGNMENT = {1: ("minimax_h3",), 2: ("lightx2v",),
              3: ("larryvrh", "comfyui")}
QUALITY_SCRIPT = prior.BENCH / "scripts" / "h3_quality_video_pair.py"


def exact_graph(model, case, target, steps=None):
    graph = prior.denoise_graph(model, "dense", case, target, steps)
    source = graph["9"]["inputs"]["model"]
    graph["6"] = {"class_type": "MiniMaxH3BSAAllExactScheduled", "inputs": {
        "model": source, "steps": steps or prior.steps_for(model),
        "warmup_percent": 20.0, "dense_layers": 1, "min_tokens": 12288,
    }}
    graph["9"]["inputs"]["model"] = ["6", 0]
    graph["10"]["inputs"]["model"] = ["6", 0]
    return graph


def server(gpu):
    root = LOCAL / f"gpu{gpu}" / "comfy"
    for sub in ("input", "temp", "user", "output"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    log_path = ROOT / f"gpu{gpu}" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
           "PYTHONUNBUFFERED": "1", "NO_PROXY": "127.0.0.1,localhost",
           "no_proxy": "127.0.0.1,localhost"}
    command = [prior.PYTHON, "main.py", "--listen", "127.0.0.1",
               "--port", str(PORT[gpu]), "--disable-auto-launch",
               "--disable-cuda-malloc", "--preview-method", "none",
               "--output-directory", str(root / "output"),
               "--temp-directory", str(root / "temp"),
               "--input-directory", str(root / "input"),
               "--user-directory", str(root / "user")]
    process = subprocess.Popen(command, cwd=prior.COMFY, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    return process, log


def run_model(gpu, model):
    root = ROOT / model
    cases = prior.cases()
    warm = root / "latents" / "warmup_5s_2step.safetensors"
    warm_record = root / "records" / "warmup.json"
    if not warm.exists() or not warm_record.exists():
        print("WARMUP START", gpu, model, flush=True)
        started = time.perf_counter()
        response = prior.execute(PORT[gpu], exact_graph(model, cases[0], warm, 2), "11")
        if not warm.exists():
            raise RuntimeError(f"warmup latent missing: {warm}")
        prior.write(warm_record, {"status": "complete", "gpu": gpu, "model": model,
                                 "seconds": time.perf_counter() - started,
                                 "sampler_seconds": response["sampler_seconds"],
                                 "latent_sha256": api.sha256(warm)})
    for case in cases:
        label = case["label"]
        latent = root / "latents" / f"{label}_bsa_all_exact.safetensors"
        record = root / "records" / f"{label}_bsa_all_exact.json"
        graph = exact_graph(model, case, latent)
        graph_path = root / "graphs" / f"{label}.json"
        prior.write(graph_path, graph)
        if not latent.exists() or not record.exists() or prior.read(record).get("status") != "complete":
            print("DENOISE START", gpu, model, label, flush=True)
            response = prior.execute(PORT[gpu], graph, "11")
            if not latent.exists() or not response["sampler_seconds"]:
                raise RuntimeError(f"denoise output missing: {model} {label}")
            prior.write(record, {"status": "complete", "gpu": gpu, "model": model,
                                 "case": label, "method": "bsa_all_exact_dense",
                                 "seed": prior.SEED, "steps": prior.steps_for(model),
                                 "prompt_sha256": api.sha256(Path(case["workflow"])),
                                 "graph_path": str(graph_path),
                                 "latent_path": str(latent), "latent_sha256": api.sha256(latent),
                                 "sampler_seconds": response["sampler_seconds"],
                                 "workflow_seconds": response["workflow_seconds"]})
            print("DENOISED", gpu, model, label, round(response["sampler_seconds"], 2), flush=True)
        else:
            print("REUSE DENOISE", gpu, model, label, flush=True)

    for case in cases:
        label = case["label"]
        latent = root / "latents" / f"{label}_bsa_all_exact.safetensors"
        local = LOCAL / model / "videos" / f"{label}_bsa_all_exact.mp4"
        dest = OUT / model / f"{label}_bsa_all_exact.mp4"
        record = root / "videos" / f"{label}_bsa_all_exact.json"
        if dest.exists() and record.exists() and prior.read(record).get("status") == "complete":
            print("REUSE VIDEO", model, label, flush=True)
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            print("DECODE START", gpu, model, label, flush=True)
            prior.execute(PORT[gpu], prior.decode_graph(latent, local, case))
        video = api.inspect_video(local)
        if (video["frames"], video["width"], video["height"]) != (
                case["frames"], case["width"], case["height"]):
            raise RuntimeError(f"bad video shape: {local}: {video}")
        digest = api.sha256(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(f".{dest.name}.copy-{os.getpid()}")
        shutil.copyfile(local, temp)
        temp.replace(dest)
        prior.write(record, {"status": "complete", "gpu": gpu, "model": model,
                             "case": label, "method": "bsa_all_exact_dense",
                             "output_path": str(dest), "local_path": str(local),
                             "sha256": digest, "video": video})
        print("VIDEO COMPLETE", gpu, model, label, flush=True)
    prior.write(root / "generation_complete.json", {"status": "complete", "model": model,
                                                     "gpu": gpu, "videos": 3})


def generate_gpu(gpu):
    process, log = server(gpu)
    try:
        api.wait_server(PORT[gpu])
        print("SERVER READY", gpu, flush=True)
        for model in ASSIGNMENT[gpu]:
            run_model(gpu, model)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


def quality_model(gpu, model):
    root = ROOT / model
    cases = prior.cases()
    refs = []
    for case in cases:
        rec = prior.read(root / "videos" / f"{case['label']}_bsa_all_exact.json")
        path = Path(rec["local_path"])
        if not path.exists() or api.sha256(path) != rec["sha256"]:
            raise RuntimeError(f"local new reference missing or changed: {path}")
        refs.append({"index": case["index"], "output_path": str(path),
                     "sha256": rec["sha256"], "video": rec["video"]})
    ref_manifest = root / "quality" / "bsa_all_exact_reference.json"
    prior.write(ref_manifest, {"status": "passed", "sample_count": 3, "records": refs})
    for method in (*prior.METHODS[1:], "old_dense"):
        records = []
        for case in cases:
            label = case["label"]
            old_method = "dense" if method == "old_dense" else method
            old = prior.read(prior.ROOT / model / "videos" / f"{label}_{old_method}.json")
            path = PRIOR_LOCAL / model / "videos" / f"{label}_{old_method}.mp4"
            if not path.exists() or api.sha256(path) != old["sha256"]:
                raise RuntimeError(f"old local candidate missing or changed: {path}")
            records.append({"index": case["index"], "output_path": str(path),
                            "sha256": old["sha256"], "video": old["video"]})
        candidate_manifest = root / "quality" / f"{method}_manifest.json"
        prior.write(candidate_manifest, {"status": "passed", "sample_count": 3,
                                         "records": records})
        config = {"work_dir": str(root / "quality" / method),
                  "reference_manifest": str(ref_manifest),
                  "candidate_manifest": str(candidate_manifest),
                  "method": method, "cases": [1, 2, 3], "workers": 1}
        config_path = root / "quality" / f"{method}_config.json"
        prior.write(config_path, config)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "H3_NUM_GPUS": "1",
               "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"}
        print("SCORE START", gpu, model, method, flush=True)
        for command in ("worker", "aggregate"):
            cmd = [prior.PYTHON, str(QUALITY_SCRIPT), command,
                   "--config", str(config_path)]
            if command == "worker":
                cmd += ["--slot", "0"]
            subprocess.run(cmd, check=True, env=env)
        print("SCORED", gpu, model, method, flush=True)
    prior.write(root / "quality_complete.json", {"status": "complete", "model": model,
                                                 "methods": 6, "pairs": 18})


def score_gpu(gpu):
    for model in ASSIGNMENT[gpu]:
        quality_model(gpu, model)


def aggregate():
    rows = []
    for model in prior.MODELS:
        root = ROOT / model
        for method in (*prior.METHODS[1:], "old_dense"):
            for case in prior.cases():
                label = case["label"]
                row = prior.read(root / "quality" / method / f"{method}_{case['index']:02}.json")
                new_dense = prior.read(root / "records" / f"{label}_bsa_all_exact.json")
                old_method = "dense" if method == "old_dense" else method
                old = prior.read(prior.ROOT / model / "records" / f"{label}_{old_method}.json")
                rows.append({"model": model, "duration": label, "method": method,
                             "psnr_db": row["psnr_db"], "ssim": row["ssim"],
                             "lpips": row["lpips"],
                             "dense_sampler_seconds": new_dense["sampler_seconds"],
                             "candidate_sampler_seconds": old["sampler_seconds"],
                             "speedup": new_dense["sampler_seconds"] / old["sampler_seconds"],
                             "reference_video": str(OUT / model / f"{label}_bsa_all_exact.mp4"),
                             "candidate_video": str(prior.OUT / model / f"{label}_{old_method}.mp4")})
    prior.write(ROOT / "results.json", {"status": "complete", "reference": "bsa_all_exact_dense",
                                        "new_dense_videos": 12, "scored_pairs": len(rows),
                                        "rows": rows})
    columns = ("model", "duration", "method", "psnr_db", "ssim", "lpips",
               "dense_sampler_seconds", "candidate_sampler_seconds", "speedup")
    csv = [",".join(columns)] + [",".join(str(row[key]) for key in columns) for row in rows]
    (ROOT / "results.csv").write_text("\n".join(csv) + "\n")
    protocol = prior.read(ROOT / "protocol.json")
    protocol["status"] = "complete"
    protocol["completed_at"] = time.time()
    protocol["quality_gpu_assignment"] = {"minimax_h3": 1, "lightx2v": 2,
                                          "larryvrh": 2, "comfyui": 2}
    prior.write(ROOT / "protocol.json", protocol)
    print("ALL COMPLETE", len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("generate", "score", "aggregate", "all"), default="all")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    prior.write(ROOT / "protocol.json", {
        "status": "running",
        "method": "BSA full-range sink_blocks, tail=False on active sparse calls; SDPA warmup/layer0",
        "seed": prior.SEED, "models": prior.MODELS, "cases": prior.cases(),
        "gpu_assignment": ASSIGNMENT, "source_dense": str(prior.ROOT),
        "warmup_percent": 20.0, "warmup_evaluations": {"minimax_h3": 4, "lightx2v": 2,
                                                    "larryvrh": 2, "comfyui": 2},
        "sdpa_dense_layers": [0], "min_tokens": 12288,
        "bsa_node_source_sha256": api.sha256(prior.REPO / "comfyui_nodes.py"),
        "comfy_sparse_source_sha256": api.sha256(prior.COMFY / "comfy_extras" /
                                                   "nodes_sparse_attention.py"),
    })
    if args.phase in ("generate", "all"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(generate_gpu, gpu) for gpu in ASSIGNMENT]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    if args.phase in ("score", "all"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(score_gpu, gpu) for gpu in ASSIGNMENT]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    if args.phase in ("aggregate", "all"):
        aggregate()


if __name__ == "__main__":
    main()
