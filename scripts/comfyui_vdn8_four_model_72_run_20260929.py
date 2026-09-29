"""Generate the 72 VDN8 ComfyUI AV videos and score each sparse arm vs dense."""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import comfyui_sol_4way_benchmark_20260923 as api


REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parent / "ComfyUI"
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_vdn8_four_model_72_20260929")
OUT = Path("/autodl-fs/data/h3_outputs/comfyui_vdn8_four_model_72_20260929")
PYTHON = "/root/miniconda3/bin/python"
BASE_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
METHODS = ("dense", "official_sol", "spark_10pct", "spark_20pct", "spark_114blocks", "spark_228blocks")
MODELS = ("minimax_h3", "lightx2v", "larryvrh", "comfyui")
LORAS = {
    "lightx2v": "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "larryvrh": "minimax_h3_turbo_v4_step600_ema.safetensors",
    "comfyui": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
}
SEED = 42
FPS = 24
PORT_BASE = 8610


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def execute(port, graph, sampler_node=None):
    for attempt in range(3):
        try:
            return api.queue_and_wait(port, graph, sampler_node)
        except Exception as error:
            if attempt == 2:
                raise
            print("RETRY", port, attempt + 1, type(error).__name__, str(error)[:500], flush=True)
            time.sleep(5 * (attempt + 1))


def cases():
    rows = []
    for index, label in enumerate(("5s", "10s", "14p4s"), 1):
        path = REPO / "workflows" / f"spark_h3_vdn8_{label}_t2va.json"
        workflow = read(path)
        node = next(node for node in workflow["nodes"] if node["type"] == "MiniMaxH3ImageToVideo")
        values = node["widgets_values_named"]
        rows.append(dict(index=index, label=label, prompt=values["prompt"],
                         width=values["width"], height=values["height"],
                         frames=values["length"], workflow=str(path),
                         workflow_sha256=api.sha256(path)))
    return rows


def steps_for(model):
    return 20 if model == "minimax_h3" else 8


def patched_model(nodes, model, method, steps):
    source = ["1", 0]
    if model in ("lightx2v", "comfyui"):
        nodes["2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": source, "lora_name": LORAS[model], "strength_model": 1.0}}
        source = ["2", 0]
    elif model == "larryvrh":
        nodes["2"] = {"class_type": "MiniMaxH3TurboLoRA", "inputs": {
            "model": source, "lora_name": LORAS[model], "strength": 1.0,
            "low_vram": False}}
        source = ["2", 0]
    if method == "dense":
        return source
    if method == "official_sol":
        nodes["6"] = {"class_type": "BlockSparseAttention", "inputs": {
            "model": source, "selection": "sol-attn", "selection.tau": 1.3,
            "start_percent": 0.2, "end_percent": 1.0, "dense_blocks": "",
            "min_tokens": 12288, "extra_tokens": 256,
            "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
    else:
        topk_mode = "topk_ratio" if "pct" in method else "topk_blocks"
        ratio = 0.1 if method == "spark_10pct" else 0.2
        blocks = 114 if method == "spark_114blocks" else 228
        nodes["6"] = {"class_type": "MiniMaxH3SparkAttentionSM120", "inputs": {
            "model": source, "enabled": True, "steps": steps,
            "warmup_ratio": 0.2, "warmup_mode": "warmup_ratio",
            "topk_mode": topk_mode, "topk_ratio": ratio, "topk_blocks": blocks,
            "dense_layers": 1, "min_tokens": 12288, "strict": True,
            "tail_granularity": "query"}}
    return ["6", 0]


def denoise_graph(model, method, case, target, steps=None):
    steps = steps or steps_for(model)
    nodes = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": BASE_MODEL, "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "3": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "prompt": case["prompt"],
            "width": case["width"], "height": case["height"],
            "length": case["frames"]}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "9": {"class_type": "BasicScheduler", "inputs": {
            "model": None, "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "10": {"class_type": "BasicGuider", "inputs": {
            "model": None, "conditioning": ["3", 0]}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["7", 0], "guider": ["10", 0],
            "sampler": ["8", 0], "sigmas": ["9", 0],
            "latent_image": ["3", 1]}},
        "12": {"class_type": "SaveMiniMaxH3AVLatentCache", "inputs": {
            "samples": ["11", 0], "cache_path": str(target)}},
    }
    source = patched_model(nodes, model, method, steps)
    nodes["9"]["inputs"]["model"] = source
    nodes["10"]["inputs"]["model"] = source
    nodes["8"] = ({"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
                   if model == "larryvrh" else
                   {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}})
    return nodes


def decode_graph(latent, output, case):
    return {
        "1": {"class_type": "LoadMiniMaxH3AVLatentCache", "inputs": {"cache_path": str(latent)}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "4": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0], "vae": ["2", 0]}},
        "5": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["1", 0], "vae": ["3", 0]}},
        "6": {"class_type": "ImageFromBatch", "inputs": {
            "image": ["4", 0], "batch_index": 0, "length": case["frames"]}},
        "7": {"class_type": "TrimAudioDuration", "inputs": {
            "audio": ["5", 0], "start_index": 0.0,
            "duration": case["frames"] / FPS}},
        "8": {"class_type": "CreateVideo", "inputs": {
            "images": ["6", 0], "audio": ["7", 0], "fps": FPS,
            "bit_depth": 8, "color_space": "sRGB", "codec": "none"}},
        "10": {"class_type": "SaveVideoLosslessUltrafast", "inputs": {
            "video": ["8", 0], "output_path": str(output)}},
    }


def start_server(gpu):
    label = MODELS[gpu]
    root = ROOT / label
    for kind in ("user", "input", "temp"):
        (root / kind).mkdir(parents=True, exist_ok=True)
    log = (root / "server.log").open("a")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
           "PYTHONUNBUFFERED": "1", "NO_PROXY": "127.0.0.1,localhost",
           "no_proxy": "127.0.0.1,localhost"}
    command = [PYTHON, "main.py", "--listen", "127.0.0.1",
               "--port", str(PORT_BASE + gpu), "--disable-auto-launch",
               "--disable-cuda-malloc", "--preview-method", "none",
               "--output-directory", str(OUT / label),
               "--temp-directory", str(root / "temp"),
               "--input-directory", str(root / "input"),
               "--user-directory", str(root / "user")]
    process = subprocess.Popen(command, cwd=COMFY, env=env, stdout=log,
                               stderr=subprocess.STDOUT)
    return process, log


def run_model(gpu, model, port):
    root = ROOT / model
    case_rows = cases()
    warmup_path = root / "warmup_5s_2step_dense.safetensors"
    if not warmup_path.exists():
        started = time.perf_counter()
        execute(port, denoise_graph(model, "dense", case_rows[0], warmup_path, 2), "11")
        write(root / "warmup.json", dict(model=model, gpu=gpu, steps=2,
                                          seconds=time.perf_counter() - started,
                                          latent_path=str(warmup_path)))
        print("WARMUP", model, round(time.perf_counter() - started, 2), flush=True)
    else:
        print("REUSE WARMUP", model, flush=True)

    for case in case_rows:
        for method in METHODS:
            stem = f"{case['label']}_{method}"
            latent = root / "latents" / f"{stem}.safetensors"
            record = root / "records" / f"{stem}.json"
            if latent.exists() and record.exists() and read(record).get("status") == "complete":
                print("REUSE DENOISE", model, stem, flush=True)
                continue
            result = execute(port, denoise_graph(model, method, case, latent), "11")
            result.pop("history", None)
            if not latent.exists() or not result.get("sampler_seconds"):
                raise RuntimeError(f"missing denoise output or timing: {model} {stem}")
            write(record, dict(status="complete", model=model, gpu=gpu, method=method,
                               case=case["label"], steps=steps_for(model), seed=SEED,
                               latent_path=str(latent), latent_sha256=api.sha256(latent),
                               **result))
            print("DENOISED", model, stem, round(result["sampler_seconds"], 2), flush=True)

    # Decode on the same resident ComfyUI process. VAEs are loaded once and reused.
    for case in case_rows:
        for method in METHODS:
            stem = f"{case['label']}_{method}"
            latent = root / "latents" / f"{stem}.safetensors"
            output = OUT / model / f"{stem}.mp4"
            record = root / "videos" / f"{stem}.json"
            if output.exists() and record.exists() and read(record).get("status") == "complete":
                print("REUSE VIDEO", model, stem, flush=True)
                continue
            result = execute(port, decode_graph(latent, output, case))
            result.pop("history", None)
            if not output.exists():
                raise RuntimeError(f"missing video: {output}")
            video = api.inspect_video(output)
            if video["frames"] != case["frames"] or video["width"] != case["width"] or video["height"] != case["height"]:
                raise RuntimeError(f"incorrect video shape: {output}: {video}")
            write(record, dict(status="complete", model=model, method=method,
                               case=case["label"], output_path=str(output),
                               sha256=api.sha256(output), video=video, **result))
            print("DECODED", model, stem, video, flush=True)
    write(root / "generation_complete.json", dict(status="complete", model=model,
           cases=len(case_rows), methods=len(METHODS), videos=len(case_rows) * len(METHODS)))


def score_model(gpu, model):
    root = ROOT / model
    case_rows = cases()
    for method in METHODS:
        records = []
        for case in case_rows:
            record = read(root / "videos" / f"{case['label']}_{method}.json")
            records.append(dict(index=case["index"], output_path=record["output_path"],
                                sha256=record["sha256"], video=record["video"]))
        write(root / "quality" / f"{method}_manifest.json", dict(
            status="passed", method=method, sample_count=len(records), records=records))
    summaries = {}
    for method in METHODS[1:]:
        config = dict(work_dir=str(root / "quality" / method),
                      reference_manifest=str(root / "quality" / "dense_manifest.json"),
                      candidate_manifest=str(root / "quality" / f"{method}_manifest.json"),
                      method=method, cases=[1, 2, 3], workers=1)
        config_path = root / "quality" / f"{method}_config.json"
        write(config_path, config)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "H3_NUM_GPUS": "1",
               "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
        subprocess.run([PYTHON, str(BENCH / "scripts" / "h3_quality_video_pair.py"),
                        "run", "--config", str(config_path)], check=True, env=env)
        summaries[method] = read(root / "quality" / method /
                                 f"{method}_quality_results.json")["summary"]
        print("SCORED", model, method, summaries[method], flush=True)
    write(root / "quality_results.json", dict(status="complete", model=model,
                                               summary=summaries))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write(ROOT / "protocol.json", dict(status="running", created=time.time(),
          models={model: dict(gpu=gpu, steps=steps_for(model), lora=LORAS.get(model))
                  for gpu, model in enumerate(MODELS)},
          cases=cases(), methods=METHODS, seed=SEED,
          sol=dict(tau=1.3, start_percent=0.2, end_percent=1.0, dense_blocks="",
                   min_tokens=12288, extra_tokens=256, sink_conditioning="exact_kv_and_rows"),
          spark=dict(warmup_ratio=0.2, dense_layers=1, min_tokens=12288,
                     tail_granularity="query", global_anchor_dtype="float32",
                     midpoint_direction_mode="fused"),
          sources={str(path): api.sha256(path) for path in
                   (REPO / "comfyui_nodes.py", REPO / "comfyui_backend.py",
                    COMFY / "comfy_extras" / "nodes_sparse_attention.py")}))
    servers = []
    try:
        for gpu in range(4):
            process, log = start_server(gpu)
            servers.append((process, log))
        for gpu in range(4):
            api.wait_server(PORT_BASE + gpu)
            print("SERVER READY", MODELS[gpu], PORT_BASE + gpu, flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(run_model, gpu, model, PORT_BASE + gpu): model
                       for gpu, model in enumerate(MODELS)}
            for future in concurrent.futures.as_completed(futures):
                future.result()
                print("MODEL GENERATED", futures[future], flush=True)
    finally:
        for process, _ in servers:
            process.terminate()
        for process, log in servers:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            log.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(score_model, gpu, model): model
                   for gpu, model in enumerate(MODELS)}
        for future in concurrent.futures.as_completed(futures):
            future.result()
            print("MODEL SCORED", futures[future], flush=True)
    summary = {model: read(ROOT / model / "quality_results.json")["summary"]
               for model in MODELS}
    write(ROOT / "results.json", dict(status="complete", videos=72,
                                      scored_pairs=60, quality=summary))
    print("ALL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
