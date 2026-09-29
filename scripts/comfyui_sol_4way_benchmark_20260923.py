#!/usr/bin/env python3
"""Four-way MiniMax-H3 Sol benchmark through denoise-only ComfyUI workers."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import av
import safetensors.torch
import torch
import websocket


NAME = "comfyui_sol_4way_blog_ablation_4prompts_20260923"
REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parent / "ComfyUI"
XMARRE = REPO.parent / "ComfyUI-Sol-H3"
SOURCE_PROTOCOL = Path("/autodl-fs/data/h3_experiments/blog_ablation_4prompts_rerun_20260922/protocol.json")
SOURCE_CONDITIONING = Path("/autodl-fs/data/h3_outputs/blog_ablation_4prompts_rerun_20260922/conditioning_cache")
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
EMBEDDINGS = COMFY / "models" / "embeddings" / NAME
METHODS = ("dense", "xmarre_sol", "comfyui_sol", "spark_h3_sol")
GPU = dict(zip(METHODS, range(4), strict=True))
PORT = {method: 8200 + gpu for method, gpu in GPU.items()}
DECODE_PORT = {method: 8210 + gpu for method, gpu in GPU.items()}
WIDTH, HEIGHT, REQUESTED_FRAMES, MODEL_FRAMES = 1344, 768, 240, 243
FPS, STEPS, SEED = 24.0, 20, 42
SAMPLER_NODE, DENOISE_SAVE_NODE, DECODE_SAVE_NODE = "11", "12", "10"
PYTHON = "/root/miniconda3/bin/python"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def cases():
    return json.loads(SOURCE_PROTOCOL.read_text())["cases"]


def condition_name(case) -> str:
    return f"{NAME}/case_{case['index']:02}_{case['sample_id']}.safetensors"


def condition_path(case) -> Path:
    return EMBEDDINGS / f"case_{case['index']:02}_{case['sample_id']}.safetensors"


def latent_path(method: str, case) -> Path:
    return ROOT / "latents" / method / f"case_{case['index']:02}_{case['sample_id']}.safetensors"


def patch_node(method: str, steps: int):
    if method == "dense":
        return None, ["1", 0]
    if method == "xmarre_sol":
        return {"class_type": "SolH3Experimental", "inputs": {
            "model": ["1", 0], "exact_fusion": False, "tau": 1.0,
            "dense_evaluations": 1, "dense_layers": 2,
        }}, ["2", 0]
    if method == "comfyui_sol":
        return {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["1", 0], "selection": "sol-attn", "selection.tau": 1.3,
            "start_percent": 0.2, "end_percent": 1.0, "dense_blocks": "",
            "min_tokens": 12288, "extra_tokens": 256,
            "sink_conditioning": "exact_kv_and_rows", "verbose": True,
        }}, ["2", 0]
    if method == "spark_h3_sol":
        return {"class_type": "MiniMaxH3SolAttentionSM120", "inputs": {
            "model": ["1", 0], "enabled": True, "steps": steps,
            "warmup_ratio": 0.2, "dense_layers": 1,
            "min_tokens": 4096, "strict": True, "tau": 1.0,
        }}, ["2", 0]
    raise ValueError(method)


def denoise_graph(method: str, case, target: Path, steps: int = STEPS):
    patch, model = patch_node(method, steps)
    nodes = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default",
        }},
        "3": {"class_type": "ConditioningLoader", "inputs": {"conditioning_name": condition_name(case)}},
        "4": {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {
            "width": WIDTH, "height": HEIGHT, "length": REQUESTED_FRAMES,
        }},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "BasicScheduler", "inputs": {
            "model": model, "scheduler": "simple", "steps": steps, "denoise": 1.0,
        }},
        "10": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["3", 0]}},
        SAMPLER_NODE: {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["7", 0], "guider": ["10", 0], "sampler": ["8", 0],
            "sigmas": ["9", 0], "latent_image": ["4", 0],
        }},
        DENOISE_SAVE_NODE: {"class_type": "SaveMiniMaxH3AVLatentCache", "inputs": {
            "samples": [SAMPLER_NODE, 0], "cache_path": str(target),
        }},
    }
    if patch is not None:
        nodes["2"] = patch
    return nodes


def decode_graph(method: str, case):
    output_path = OUT / method / f"{case['index']:02}_{case['sample_id']}_ultrafast.mp4"
    return {
        "1": {"class_type": "LoadMiniMaxH3AVLatentCache", "inputs": {"cache_path": str(latent_path(method, case))}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "4": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0], "vae": ["2", 0]}},
        "5": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["1", 0], "vae": ["3", 0]}},
        "6": {"class_type": "ImageFromBatch", "inputs": {
            "image": ["4", 0], "batch_index": 0, "length": REQUESTED_FRAMES,
        }},
        "7": {"class_type": "TrimAudioDuration", "inputs": {
            "audio": ["5", 0], "start_index": 0.0, "duration": 10.0,
        }},
        "8": {"class_type": "CreateVideo", "inputs": {
            "images": ["6", 0], "audio": ["7", 0], "fps": FPS,
            "bit_depth": 8, "color_space": "sRGB", "codec": "none",
        }},
        DECODE_SAVE_NODE: {"class_type": "SaveVideoLosslessUltrafast", "inputs": {
            "video": ["8", 0], "output_path": str(output_path),
        }},
    }


def api_json(port: int, endpoint: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{endpoint}") as response:
        return json.load(response)


def wait_server(port: int, timeout: float = 300.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return api_json(port, "/system_stats")
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(1)
    raise TimeoutError(f"ComfyUI on port {port} did not start")


def queue_and_wait(port: int, graph: dict, sampler_node: str | None = None):
    client_id, prompt_id = str(uuid.uuid4()), str(uuid.uuid4())
    ws = websocket.WebSocket(); ws.settimeout(7200)
    ws.connect(f"ws://127.0.0.1:{port}/ws?clientId={client_id}")
    payload = json.dumps({"prompt": graph, "client_id": client_id, "prompt_id": prompt_id}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/prompt", data=payload,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        queued = json.load(response)
    if queued.get("prompt_id") != prompt_id:
        raise RuntimeError(f"unexpected queue response: {queued}")
    started, sampler_started, sampler_seconds, error = time.perf_counter(), None, None, None
    while True:
        message = ws.recv()
        if not isinstance(message, str):
            continue
        event = json.loads(message); data = event.get("data", {})
        if data.get("prompt_id") != prompt_id:
            continue
        event_type, node, now = event.get("type"), data.get("node"), time.perf_counter()
        if event_type == "executing":
            if node == sampler_node:
                sampler_started = now
            elif sampler_started is not None and sampler_seconds is None:
                sampler_seconds = now - sampler_started
            if node is None:
                break
        elif event_type == "executed" and node == sampler_node and sampler_started is not None:
            sampler_seconds = now - sampler_started
        elif event_type == "execution_error":
            error = data; break
    ws.close()
    if error is not None:
        raise RuntimeError(json.dumps(error, ensure_ascii=False))
    history = api_json(port, f"/history/{prompt_id}")[prompt_id]
    if history.get("status", {}).get("status_str") != "success":
        raise RuntimeError(json.dumps(history.get("status"), ensure_ascii=False))
    return {"prompt_id": prompt_id, "sampler_seconds": sampler_seconds,
            "workflow_seconds": time.perf_counter() - started, "history": history}


def inspect_video(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
        return {"frames": frames, "width": stream.width, "height": stream.height,
                "average_rate": float(stream.average_rate), "audio_streams": len(container.streams.audio)}


def convert_conditioning() -> list[dict]:
    source = {}
    for path in SOURCE_CONDITIONING.glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source[payload["prompt"]] = (path, payload)
    records = []; EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    for case in cases():
        old_path, payload = source[case["prompt"]]; values = payload["values"]
        target = condition_path(case)
        safetensors.torch.save_file({
            "conditioning": values["prompt_embeds"].detach().cpu().contiguous(),
            "minimax_token_tags": values["text_token_tags"].detach().cpu().contiguous(),
        }, str(target), metadata={"conditioning_options": "{}"})
        records.append({"case": case["index"], "source": str(old_path), "source_sha256": sha256(old_path),
                        "converted": str(target), "converted_sha256": sha256(target),
                        "shape": list(values["prompt_embeds"].shape), "dtype": str(values["prompt_embeds"].dtype),
                        "token_tags_exact_key_mapping": True})
    return records


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    ROOT.mkdir(parents=True); OUT.mkdir(parents=True)
    for method in METHODS:
        for base in (ROOT / "user", ROOT / "temp", ROOT / "input", ROOT / "latents"):
            (base / method).mkdir(parents=True, exist_ok=True)
        (OUT / method).mkdir(parents=True)
    for base in (ROOT / "user", ROOT / "temp", ROOT / "input"):
        (base / "decoder").mkdir(parents=True, exist_ok=True)
    converted = convert_conditioning()
    protocol = {
        "name": NAME,
        "pipeline": "ComfyUI native MiniMax-H3; cached Diffusers BF16 conditioning; denoise and decode separated",
        "methods": {
            "dense": {"gpu": 0, "patch": None},
            "xmarre_sol": {"gpu": 1, "node": "SolH3Experimental", "tau": 1.0, "dense_evaluations": 1, "dense_layers": 2},
            "comfyui_sol": {"gpu": 2, "node": "BlockSparseAttention", "tau": 1.3, "start_percent": 0.2, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows"},
            "spark_h3_sol": {"gpu": 3, "node": "MiniMaxH3SolAttentionSM120", "tau": 1.0, "warmup_ratio": 0.2, "dense_layers": 1, "spark_features": False},
        },
        "cases": cases(),
        "settings": {"seed": SEED, "steps": STEPS, "sampler": "res_multistep", "scheduler": "simple",
                     "width": WIDTH, "height": HEIGHT, "requested_frames": REQUESTED_FRAMES,
                     "model_aligned_frames": MODEL_FRAMES, "saved_frames": REQUESTED_FRAMES,
                     "fps": FPS, "duration_seconds": 10.0, "turbo_lora": False,
                     "warmup": "one unmeasured 5-step full-resolution denoise per method",
                     "timing": "SamplerCustomAdvanced execution only; conditioning and VAE excluded",
                     "video": "single ComfyUI decoder; MP4/H.264 CRF 0"},
        "conditioning": converted,
        "revisions": {"comfyui": git_revision(COMFY), "xmarre": git_revision(XMARRE), "spark_h3": git_revision(REPO)},
        "source_protocol": str(SOURCE_PROTOCOL), "source_protocol_sha256": sha256(SOURCE_PROTOCOL),
    }
    write_json(ROOT / "protocol.json", protocol)


def start_server(label: str, gpu: int, port: int, output: Path):
    log = (ROOT / f"server_{label}.log").open("a")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    command = [PYTHON, "main.py", "--listen", "127.0.0.1", "--port", str(port),
               "--disable-auto-launch", "--disable-cuda-malloc", "--preview-method", "none",
               "--output-directory", str(output), "--temp-directory", str(ROOT / "temp" / label),
               "--input-directory", str(ROOT / "input" / label), "--user-directory", str(ROOT / "user" / label)]
    process = subprocess.Popen(command, cwd=COMFY, env=env, stdout=log, stderr=subprocess.STDOUT)
    return process, log


def stop_servers(servers):
    for _, process, _ in servers:
        process.terminate()
    for _, process, log in servers:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        log.close()


def run_method(method: str):
    source_cases = cases(); method_root = ROOT / method
    warm_target = ROOT / "latents" / method / "_warmup_5step.safetensors"
    warm = queue_and_wait(PORT[method], denoise_graph(method, source_cases[0], warm_target, 5), SAMPLER_NODE)
    warm.pop("history"); warm.update(latent_path=str(warm_target), latent_sha256=sha256(warm_target))
    write_json(method_root / "warmup.json", warm)
    rows = []
    for case in source_cases:
        target = latent_path(method, case)
        row = queue_and_wait(PORT[method], denoise_graph(method, case, target), SAMPLER_NODE); row.pop("history")
        row.update(method=method, gpu=GPU[method], case=case["index"], sample_id=case["sample_id"],
                   prompt_sha256=case["prompt_sha256"], seed=SEED, steps=STEPS,
                   width=WIDTH, height=HEIGHT, requested_frames=REQUESTED_FRAMES, model_frames=MODEL_FRAMES,
                   latent_path=str(target), latent_sha256=sha256(target), latent_bytes=target.stat().st_size)
        rows.append(row); write_json(method_root / f"case_{case['index']:02}.json", row)
    manifest = {"status": "denoise_complete", "method": method, "gpu": GPU[method], "records": rows}
    write_json(method_root / "denoise_manifest.json", manifest)
    return manifest


def denoise_all():
    servers = []
    try:
        for method in METHODS:
            process, log = start_server(method, GPU[method], PORT[method], OUT / method)
            servers.append((method, process, log))
        for method in METHODS:
            write_json(ROOT / f"system_{method}.json", wait_server(PORT[method]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(run_method, method): method for method in METHODS}
            manifests = {method: future.result() for future, method in futures.items()}
        write_json(ROOT / "denoise_summary.json", {"status": "complete", "methods": manifests})
    finally:
        stop_servers(servers)


def decode_method(method: str):
    records = []
    for case in cases():
        pattern = f"{case['index']:02}_{case['sample_id']}_*.mp4"
        matches = sorted((OUT / method).glob(pattern))
        if matches:
            if len(matches) != 1:
                raise RuntimeError(f"ambiguous existing decode outputs: {matches}")
            result = {"prompt_id": None, "sampler_seconds": None,
                      "workflow_seconds": None, "reused_existing": True}
            path = matches[0]
        else:
            result = queue_and_wait(DECODE_PORT[method], decode_graph(method, case))
            result.pop("history")
            matches = sorted((OUT / method).glob(pattern))
            if len(matches) != 1:
                raise RuntimeError(f"expected one decoded video for {pattern}, got {matches}")
            path = matches[0]
            result["reused_existing"] = False
        info = inspect_video(path)
        expected = {"frames": REQUESTED_FRAMES, "width": WIDTH, "height": HEIGHT,
                    "average_rate": FPS, "audio_streams": 1}
        if info != expected:
            raise RuntimeError(f"invalid video {path}: {info}")
        result.update(method=method, case=case["index"], sample_id=case["sample_id"],
                      video_path=str(path), video_sha256=sha256(path), video_bytes=path.stat().st_size,
                      video=info)
        records.append(result); write_json(ROOT / "decode" / method / f"case_{case['index']:02}.json", result)
    return records


def decode_all():
    servers = []
    try:
        for method in METHODS:
            label = f"decoder_{method}"
            for base in (ROOT / "user", ROOT / "temp", ROOT / "input"):
                (base / label).mkdir(parents=True, exist_ok=True)
            process, log = start_server(label, GPU[method], DECODE_PORT[method], OUT)
            servers.append((label, process, log))
        for method in METHODS:
            write_json(ROOT / f"system_decoder_{method}.json", wait_server(DECODE_PORT[method]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(decode_method, method): method for method in METHODS}
            by_method = {method: future.result() for future, method in futures.items()}
        records = [record for method in METHODS for record in by_method[method]]
        write_json(ROOT / "decode_summary.json", {"status": "complete", "records": records})
    finally:
        stop_servers(servers)


def launch():
    prepare(); denoise_all(); decode_all()
    write_json(ROOT / "generation_summary.json", {"status": "complete",
               "denoise": str(ROOT / "denoise_summary.json"), "decode": str(ROOT / "decode_summary.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "denoise", "decode", "launch")); args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "denoise": denoise_all()
    elif args.command == "decode": decode_all()
    else: launch()
