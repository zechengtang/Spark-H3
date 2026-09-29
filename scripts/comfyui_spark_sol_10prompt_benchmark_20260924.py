#!/usr/bin/env python3
"""ComfyUI Spark/Sol speed and quality generation on 10 prompts, 5s 768p."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import statistics

import safetensors.torch
import torch

import comfyui_sol_4way_benchmark_20260923 as base


NAME = "comfyui_spark_sol_10prompt_5s768p_20260924"
REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parent / "ComfyUI"
SOURCE_ROOT = Path("/autodl-fs/data/h3_experiments/early_steps_sparsity_10prompt_5s480p_seed42_20260924")
SOURCE_PROTOCOL = SOURCE_ROOT / "protocol.json"
SOURCE_CONDITIONING = Path("/autodl-fs/data/h3_outputs/early_steps_sparsity_10prompt_5s480p_seed42_20260924/conditioning_cache")
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
EMBEDDINGS = COMFY / "models" / "embeddings" / NAME

METHODS = (
    "dense",
    "sol_tau1_extra0",
    "sol_tau1_extra256",
    "sol_tau13_extra256",
    "spark_topk10",
)
QUALITY_METHODS = (
    "dense",
    "sol_tau1_extra0",
    "sol_tau1_extra256",
    "sol_tau13_extra256",
    "spark_topk10",
)
GPUS = (0, 1, 2, 3)
PORTS = tuple(8270 + gpu for gpu in GPUS)
DECODE_PORTS = {method: 8280 + gpu for gpu, method in enumerate(QUALITY_METHODS)}
WIDTH, HEIGHT, REQUESTED_FRAMES, MODEL_FRAMES = 1344, 768, 120, 123
FPS, STEPS, SEED = 24.0, 20, 42
SAMPLER_NODE = "11"


def cases():
    return json.loads(SOURCE_PROTOCOL.read_text())["cases"]


def condition_name(case):
    return f"{NAME}/case_{case['index']:02}_{case['sample_id']}.safetensors"


def condition_path(case):
    return EMBEDDINGS / f"case_{case['index']:02}_{case['sample_id']}.safetensors"


def latent_path(method, case):
    return ROOT / "latents" / method / f"case_{case['index']:02}_{case['sample_id']}.safetensors"


def patch_node(method, steps):
    if method == "dense":
        return None, ["1", 0]
    if method.startswith("sol_"):
        tau = 1.3 if method == "sol_tau13_extra256" else 1.0
        extra = 0 if method == "sol_tau1_extra0" else 256
        return {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["1", 0], "selection": "sol-attn", "selection.tau": tau,
            # 0.20 lands on a float32 sigma boundary and produces five dense
            # evaluations with the H3 simple schedule. 0.19 gives the intended
            # four, matching Spark's exact 20% evaluation-count warmup.
            "start_percent": 0.19, "end_percent": 1.0,
            "dense_blocks": "0", "min_tokens": 4096, "extra_tokens": extra,
            "sink_conditioning": "exact_kv_and_rows", "verbose": True,
        }}, ["2", 0]
    if method == "spark_topk10":
        return {"class_type": "MiniMaxH3SparkAttentionSM120", "inputs": {
            "model": ["1", 0], "enabled": True, "steps": steps,
            "warmup_percent": 20.0, "topk_ratio": 0.1,
            "dense_layers": 1, "min_tokens": 4096, "strict": True,
        }}, ["2", 0]
    raise ValueError(method)


def denoise_graph(method, case, target, steps=STEPS):
    patch, model = patch_node(method, steps)
    nodes = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "weight_dtype": "default",
        }},
        "3": {"class_type": "ConditioningLoader", "inputs": {
            "conditioning_name": condition_name(case),
        }},
        "4": {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {
            "width": WIDTH, "height": HEIGHT, "length": REQUESTED_FRAMES,
        }},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "BasicScheduler", "inputs": {
            "model": model, "scheduler": "simple", "steps": steps, "denoise": 1.0,
        }},
        "10": {"class_type": "BasicGuider", "inputs": {
            "model": model, "conditioning": ["3", 0],
        }},
        SAMPLER_NODE: {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["7", 0], "guider": ["10", 0], "sampler": ["8", 0],
            "sigmas": ["9", 0], "latent_image": ["4", 0],
        }},
        "12": {"class_type": "SaveMiniMaxH3AVLatentCache", "inputs": {
            "samples": [SAMPLER_NODE, 0], "cache_path": str(target),
        }},
    }
    if patch is not None:
        nodes["2"] = patch
    return nodes


def decode_graph(method, case):
    output = OUT / method / f"{case['index']:02}_{case['sample_id']}_ultrafast.mp4"
    return {
        "1": {"class_type": "LoadMiniMaxH3AVLatentCache", "inputs": {
            "cache_path": str(latent_path(method, case)),
        }},
        "2": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_video_vae_fp16.safetensors",
        }},
        "3": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_audio_vae_fp32.safetensors",
        }},
        "4": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0], "vae": ["2", 0]}},
        "5": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["1", 0], "vae": ["3", 0]}},
        "6": {"class_type": "ImageFromBatch", "inputs": {
            "image": ["4", 0], "batch_index": 0, "length": REQUESTED_FRAMES,
        }},
        "7": {"class_type": "TrimAudioDuration", "inputs": {
            "audio": ["5", 0], "start_index": 0.0, "duration": 5.0,
        }},
        "8": {"class_type": "CreateVideo", "inputs": {
            "images": ["6", 0], "audio": ["7", 0], "fps": FPS,
            "bit_depth": 8, "color_space": "sRGB", "codec": "none",
        }},
        "10": {"class_type": "SaveVideoLosslessUltrafast", "inputs": {
            "video": ["8", 0], "output_path": str(output),
        }},
    }


def convert_conditioning():
    source = {}
    for path in SOURCE_CONDITIONING.glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source[payload["prompt"]] = (path, payload)
    records = []
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    for case in cases():
        old_path, payload = source[case["prompt"]]
        values = payload["values"]
        target = condition_path(case)
        safetensors.torch.save_file({
            "conditioning": values["prompt_embeds"].detach().cpu().contiguous(),
            "minimax_token_tags": values["text_token_tags"].detach().cpu().contiguous(),
        }, str(target), metadata={"conditioning_options": "{}"})
        records.append({
            "case": case["index"], "source": str(old_path),
            "source_sha256": base.sha256(old_path), "converted": str(target),
            "converted_sha256": base.sha256(target),
            "shape": list(values["prompt_embeds"].shape),
            "dtype": str(values["prompt_embeds"].dtype),
        })
    return records


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for method in METHODS:
        (ROOT / "latents" / method).mkdir(parents=True)
        (ROOT / "records" / method).mkdir(parents=True)
        (ROOT / "warmup" / method).mkdir(parents=True)
        (OUT / method).mkdir(parents=True)
    for gpu in GPUS:
        for section in ("user", "temp", "input"):
            (ROOT / section / f"gpu{gpu}").mkdir(parents=True)
    converted = convert_conditioning()
    base.write_json(ROOT / "protocol.json", {
        "name": NAME,
        "purpose": "ComfyUI-native Spark Top-K 10% speed vs official Sol; requested tau/extra-token quality comparisons",
        "pipeline": "ComfyUI denoise-only with reused Diffusers BF16 conditioning",
        "methods": {
            "dense": {"patch": None},
            "sol_tau1_extra0": {"node": "BlockSparseAttention", "tau": 1.0, "extra_tokens": 0},
            "sol_tau1_extra256": {"node": "BlockSparseAttention", "tau": 1.0, "extra_tokens": 256},
            "sol_tau13_extra256": {"node": "BlockSparseAttention", "tau": 1.3, "extra_tokens": 256},
            "spark_topk10": {"node": "MiniMaxH3SparkAttentionSM120", "topk_ratio": 0.1},
        },
        "shared_sparse": {
            "dense_evaluations": 4, "sol_start_percent": 0.19,
            "dense_blocks": [0], "min_tokens": 4096,
            "sink_conditioning": "exact_kv_and_rows", "end_percent": 1.0,
        },
        "settings": {
            "seed": SEED, "steps": STEPS, "sampler": "res_multistep", "scheduler": "simple",
            "width": WIDTH, "height": HEIGHT, "requested_frames": REQUESTED_FRAMES,
            "model_aligned_frames": MODEL_FRAMES, "fps": FPS, "duration_seconds": 5.0,
            "turbo_lora": False, "timing": "SamplerCustomAdvanced only",
            "pairing": "all methods for a case run on the same GPU; per-case method order rotates",
            "warmup": "one excluded 5-step run per method on every GPU",
        },
        "quality_comparison": [
            "sol_tau1_extra0",
            "sol_tau1_extra256",
            "sol_tau13_extra256",
            "spark_topk10",
        ],
        "cases": cases(), "conditioning": converted,
        "source_protocol": str(SOURCE_PROTOCOL),
        "revisions": {"comfyui": base.git_revision(COMFY), "spark_h3": base.git_revision(REPO)},
    })


def assigned_cases(gpu):
    return [case for case in cases() if (case["index"] - 1) % len(GPUS) == gpu]


def method_order(case_index):
    shift = (case_index - 1) % len(METHODS)
    return METHODS[shift:] + METHODS[:shift]


def run_gpu(gpu):
    port = PORTS[gpu]
    gpu_cases = assigned_cases(gpu)
    for method in METHODS:
        target = ROOT / "warmup" / method / f"gpu{gpu}.safetensors"
        result = base.queue_and_wait(port, denoise_graph(method, gpu_cases[0], target, steps=5), SAMPLER_NODE)
        result.pop("history")
        base.write_json(ROOT / "warmup" / method / f"gpu{gpu}.json", {
            **result, "method": method, "gpu": gpu, "steps": 5,
            "latent_path": str(target), "latent_sha256": base.sha256(target),
        })
    records = []
    for case in gpu_cases:
        order = method_order(case["index"])
        for position, method in enumerate(order, 1):
            target = latent_path(method, case)
            result = base.queue_and_wait(port, denoise_graph(method, case, target), SAMPLER_NODE)
            result.pop("history")
            row = {
                **result, "method": method, "gpu": gpu, "execution_order": position,
                "case": case["index"], "sample_id": case["sample_id"],
                "prompt_sha256": case["prompt_sha256"], "seed": SEED, "steps": STEPS,
                "width": WIDTH, "height": HEIGHT, "requested_frames": REQUESTED_FRAMES,
                "model_frames": MODEL_FRAMES, "latent_path": str(target),
                "latent_sha256": base.sha256(target), "latent_bytes": target.stat().st_size,
            }
            records.append(row)
            base.write_json(ROOT / "records" / method / f"case_{case['index']:02}.json", row)
    return records


def summarize(records):
    methods = {}
    dense_mean = statistics.fmean(r["sampler_seconds"] for r in records if r["method"] == "dense")
    for method in METHODS:
        rows = sorted((r for r in records if r["method"] == method), key=lambda r: r["case"])
        values = [r["sampler_seconds"] for r in rows]
        methods[method] = {
            "sampler_seconds": values, "mean_seconds": statistics.fmean(values),
            "median_seconds": statistics.median(values),
            "speedup_vs_dense": dense_mean / statistics.fmean(values), "records": rows,
        }
    return {"status": "complete", "methods": methods}


def denoise():
    if not ROOT.exists():
        prepare()
    base.ROOT, base.OUT = ROOT, OUT
    servers = []
    try:
        for gpu, port in zip(GPUS, PORTS, strict=True):
            process, log = base.start_server(f"gpu{gpu}", gpu, port, OUT)
            servers.append((f"gpu{gpu}", process, log))
        for gpu, port in zip(GPUS, PORTS, strict=True):
            base.write_json(ROOT / f"system_gpu{gpu}.json", base.wait_server(port))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            records = [row for rows in executor.map(run_gpu, GPUS) for row in rows]
        base.write_json(ROOT / "denoise_summary.json", summarize(records))
    finally:
        base.stop_servers(servers)


def decode_method(method):
    records = []
    port = DECODE_PORTS[method]
    for case in cases():
        pattern = f"{case['index']:02}_{case['sample_id']}_*.mp4"
        matches = sorted((OUT / method).glob(pattern))
        if not matches:
            result = base.queue_and_wait(port, decode_graph(method, case))
            result.pop("history")
            matches = sorted((OUT / method).glob(pattern))
        else:
            result = {"reused_existing": True}
        if len(matches) != 1:
            raise RuntimeError(f"expected one decode for {method}/{pattern}, got {matches}")
        path = matches[0]
        info = base.inspect_video(path)
        expected = {"frames": REQUESTED_FRAMES, "width": WIDTH, "height": HEIGHT,
                    "average_rate": FPS, "audio_streams": 1}
        if info != expected:
            raise RuntimeError(f"invalid video {path}: {info}")
        row = {
            **result, "method": method, "case": case["index"], "sample_id": case["sample_id"],
            "video_path": str(path), "video_sha256": base.sha256(path),
            "video_bytes": path.stat().st_size, "video": info,
        }
        records.append(row)
        base.write_json(ROOT / "decode" / method / f"case_{case['index']:02}.json", row)
    return records


def decode():
    base.ROOT, base.OUT = ROOT, OUT
    servers = []
    try:
        for method_index, method in enumerate(QUALITY_METHODS):
            gpu = GPUS[method_index % len(GPUS)]
            label = f"decoder_{method}"
            for section in ("user", "temp", "input"):
                (ROOT / section / label).mkdir(parents=True, exist_ok=True)
            process, log = base.start_server(label, gpu, DECODE_PORTS[method], OUT)
            servers.append((label, process, log))
        for method in QUALITY_METHODS:
            base.wait_server(DECODE_PORTS[method])
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            by_method = dict(zip(QUALITY_METHODS, executor.map(decode_method, QUALITY_METHODS), strict=True))
        base.write_json(ROOT / "decode_summary.json", {
            "status": "complete", "records": [r for method in QUALITY_METHODS for r in by_method[method]],
        })
    finally:
        base.stop_servers(servers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "denoise", "decode", "all"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "denoise":
        denoise()
    elif args.command == "decode":
        decode()
    else:
        denoise(); decode()
