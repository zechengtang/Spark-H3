#!/usr/bin/env python3
"""Encode four T2VA prompts in ComfyUI and compare with Diffusers caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import safetensors.torch
import torch


REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parent / "ComfyUI"
PROTOCOL = Path(
    "/autodl-fs/data/h3_experiments/blog_ablation_4prompts_rerun_20260922/protocol.json"
)
DIFFUSERS_CACHE = Path(
    "/autodl-fs/data/h3_outputs/blog_ablation_4prompts_rerun_20260922/conditioning_cache"
)
ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_conditioning_compare_20260923")
PORT = 8199


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def api_json(endpoint: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{endpoint}") as response:
        return json.load(response)


def wait_server(timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            api_json("/system_stats")
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(1)
    raise TimeoutError("ComfyUI did not start")


def cases():
    return json.loads(PROTOCOL.read_text())["cases"]


def graph():
    nodes = {
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default",
            },
        }
    }
    for case in cases():
        index = int(case["index"])
        encode_id = str(index * 2)
        save_id = str(index * 2 + 1)
        nodes[encode_id] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 0], "text": case["prompt"]},
        }
        nodes[save_id] = {
            "class_type": "SaveConditioning",
            "inputs": {
                "conditioning": [encode_id, 0],
                "filename_prefix": f"case_{index:02}_{case['sample_id']}",
            },
        }
    return nodes


def queue_and_wait() -> dict:
    prompt_id = str(uuid.uuid4())
    body = json.dumps({"prompt": graph(), "prompt_id": prompt_id}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        queued = json.load(response)
    if queued.get("prompt_id") != prompt_id:
        raise RuntimeError(f"unexpected queue result: {queued}")
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        history = api_json(f"/history/{prompt_id}").get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "success" and status.get("completed"):
                return history
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False))
        time.sleep(1)
    raise TimeoutError("conditioning graph did not finish")


def load_diffusers_by_prompt():
    result = {}
    for path in DIFFUSERS_CACHE.glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        result[payload["prompt"]] = (path, payload)
    return result


def tensor_stats(left: torch.Tensor, right: torch.Tensor) -> dict:
    result = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "same_shape": left.shape == right.shape,
    }
    if left.shape != right.shape:
        return result
    result["exact_equal"] = torch.equal(left, right)
    if not left.is_floating_point():
        result["different_elements"] = int(torch.count_nonzero(left != right))
        return result
    left32, right32 = left.float(), right.float()
    difference = left32 - right32
    absolute = difference.abs().flatten()
    flat_left, flat_right = left32.flatten().double(), right32.flatten().double()
    quantiles = torch.quantile(
        absolute, torch.tensor([0.5, 0.9, 0.99, 0.999, 0.9999])
    )
    token_cosine = (
        torch.nn.functional.cosine_similarity(left32, right32, dim=-1)
        if left32.ndim == 3
        else None
    )
    result.update(
        max_abs_error=float(difference.abs().max()),
        mean_abs_error=float(difference.abs().mean()),
        rmse=float(difference.square().mean().sqrt()),
        cosine_similarity=float(torch.nn.functional.cosine_similarity(flat_left, flat_right, dim=0)),
        left_rms=float(left32.square().mean().sqrt()),
        right_rms=float(right32.square().mean().sqrt()),
        abs_error_quantiles=dict(
            zip(("p50", "p90", "p99", "p99_9", "p99_99"), map(float, quantiles), strict=True)
        ),
        equal_after_left_bfloat16_fraction=float((left.to(torch.bfloat16) == right).float().mean()),
    )
    if token_cosine is not None:
        result["per_token_cosine_mean"] = float(token_cosine.mean())
        result["per_token_cosine_min"] = float(token_cosine.min())
    return result


def compare() -> dict:
    old = load_diffusers_by_prompt()
    records = []
    for case in cases():
        pattern = f"case_{case['index']:02}_{case['sample_id']}_*.safetensors"
        matches = list((ROOT / "comfy_cache").glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"expected one Comfy cache for {pattern}, got {matches}")
        comfy_path = matches[0]
        diffusers_path, diffusers_payload = old[case["prompt"]]
        comfy = safetensors.torch.load_file(str(comfy_path), device="cpu")
        values = diffusers_payload["values"]
        record = {
            "case": case["index"],
            "sample_id": case["sample_id"],
            "prompt_sha256": case["prompt_sha256"],
            "comfy_cache": str(comfy_path),
            "comfy_cache_sha256": sha256(comfy_path),
            "diffusers_cache": str(diffusers_path),
            "diffusers_cache_sha256": sha256(diffusers_path),
            "embeddings": tensor_stats(comfy["conditioning"], values["prompt_embeds"]),
            "token_tags": tensor_stats(comfy["minimax_token_tags"], values["text_token_tags"]),
        }
        records.append(record)
    return {
        "status": "complete",
        "comparison": "ComfyUI conditioning vs existing Diffusers PipelineState conditioning",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "all_shapes_equal": all(r["embeddings"]["same_shape"] for r in records),
        "all_token_tags_exact": all(r["token_tags"].get("exact_equal") for r in records),
        "all_embeddings_exact": all(r["embeddings"].get("exact_equal") for r in records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    if args.compare_only:
        report = compare()
        write_json(ROOT / "comparison.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT}")
    for directory in ("comfy_cache", "input", "temp", "user"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    log = (ROOT / "server.log").open("w")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    command = [
        "/root/miniconda3/bin/python",
        "main.py",
        "--listen", "127.0.0.1",
        "--port", str(PORT),
        "--disable-auto-launch",
        "--disable-cuda-malloc",
        "--preview-method", "none",
        "--output-directory", str(ROOT / "comfy_cache"),
        "--temp-directory", str(ROOT / "temp"),
        "--input-directory", str(ROOT / "input"),
        "--user-directory", str(ROOT / "user"),
    ]
    process = subprocess.Popen(command, cwd=COMFY, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_server()
        write_json(ROOT / "system.json", api_json("/system_stats"))
        started = time.perf_counter()
        history = queue_and_wait()
        write_json(ROOT / "execution.json", {
            "seconds": time.perf_counter() - started,
            "status": history["status"],
        })
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
    report = compare()
    write_json(ROOT / "comparison.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
