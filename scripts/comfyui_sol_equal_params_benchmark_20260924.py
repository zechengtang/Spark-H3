#!/usr/bin/env python3
"""Paired, denoise-only xmarre vs native ComfyUI Sol benchmark.

Each GPU evaluates one prompt with both implementations.  Method order is
counterbalanced across GPUs, so every comparison is paired on the same device
without paying four-model decode cost.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import statistics

import comfyui_sol_4way_benchmark_20260923 as base


NAME = "comfyui_sol_equal_params_blog_ablation_4prompts_20260924"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
PORTS = tuple(8230 + i for i in range(4))
METHODS = ("xmarre_sol", "comfyui_sol")


def target(method: str, case: dict) -> Path:
    return ROOT / "latents" / method / f"case_{case['index']:02}_{case['sample_id']}.safetensors"


def graph(method: str, case: dict, path: Path, steps: int = base.STEPS) -> dict:
    value = base.denoise_graph(method, case, path, steps)
    if method == "comfyui_sol":
        # Match xmarre's semantic controls: tau=1, one dense denoiser
        # evaluation, transformer blocks 0 and 1 dense, exact packed-prefix KV,
        # and no extra selected tokens.  start_percent=0.05 is one of 20 steps
        # with the simple scheduler; the server log is retained for audit.
        value["2"]["inputs"] = {
            "model": ["1", 0],
            "selection": "sol-attn",
            "selection.tau": 1.0,
            "start_percent": 0.05,
            "end_percent": 1.0,
            "dense_blocks": "0,1",
            "min_tokens": 0,
            "extra_tokens": 0,
            "sink_conditioning": "exact_kv",
            "verbose": True,
        }
    return value


def run_gpu(gpu: int, case: dict) -> list[dict]:
    port = PORTS[gpu]
    # Reverse half the devices to cancel first/second method cache and thermal
    # effects while retaining a same-GPU paired observation for every prompt.
    order = METHODS if gpu % 2 == 0 else tuple(reversed(METHODS))
    rows = []
    for method in order:
        warm = ROOT / "warmup" / f"gpu{gpu}_{method}.safetensors"
        warm_result = base.queue_and_wait(port, graph(method, case, warm, 5), base.SAMPLER_NODE)
        warm_result.pop("history")
        base.write_json(ROOT / "warmup" / f"gpu{gpu}_{method}.json", {
            **warm_result, "gpu": gpu, "method": method, "steps": 5,
            "latent_path": str(warm), "latent_sha256": base.sha256(warm),
        })

        path = target(method, case)
        result = base.queue_and_wait(port, graph(method, case, path), base.SAMPLER_NODE)
        result.pop("history")
        row = {
            **result,
            "method": method,
            "gpu": gpu,
            "execution_order": order.index(method) + 1,
            "case": case["index"],
            "sample_id": case["sample_id"],
            "prompt_sha256": case["prompt_sha256"],
            "seed": base.SEED,
            "steps": base.STEPS,
            "width": base.WIDTH,
            "height": base.HEIGHT,
            "requested_frames": base.REQUESTED_FRAMES,
            "latent_path": str(path),
            "latent_sha256": base.sha256(path),
            "latent_bytes": path.stat().st_size,
        }
        rows.append(row)
        base.write_json(ROOT / "records" / method / f"case_{case['index']:02}.json", row)
    return rows


def summarize(records: list[dict]) -> dict:
    methods = {}
    for method in METHODS:
        rows = sorted((r for r in records if r["method"] == method), key=lambda r: r["case"])
        times = [r["sampler_seconds"] for r in rows]
        methods[method] = {
            "sampler_seconds": times,
            "mean_seconds": statistics.fmean(times),
            "median_seconds": statistics.median(times),
            "records": rows,
        }
    x = methods["xmarre_sol"]["mean_seconds"]
    c = methods["comfyui_sol"]["mean_seconds"]
    paired = []
    for case in base.cases():
        by_method = {r["method"]: r for r in records if r["case"] == case["index"]}
        paired.append({
            "case": case["index"],
            "gpu": by_method["xmarre_sol"]["gpu"],
            "xmarre_seconds": by_method["xmarre_sol"]["sampler_seconds"],
            "comfyui_seconds": by_method["comfyui_sol"]["sampler_seconds"],
            "comfyui_over_xmarre_speedup": (
                by_method["xmarre_sol"]["sampler_seconds"] /
                by_method["comfyui_sol"]["sampler_seconds"]
            ),
        })
    return {
        "status": "complete",
        "methods": methods,
        "paired": paired,
        "mean_comfyui_over_xmarre_speedup": x / c,
        "faster_mean": "comfyui_sol" if c < x else "xmarre_sol",
    }


def main() -> None:
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    for path in (ROOT, OUT):
        path.mkdir(parents=True)
    for section in ("latents", "records"):
        for method in METHODS:
            (ROOT / section / method).mkdir(parents=True, exist_ok=True)
    (ROOT / "warmup").mkdir(parents=True)
    for gpu in range(4):
        for section in ("user", "temp", "input"):
            (ROOT / section / f"gpu{gpu}").mkdir(parents=True, exist_ok=True)

    protocol = {
        "name": NAME,
        "purpose": "same-GPU paired speed comparison of xmarre and native ComfyUI Sol",
        "pipeline": "ComfyUI denoise only; reused Diffusers BF16 conditioning",
        "settings": {
            "seed": base.SEED, "steps": base.STEPS, "sampler": "res_multistep",
            "scheduler": "simple", "width": base.WIDTH, "height": base.HEIGHT,
            "requested_frames": base.REQUESTED_FRAMES, "fps": base.FPS,
            "timing": "SamplerCustomAdvanced only; conditioning and VAE excluded",
            "pairing": "one prompt and both methods on each GPU; method order counterbalanced",
        },
        "parameter_alignment": {
            "shared": {"tau": 1.0, "dense_evaluations": 1, "dense_blocks": [0, 1],
                       "extra_tokens": 0, "sink": "packed prefix exact KV"},
            "xmarre_sol": {"dense_evaluations": 1, "dense_layers": 2,
                           "exact_fusion": False, "sink_mode": "prefix"},
            "comfyui_sol": {"start_percent": 0.05, "dense_blocks": "0,1",
                            "extra_tokens": 0, "sink_conditioning": "exact_kv",
                            "min_tokens": 0},
            "qualification": "semantic controls aligned; implementations and sparse kernels remain distinct",
        },
        "cases": base.cases(),
        "revisions": {"comfyui": base.git_revision(base.COMFY),
                      "xmarre": base.git_revision(base.XMARRE),
                      "spark_h3": base.git_revision(base.REPO)},
    }
    base.write_json(ROOT / "protocol.json", protocol)

    # Repoint only the imported server helper's experiment directories.  Keep
    # base.NAME unchanged so ConditioningLoader uses the already converted
    # main-experiment embeddings.
    base.ROOT = ROOT
    base.OUT = OUT
    servers = []
    try:
        for gpu, port in enumerate(PORTS):
            label = f"gpu{gpu}"
            process, log = base.start_server(label, gpu, port, OUT)
            servers.append((label, process, log))
        for gpu, port in enumerate(PORTS):
            base.write_json(ROOT / f"system_gpu{gpu}.json", base.wait_server(port))
        source_cases = base.cases()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_gpu, gpu, source_cases[gpu]) for gpu in range(4)]
            records = [row for future in futures for row in future.result()]
        base.write_json(ROOT / "summary.json", summarize(records))
    finally:
        base.stop_servers(servers)


if __name__ == "__main__":
    main()
