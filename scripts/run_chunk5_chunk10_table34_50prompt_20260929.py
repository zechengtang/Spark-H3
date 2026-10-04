"""Table-3/4-aligned 50-prompt chunk-5 versus chunk-10 experiment.

The original fixed temporal-chunk implementation is used for reblocking.  A
small compatibility patch publishes the permutation-invariant global root
required by the blog's global reweight setting (levels_up=99).  Each
worker/arm performs one excluded full-generation warmup.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time


HERE = Path(__file__).resolve()
MAIN_REPO = HERE.parents[1]
SOURCE_REPO = MAIN_REPO.parent / "MiniMax-H3-Sparse"
EXPERIMENTS_REPO = MAIN_REPO.parent / "MiniMax-H3-Experiments"
BASE_RUNNER = EXPERIMENTS_REPO / "scripts/strict_splitter_benchmark.py"
COMPAT_PATCH = MAIN_REPO / "scripts/fixed_chunk_global_reweight_compat.patch"
spec = importlib.util.spec_from_file_location("table34_base", BASE_RUNNER)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

NAME = "chunk5_chunk10_table34_50prompt_20260929"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
REPORT = MAIN_REPO / "reports" / NAME
SNAP = ROOT / "snapshot"
ARMS = ("chunk5_topk10_reblock", "chunk10_topk10_reblock")
CHUNK_FRAMES = {ARMS[0]: 5, ARMS[1]: 10}
CASES = list(range(1, 51))
GPUS = tuple(range(4))

base.ROOT, base.OUT, base.SNAP = ROOT, OUT, SNAP
base.ENV.update(
    H3_GPU_COUNT="4",
    H3_TEMPORAL_MIN_FRAMES="0",
    H3_TEMPORAL_CHUNK_FRAMES="0",
)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def config(h3, arm: str):
    return h3.H3SparseAttentionConfig.sol(
        20,
        warmup_percent=20,
        sol_dense_layers=1,
        sol_tau=1.0,
        sol_log_density=False,
        sol_force_local_blocks=False,
        sol_route_topk_ratio=0.1,
        sol_route_topk_cutoff_mode="gemm_radix",
        sol_landmark_preprocess=True,
        sol_landmark_preprocess_version="v2",
        landmark_tree_v2_initial_order="flat",
        landmark_tree_v2_children=16,
        landmark_tree_v2_fanout_mode="power_of_two_fanout",
        landmark_tree_v2_landmark_mode="midpoint",
        landmark_tree_v2_landmark_count=32,
        landmark_tree_v2_group_size=1,
        landmark_tree_v2_minimum_frames=0,
        landmark_tree_v2_chunk_frames=CHUNK_FRAMES[arm],
        sol_virtual_query_levels_up=99,
        sol_virtual_query_target_blocks=None,
    )


def prepare() -> None:
    if ROOT.exists() or OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing experiment: {ROOT} / {OUT}")
    SNAP.mkdir(parents=True)
    OUT.mkdir(parents=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    source_trees = {
        "h3_sparse_attention": SOURCE_REPO / "h3_sparse_attention",
        "sol_attn": EXPERIMENTS_REPO / "sol_attn",
    }
    for name, source in source_trees.items():
        shutil.copytree(
            source,
            SNAP / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    subprocess.run(
        ["git", "apply", "--unsafe-paths", str(COMPAT_PATCH)], cwd=SNAP, check=True
    )
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(
            base.BASE / name, target_is_directory=name == "conditioning_cache"
        )

    h3, pipeline = base.imports()
    cases = pipeline.load_cases(base.SAMPLES, CASES)
    references = base.read(base.BASE / "dense/generation_manifest.json")
    for case in cases:
        reference = next(r for r in references["records"] if r["index"] == case["index"])
        assert hashlib.sha256(case["prompt"].encode()).hexdigest() == reference["prompt_sha256"]

    configs = {arm: dataclasses.asdict(config(h3, arm)) for arm in ARMS}
    protocol = {
        "status": "prepared",
        "created_utc": now(),
        "cases": cases,
        "configs": configs,
        "gpus": list(GPUS),
        "seed": 42,
        "steps": 20,
        "evaluations": 19,
        "frames": 240,
        "latent_frames": 72,
        "fps": 24,
        "height": 768,
        "width": 1344,
        "timing": (
            "Table 3/4 timing: synchronized denoising only; torch.compile on; "
            "one excluded full-generation run of each worker/arm's first prompt; "
            "loading, conditioning, latent saving, decoding, and scoring excluded. "
            "All 50 prompts are subsequently measured."
        ),
        "schedule": (
            "Four workers each process every fourth prompt for both arms. Arm order "
            "is reversed on odd GPUs to reduce order bias."
        ),
        "table34_alignment": {
            "matched": [
                "50 VBench core-5 20% prompts",
                "seed 42",
                "20-point schedule / 19 denoiser evaluations",
                "1344x768, 240 frames, 24 fps",
                "per-block torch.compile",
                "20% sparse-schedule warmup and one always-dense transformer layer",
                "Top-K10 gemm_radix routing",
                "forced local blocks disabled",
                "LMv2 cosine, flat order, group1, midpoint32, strict power-of-two fanout16",
                "global reweight with levels_up=99 and native_mean routing",
            ],
            "fixed_chunk_compatibility": (
                "The original chunk permutation is unchanged. A compatibility patch publishes "
                "the permutation-invariant global video root consumed by levels_up=99; lower "
                "hierarchy frontiers are not consumed."
            ),
        },
        "chunk_layouts": {
            ARMS[0]: {"latent_frame_lengths": [5] * 13 + [7], "tail_tokens": 640},
            ARMS[1]: {"latent_frame_lengths": [10] * 6 + [12], "tail_tokens": 192},
        },
        "environment": base.ENV,
        "source_repo": str(SOURCE_REPO),
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE_REPO, text=True
        ).strip(),
        "dense_reference": str(base.BASE / "dense/generation_manifest.json"),
        "outputs": str(OUT),
    }
    base.write(ROOT / "protocol.json", protocol)
    base.write(REPORT / "protocol.json", protocol)
    shutil.copy2(HERE, ROOT / "runner_source.py")
    manifest_paths = [
        *SNAP.rglob("*.py"),
        HERE,
        COMPAT_PATCH,
        BASE_RUNNER,
        base.SAMPLES,
        base.BASE / "dense/generation_manifest.json",
    ]
    manifest = {str(path): base.sha(path) for path in manifest_paths}
    base.write(ROOT / "source_manifest.json", manifest)
    base.write(REPORT / "source_manifest.json", manifest)


def _run_generation(h3, pipeline, pipe, case, state, cfg):
    import torch

    prepared = pipeline.clone_state(state)
    prepared.values["prompt_embeds"] = prepared.values["prompt_embeds"].cuda()
    with h3.install_h3_sparse_attention(pipe.transformer, cfg) as plugin:
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            result = pipe(
                state=prepared,
                num_frames=240,
                height=768,
                width=1344,
                num_inference_steps=20,
                generator=torch.Generator(device="cpu").manual_seed(42),
                output=["latents", "audio_latents"],
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        summary = plugin.summary()
    return result, prepared, seconds, summary


def worker(gpu: int) -> None:
    import torch

    torch.set_num_threads(4)
    my_case_ids = CASES[gpu::len(GPUS)]
    arm_order = ARMS if gpu % 2 == 0 else tuple(reversed(ARMS))
    h3, pipeline = base.imports()
    args = pipeline.build_parser().parse_args(
        [
            "denoise",
            "--samples",
            str(base.SAMPLES),
            "--output",
            str(OUT),
            "--method",
            "sol",
            "--case-indices",
            *map(str, my_case_ids),
            "--frames",
            "240",
            "--height",
            "768",
            "--width",
            "1344",
            "--steps",
            "20",
            "--workers",
            "1",
        ]
    )
    cases = pipeline.load_cases(base.SAMPLES, my_case_ids)
    workflow, states = pipeline.configure_denoise_workflow(args, cases)
    base.write(
        ROOT / f"status_gpu{gpu}.json",
        {"stage": "loading", "gpu": gpu, "cases": my_case_ids, "arm_order": arm_order},
    )
    pipe, manager, acceleration, placement = pipeline.load_denoiser(args, workflow)
    if not acceleration.config.torch_compile or len(acceleration._forward_originals) != 50:
        raise RuntimeError("Table 3/4 requires 50 per-block torch.compile wrappers")
    try:
        for arm in arm_order:
            cfg = config(h3, arm)
            first_case, first_state = cases[0], states[0]
            base.write(
                ROOT / f"status_gpu{gpu}.json",
                {"stage": "excluded_warmup", "gpu": gpu, "arm": arm, "case": first_case["index"]},
            )
            result, prepared, seconds, summary = _run_generation(
                h3, pipeline, pipe, first_case, first_state, cfg
            )
            assert summary["completed_evaluations"] == 19 and summary["dense_evaluations"] == 4
            base.write(
                ROOT / f"warmup_{arm}_gpu{gpu}.json",
                {
                    "excluded_from_timing": True,
                    "case": first_case["index"],
                    "seconds": seconds,
                    "attention_summary": summary,
                },
            )
            del result, prepared

            for case, state in zip(cases, states):
                target = ROOT / "records" / f"{arm}_{case['index']:02}.json"
                if target.exists():
                    continue
                base.write(
                    ROOT / f"status_gpu{gpu}.json",
                    {"stage": "measuring", "gpu": gpu, "arm": arm, "case": case["index"]},
                )
                result, prepared, seconds, summary = _run_generation(
                    h3, pipeline, pipe, case, state, cfg
                )
                assert summary["completed_evaluations"] == 19
                assert summary["dense_evaluations"] == 4
                calls = summary["processor_calls"]
                assert calls.get("sol_topk_calls") == 735
                assert calls.get("sol_landmark_preprocess_calls") == 735
                assert calls.get("sol_virtual_query_calls") == 735
                assert calls.get("sol_virtual_query_fused_permute_calls") == 735
                payload = {
                    key: result[key].detach().cpu().contiguous()
                    for key in ("latents", "audio_latents")
                }
                assert all(torch.isfinite(value).all() for value in payload.values())
                latent_path = OUT / "latents" / f"{arm}_{case['index']:02}.pt"
                latent_path.parent.mkdir(exist_ok=True)
                pipeline.atomic_torch_save(payload, latent_path)
                record = {
                    "arm": arm,
                    "chunk_frames": CHUNK_FRAMES[arm],
                    "case": case["index"],
                    "sample_id": case["sample_id"],
                    "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
                    "gpu": gpu,
                    "config": dataclasses.asdict(cfg),
                    "denoise_seconds": seconds,
                    "timing_included": True,
                    "attention_summary": summary,
                    "latent_path": str(latent_path),
                    "latent_sha256": base.sha(latent_path),
                    "seed": 42,
                    "steps": 20,
                    "requested_frames": 240,
                    "height": 768,
                    "width": 1344,
                    "torch_compile": True,
                    "compiled_blocks": len(acceleration._forward_originals),
                    "placement": placement,
                }
                base.write(target, record)
                print(arm, case["index"], f"{seconds:.3f}s", flush=True)
                del payload, result, prepared
        base.write(
            ROOT / f"status_gpu{gpu}.json",
            {"stage": "complete", "gpu": gpu, "cases": my_case_ids, "arm_order": arm_order},
        )
    finally:
        acceleration.remove()


def aggregate() -> None:
    rows = [
        base.read(ROOT / "records" / f"{arm}_{case:02}.json")
        for arm in ARMS
        for case in CASES
    ]
    summary = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        values = [row["denoise_seconds"] for row in arm_rows]
        summary[arm] = {
            "n": len(values),
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
            "per_prompt_seconds": {str(row["case"]): row["denoise_seconds"] for row in arm_rows},
        }
    paired = [
        next(r["denoise_seconds"] for r in rows if r["arm"] == ARMS[0] and r["case"] == case)
        - next(r["denoise_seconds"] for r in rows if r["arm"] == ARMS[1] and r["case"] == case)
        for case in CASES
    ]
    result = {
        "status": "generation_complete",
        "completed_utc": now(),
        "summary": summary,
        "paired_chunk5_minus_chunk10_seconds": {
            "mean": statistics.mean(paired),
            "median": statistics.median(paired),
            "values": dict(zip(map(str, CASES), paired)),
        },
        "rows": rows,
    }
    base.write(ROOT / "results.json", result)
    base.write(REPORT / "results.json", result)
    protocol = base.read(ROOT / "protocol.json")
    protocol.update(status="generation_complete", completed_utc=now())
    base.write(ROOT / "protocol.json", protocol)
    base.write(REPORT / "protocol.json", protocol)


def run() -> None:
    protocol = base.read(ROOT / "protocol.json")
    protocol.update(status="running", launched_utc=now())
    base.write(ROOT / "protocol.json", protocol)
    base.write(REPORT / "protocol.json", protocol)
    jobs = []
    for gpu in GPUS:
        log = (ROOT / f"gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [sys.executable, str(HERE), "worker", str(gpu)],
            env={**os.environ, **base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, process, log))
    base.write(ROOT / "pids.json", {str(gpu): process.pid for gpu, process, _ in jobs})
    codes = {}
    for gpu, process, log in jobs:
        codes[str(gpu)] = process.wait()
        log.close()
    base.write(ROOT / "exit_codes.json", codes)
    if any(codes.values()):
        protocol = base.read(ROOT / "protocol.json")
        protocol.update(status="needs_attention", exit_codes=codes, failed_utc=now())
        base.write(ROOT / "protocol.json", protocol)
        raise RuntimeError(codes)
    aggregate()


if __name__ == "__main__":
    os.environ.update(base.ENV)
    command = sys.argv[1]
    if command == "worker":
        worker(int(sys.argv[2]))
    else:
        globals()[command]()
