"""Compiled Table-4-aligned Spark TopK30 ablation on 25 or 50 prompts (GPU2-5).

Set H3_TOPK30_ALL50=1 to extend the completed first-25 run to 50 prompts.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import diffusers_spark_reweight_components_50prompt_20260926 as shared


route = shared.route
ALL50 = os.environ.get("H3_TOPK30_ALL50") == "1"
NAME = ("diffusers_spark_topk30_50prompt_10s768p_20260928" if ALL50 else
        "diffusers_spark_topk30_25prompt_10s768p_20260928")
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SOURCE25_NAME = "diffusers_spark_topk30_25prompt_10s768p_20260928"
SOURCE25_ROOT = Path("/autodl-fs/data/h3_experiments") / SOURCE25_NAME
SOURCE25_OUT = Path("/autodl-fs/data/h3_outputs") / SOURCE25_NAME
SAMPLES = shared.SAMPLES
DENSE = shared.DENSE
COND = shared.COND
METHODS = ("spark_topk30",)
GPUS = (2, 3, 4, 5)
CASES = tuple(range(1, 51 if ALL50 else 26))
SCRIPT_PATH = Path(__file__).resolve()


def config(method):
    from h3_sparse_attention import H3SparseAttentionConfig

    if method not in METHODS:
        raise ValueError(method)
    return H3SparseAttentionConfig.spark(
        20, sol_route_topk_ratio=0.3, sol_log_density=False,
        sol_video_tail_mode="dense", sol_tail_granularity="query",
        sol_global_anchor_dtype="bfloat16",
        sol_reweight_summary_math="tensorcore", sol_reweight_logmass_key="stored",
        sol_reweight_components="full",
    )


def cases():
    return route.base.pipeline().load_cases(
        SAMPLES, list(CASES), expected_indices=tuple(range(1, 51))
    )


def configure():
    route.ROOT, route.OUT = ROOT, OUT
    route.SAMPLES, route.CASES = SAMPLES, CASES
    route.METHODS, route.GPUS = METHODS, GPUS
    route.EXECUTIONS = {METHODS[0]: "threshold"}
    route.DENSE_OUT = ROOT / "dense_reference"
    route.spark_config = config
    route.cases = cases
    route.base.FRAMES = 240
    route.base.INTERNAL_FRAMES = 244
    shared.ROOT, shared.OUT = ROOT, OUT
    shared.METHODS, shared.GPUS, shared.CASES = METHODS, GPUS, CASES
    shared.cases, shared.config = cases, config


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError("Refusing to overwrite existing experiment")
    selected = cases()
    dense = json.loads(DENSE.read_text())
    by_id = {row["sample_id"]: row for row in dense["records"]}
    if len(selected) != len(CASES) or len(by_id) != 50:
        raise ValueError("Expected selected cases and 50 dense references")
    dense_rows = []
    for case in selected:
        row = dict(by_id[case["sample_id"]])
        if row["prompt_sha256"] != case["prompt_sha256"] or not Path(row["output_path"]).is_file():
            raise ValueError(f"Dense reference mismatch: {case['sample_id']}")
        row["index"] = case["index"]
        dense_rows.append(row)
    for folder in (ROOT / "records" / METHODS[0], ROOT / "warmup" / METHODS[0],
                   ROOT / "decode_records" / METHODS[0], ROOT / "dense_reference",
                   OUT / METHODS[0] / "latents", OUT / METHODS[0] / "videos",
                   OUT / METHODS[0] / "denoise_records"):
        folder.mkdir(parents=True, exist_ok=True)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        (OUT / name).symlink_to(COND / name, target_is_directory=name == "conditioning_cache")
    if ALL50:
        (ROOT / "vbench.py").symlink_to(SOURCE25_ROOT / "vbench.py")
    route.base.write(ROOT / "dense_reference" / "generation_manifest.json", dict(
        schema_version=1, status="passed", method="dense", sample_count=len(CASES),
        settings=dense.get("settings"), records=dense_rows, source_manifest=str(DENSE)))
    if ALL50:
        source_protocol = json.loads((SOURCE25_ROOT / "protocol.json").read_text())
        if (source_protocol.get("torch_compile") is not True
                or source_protocol["configs"][METHODS[0]] !=
                json.loads(json.dumps(dataclasses.asdict(config(METHODS[0]))))):
            raise ValueError("first-25 TopK30 source settings mismatch")
        for case in selected[:25]:
            index = case["index"]
            old = json.loads((SOURCE25_ROOT / "records" / METHODS[0] /
                              f"case_{index:02}.json").read_text())
            decoded = json.loads((SOURCE25_ROOT / "decode_records" / METHODS[0] /
                                  f"case_{index:02}.json").read_text())
            if (old["sample_id"] != case["sample_id"] or
                    old["prompt_sha256"] != case["prompt_sha256"] or
                    old.get("torch_compile") is not True or
                    decoded["sample_id"] != case["sample_id"]):
                raise ValueError(f"first-25 TopK30 source mismatch: {index}")
            source_latent = Path(old["latent_path"])
            source_video = Path(decoded["output_path"])
            source_provenance = (source_latent.parent.parent / "denoise_records" /
                                 f"{source_latent.stem}.json")
            provenance = json.loads(source_provenance.read_text())
            if (provenance.get("torch_compile") is not True or
                    provenance.get("compile_wrapped_blocks") != 50 or
                    provenance.get("latent_sha256") != old["latent_sha256"]):
                raise ValueError(f"first-25 compile provenance mismatch: {index}")
            latent = OUT / METHODS[0] / "latents" / source_latent.name
            video = OUT / METHODS[0] / "videos" / source_video.name
            local_provenance = OUT / METHODS[0] / "denoise_records" / source_provenance.name
            for source, target in ((source_latent, latent), (source_video, video),
                                   (source_provenance, local_provenance)):
                os.link(source, target)
            route.base.write(ROOT / "records" / METHODS[0] / f"case_{index:02}.json",
                             dict(old, latent_path=str(latent), reused_from=str(SOURCE25_ROOT)))
            route.base.write(ROOT / "decode_records" / METHODS[0] / f"case_{index:02}.json",
                             dict(decoded, output_path=str(video), reused_from=str(SOURCE25_ROOT)))
    route.base.write(ROOT / "protocol.json", dict(
        name=NAME, status="prepared", pipeline="native PyTorch/Diffusers",
        dataset=f"Table 4 core-five first {len(CASES)} prompts", cases=selected, methods=METHODS,
        configs={METHODS[0]: dataclasses.asdict(config(METHODS[0]))},
        dense_reference_manifest=str(DENSE), conditioning_source=str(COND),
        seed=42, requested_steps=20, actual_evaluations=19, torch_compile=True,
        frames=240, resolution="1344x768", gpus=GPUS,
        route_execution="threshold", topk_ratio=config(METHODS[0]).sol_route_topk_ratio,
        tail_granularity=config(METHODS[0]).sol_tail_granularity,
        warmup="one excluded full denoise per GPU",
        reused_first_25_from=str(SOURCE25_ROOT) if ALL50 else None,
        metrics="paired PSNR/SSIM/LPIPS against Table 4 dense and assigned VBench core-five",
    ))
    route.base.pipeline().configure_denoise_workflow(route.parse_args("denoise"), selected)


def spawn(stage):
    jobs = []
    for rank, gpu in enumerate(GPUS):
        if stage == "decode_worker":
            marker = ROOT / f"decode_gpu{gpu}.json"
            assigned = cases()[rank::len(GPUS)]
            if (marker.is_file()
                    and json.loads(marker.read_text()).get("status") == "complete"
                    and all((ROOT / "decode_records" / METHODS[0] /
                             f"case_{case['index']:02}.json").is_file()
                            for case in assigned)):
                continue
        log = (ROOT / f"{stage}_gpu{gpu}.log").open("a")
        proc = subprocess.Popen(
            [str(route.base.PYTHON), str(SCRIPT_PATH), stage, str(rank)],
            env={**os.environ, **route.base.ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log, stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, proc, log))
    codes = [(gpu, proc.wait()) for gpu, proc, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"{stage} failed: {codes}")


def aggregate():
    rows = [json.loads((ROOT / "records" / METHODS[0] / f"case_{case:02}.json").read_text())
            for case in CASES]
    seconds = [row["denoise_seconds"] for row in rows]
    route.base.write(ROOT / "results.json", dict(
        status="runtime_complete", method=METHODS[0], sample_count=len(CASES),
        mean_seconds=statistics.mean(seconds), median_seconds=statistics.median(seconds),
        min_seconds=min(seconds), max_seconds=max(seconds),
        reused=sum("reused_from" in row for row in rows),
        newly_measured=sum("reused_from" not in row for row in rows)))


def manifests():
    rows = []
    for case in cases():
        record = json.loads((ROOT / "decode_records" / METHODS[0] /
                             f"case_{case['index']:02}.json").read_text())
        rows.append(dict(index=case["index"], sample_id=case["sample_id"],
                         prompt_sha256=case["prompt_sha256"],
                         evaluation_prompt=case["original_prompt"],
                         vbench_dimensions=case["vbench_dimensions"],
                         output_path=record["output_path"], sha256=record["sha256"],
                         video=dict(frames=240, width=1344, height=768, fps=24.0)))
    route.base.write(OUT / METHODS[0] / "generation_manifest.json", dict(
        schema_version=1, status="passed", method=METHODS[0], sample_count=len(rows),
        settings=dict(seed=42, steps=20, frames=240, height=768, width=1344, fps=24),
        records=rows))


def quality():
    method = METHODS[0]
    work = ROOT / "quality" / method
    cfg = ROOT / "quality" / f"{method}_config.json"
    route.base.write(cfg, dict(
        work_dir=str(work), reference_manifest=str(ROOT / "dense_reference" / "generation_manifest.json"),
        candidate_manifest=str(OUT / method / "generation_manifest.json"),
        method=method, cases=list(CASES), workers=4))
    subprocess.run([str(route.base.PYTHON), str(route.QUALITY_SCRIPT), "run", "--config", str(cfg)],
                   check=True, env={**os.environ, **route.base.ENV,
                                    "CUDA_VISIBLE_DEVICES": "2,3,4,5", "H3_NUM_GPUS": "4"})
    result = json.loads((ROOT / "results.json").read_text())
    result["quality"] = json.loads((work / f"{method}_quality_results.json").read_text())["summary"]
    result["status"] = "quality_complete"
    route.base.write(ROOT / "results.json", result)


def vbench():
    spec = importlib.util.spec_from_file_location("spark25_vbench", ROOT / "vbench.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.prepare()
    queue = Path("/autodl-fs/data/h3_repos/MiniMax-H3-Experiments/scripts/h3_vbench_queue.py")
    subprocess.run([str(route.base.PYTHON), str(queue), "run", "--experiment", str(ROOT),
                    "--gpus", *(str(gpu) for gpu in GPUS)], check=True)
    scores = json.loads((ROOT / "vbench" / "results.json").read_text())
    if scores.get("status") != "complete" or any(
        count != len(CASES) for count in scores["counts"][METHODS[0]].values()
    ):
        raise RuntimeError("incomplete TopK30 VBench scores")
    result = json.loads((ROOT / "results.json").read_text())
    key = f"scores_percent_all_{len(CASES)}"
    result[f"vbench_{key}"] = scores[key][METHODS[0]]
    result["status"] = "complete"
    route.base.write(ROOT / "results.json", result)


def run():
    if not ROOT.exists() and not OUT.exists():
        prepare()
    elif not (ROOT / "protocol.json").is_file():
        raise RuntimeError("Experiment directory exists without protocol")
    spawn("generate_worker")
    aggregate()
    spawn("decode_worker")
    manifests()
    quality()
    vbench()


if __name__ == "__main__":
    configure()
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command in ("generate_worker", "decode_worker"):
        getattr(shared, command)(int(sys.argv[2]))
    elif command in ("prepare", "run", "aggregate", "manifests", "quality", "vbench"):
        globals()[command]()
    else:
        raise ValueError(command)
