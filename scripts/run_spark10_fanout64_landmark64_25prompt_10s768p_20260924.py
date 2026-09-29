#!/usr/bin/env python3
"""Spark-H3 10% with fanout=64 and 64 landmarks on 25 VBench prompts."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time


NAME = "spark10_fanout64_landmark64_25prompt_10s768p_seed42_20260924"
IMPL = Path(__file__).resolve().parents[1]
BENCH = IMPL.parent / "MiniMax-H3-Benchmark"
SCRIPTS = BENCH / "scripts"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
REPORT = IMPL / "reports" / NAME
SAMPLES = BENCH / "vbench_core5_percent_subsets/10pct/samples.json"
HISTORICAL_DENSE = Path(
    "/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913/"
    "dense/generation_manifest.json"
)
HISTORICAL_ROOT = HISTORICAL_DENSE.parent.parent
FV_EVAL = IMPL.parent / "FastVideo/examples/inference/eval"
MODEL_WEIGHTS = Path(
    "/autodl-fs/data/h3_experiments/unsplit_f16_fp16_10prompt_20260915/"
    "vbench/model_weights.json"
)
PYTHON = Path("/root/miniconda3/bin/python")
ARM = "spark10_f64_l64"
ARMS = ("dense", ARM)
LABELS = {"dense": "Dense", ARM: "Spark-H3-10pct fanout64 landmark64"}
CASES = tuple(range(1, 26))
GPUS = (0, 1, 2, 3)
SEED, STEPS, FRAMES, FPS = 42, 20, 240, 24
HEIGHT, WIDTH, INTERNAL_FRAMES = 768, 1344, 243
ENV = dict(
    HF_HUB_OFFLINE="1",
    OMP_NUM_THREADS="4",
    TORCHINDUCTOR_COMPILE_THREADS="4",
    PYTHONUNBUFFERED="1",
    PYTHONDONTWRITEBYTECODE="1",
    FLASHINFER_CUDA_ARCH_LIST="12.0",
    H3_SOL_LAYOUT_FAST="1",
    H3_METRIC_FACTOR="cholesky",
    H3_LMV2_COS_PRECISION="fp16",
    H3_LMV2_FUSED_NODE="1",
    H3_LMV2_GROUP1_FAST="1",
    H3_LMV2_COS_FAST="1",
    H3_LMV2_SMALL_PROXY_FAST="1",
    H3_LMV2_FP8_FEATURES="0",
    H3_IMPL_REPO=str(IMPL),
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pipeline():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import _impl_bootstrap  # noqa: F401
    return importlib.import_module("minimax_h3_vbench_4gpu_pipeline")


def config():
    pipeline()
    from h3_sparse_attention import H3SparseAttentionConfig
    return H3SparseAttentionConfig.spark(
        STEPS,
        landmark_tree_v2_fanout=64,
        landmark_tree_v2_landmark_count=64,
        sol_log_density=True,
    )


def parse_args(command):
    return pipeline().build_parser().parse_args([
        command,
        "--samples", str(SAMPLES),
        "--output", str(OUT),
        "--method", "dense",
        "--steps", str(STEPS),
        "--frames", str(FRAMES),
        "--height", str(HEIGHT),
        "--width", str(WIDTH),
        "--workers", "1",
        "--no-torch-compile",
    ])


def dense_manifest(cases):
    historical = read(HISTORICAL_DENSE)
    by_id = {row["sample_id"]: row for row in historical["records"]}
    records = []
    for case in cases:
        source = by_id[case["sample_id"]]
        assert source["prompt_sha256"] == case["prompt_sha256"]
        assert sha(source["output_path"]) == source["sha256"]
        records.append(dict(
            index=case["index"],
            sample_id=case["sample_id"],
            prompt_sha256=case["prompt_sha256"],
            evaluation_prompt=case["original_prompt"],
            vbench_dimensions=case["vbench_dimensions"],
            output_path=source["output_path"],
            sha256=source["sha256"],
            video=dict(frames=FRAMES, width=WIDTH, height=HEIGHT, fps=float(FPS)),
            reused_from=dict(manifest=str(HISTORICAL_DENSE), historical_index=source["index"]),
        ))
    manifest = dict(
        schema_version=1, status="passed", method="dense", sample_count=len(records),
        settings=dict(seed=SEED, steps=STEPS, frames=FRAMES,
                      height=HEIGHT, width=WIDTH, fps=FPS),
        records=records,
    )
    write(OUT / "dense/generation_manifest.json", manifest)
    return manifest


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config()
    assert cfg.total_evaluations == 19
    assert cfg.dense_evaluations == 4
    assert cfg.sol_dense_layers == 1
    assert cfg.landmark_tree_v2_fanout == 64
    assert cfg.landmark_tree_v2_landmark_count == 64
    p = pipeline()
    cases = p.load_cases(SAMPLES, list(CASES), expected_indices=CASES)
    dense = dense_manifest(cases)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        target = OUT / name
        source = HISTORICAL_ROOT / name
        if target.exists() and not target.is_symlink():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        if not target.exists():
            target.symlink_to(source, target_is_directory=source.is_dir())
    snapshot = ROOT / "snapshot"
    if not snapshot.exists():
        for name in ("h3_sparse_attention", "sol_attn"):
            shutil.copytree(IMPL / name, snapshot / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, ROOT / "runner_source.py")
    write(ROOT / "source_manifest.json", {
        str(path.relative_to(snapshot)): sha(path)
        for path in sorted(snapshot.rglob("*.py"))
    })
    write(ROOT / "protocol.json", dict(
        name=NAME,
        purpose="Evaluate stock Spark-H3 10% with fanout 64 and 64 landmarks",
        dataset="VBench core-five 10% subset (25 prompts)",
        cases=cases,
        arms=list(ARMS), labels=LABELS,
        configuration=dataclasses.asdict(cfg),
        seed=SEED, requested_steps=STEPS, transformer_evaluations=STEPS - 1,
        frames=FRAMES, internal_vae_aligned_frames=INTERNAL_FRAMES, fps=FPS,
        height=HEIGHT, width=WIDTH, gpus=list(GPUS), torch_compile=False,
        policy=dict(
            dense_warmup="first 4 of 19 transformer evaluations (20%)",
            layer_0="dense after warmup",
            remaining_sparse_calls="15 evaluations x 49 layers = 735",
            retained_ratio=0.10,
            virtual_query="target-189 reweight (Spark preset)",
            fanout=64, landmark_count=64,
        ),
        paired_metrics="PSNR pooled RGB MSE; SSIM Gaussian 11x11 sigma 1.5; LPIPS AlexNet v0.1",
        vbench="all five core dimensions on every video",
        dense_reference=dense,
        implementation=str(IMPL), environment=ENV,
    ))
    # Fill the shared text-conditioning cache before concurrent workers.
    p.configure_denoise_workflow(parse_args("denoise"), cases)
    write(ROOT / "status.json", dict(stage="prepared"))
    print("PREPARED", ROOT, flush=True)


def generate_worker(rank):
    rank = int(rank)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = pipeline()
    from h3_sparse_attention import install_h3_sparse_attention
    cases = p.load_cases(SAMPLES, list(CASES), expected_indices=CASES)[rank::len(GPUS)]
    args = parse_args("denoise")
    workflow, states = p.configure_denoise_workflow(args, cases)
    pipe, manager, acceleration, placement = p.load_denoiser(args, workflow)
    cfg = config()
    try:
        for case, original_state in zip(cases, states, strict=True):
            target = OUT / ARM / "latents" / f"{p.case_stem(case)}.pt"
            record = ROOT / "records" / f"{ARM}_{case['index']:02}.json"
            if record.exists() and target.exists() and sha(target) == read(record)["latent_sha256"]:
                print("REUSE", ARM, case["index"], flush=True)
                continue
            state = p.clone_state(original_state)
            state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with install_h3_sparse_attention(pipe.transformer, cfg) as plugin, torch.inference_mode():
                result = pipe(
                    state=state,
                    num_frames=FRAMES,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=STEPS,
                    generator=torch.Generator(device="cpu").manual_seed(SEED),
                    output=["latents", "audio_latents"],
                )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                summary = plugin.summary()
            calls = summary["processor_calls"]
            assert summary["completed_evaluations"] == 19, summary
            assert summary["dense_evaluations"] == 4, summary
            assert calls.get("dense:warmup") == 200, calls
            assert calls.get("dense:dense_layer") == 15, calls
            assert calls.get("sparse:sol") == 735, calls
            assert calls.get("sol_landmark_preprocess_calls") == 735, calls
            payload = {key: result[key].detach().cpu().contiguous()
                       for key in ("latents", "audio_latents")}
            assert all(torch.isfinite(value).all() for value in payload.values())
            p.atomic_torch_save(payload, target)
            write(record, dict(
                arm=ARM, case=case["index"], sample_id=case["sample_id"],
                prompt_sha256=case["prompt_sha256"], config=dataclasses.asdict(cfg),
                attention_summary=summary, latent_path=str(target),
                latent_sha256=sha(target), seed=SEED, steps=STEPS, frames=FRAMES,
                height=HEIGHT, width=WIDTH, denoise_seconds=seconds,
                peak_memory_bytes=torch.cuda.max_memory_allocated(), placement=placement,
                worker_rank=rank,
            ))
            print("GENERATED", ARM, case["index"], f"{seconds:.2f}s", flush=True)
            del result, payload, state
        write(ROOT / f"worker_gpu{rank}.json", dict(status="complete", rank=rank))
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def decode_worker(rank):
    rank = int(rank)
    import numpy as np
    import torch
    torch.set_num_threads(4)
    if str(FV_EVAL) not in sys.path:
        sys.path.insert(0, str(FV_EVAL))
    from fasth3_vbench_archive import archive, file_sha
    p = pipeline()
    cases = p.load_cases(SAMPLES, list(CASES), expected_indices=CASES)[rank::len(GPUS)]
    pipe, manager, acceleration = p.load_decoder(parse_args("decode"))
    try:
        for case in cases:
            stem = p.case_stem(case)
            latent = OUT / ARM / "latents" / f"{stem}.pt"
            target = OUT / ARM / "videos" / f"{stem}.mkv"
            record = ROOT / "decode_records" / f"{ARM}_{case['index']:02}.json"
            if record.exists() and target.exists() and file_sha(target) == read(record)["sha256"]:
                print("REUSE DECODE", ARM, case["index"], flush=True)
                continue
            payload = torch.load(latent, map_location="cpu", weights_only=True)
            with torch.inference_mode():
                result = pipe(
                    latents=payload["latents"].to("cuda"),
                    audio_latents=payload["audio_latents"].to("cuda"),
                    output_type="np", output=["videos", "audio", "sampling_rate"],
                )
            floats = result["videos"][0][:FRAMES]
            assert np.isfinite(floats).all()
            frames = np.clip(np.round(floats * 255), 0, 255).astype(np.uint8)
            assert frames.shape == (FRAMES, HEIGHT, WIDTH, 3)
            meta = archive(frames, result["audio"][0], int(result["sampling_rate"]),
                           target, fps=FPS, threads=8)
            write(record, dict(arm=ARM, case=case["index"], sample_id=case["sample_id"],
                               output_path=str(target), sha256=file_sha(target), archive=meta))
            print("DECODED", ARM, case["index"], flush=True)
            del payload, result, floats, frames
        write(ROOT / f"decode_gpu{rank}.json", dict(status="complete", rank=rank))
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def candidate_manifest():
    p = pipeline()
    cases = p.load_cases(SAMPLES, list(CASES), expected_indices=CASES)
    records = []
    for case in cases:
        denoise = read(ROOT / "records" / f"{ARM}_{case['index']:02}.json")
        decode = read(ROOT / "decode_records" / f"{ARM}_{case['index']:02}.json")
        assert sha(denoise["latent_path"]) == denoise["latent_sha256"]
        assert sha(decode["output_path"]) == decode["sha256"]
        records.append(dict(
            index=case["index"], sample_id=case["sample_id"],
            prompt_sha256=case["prompt_sha256"], evaluation_prompt=case["original_prompt"],
            vbench_dimensions=case["vbench_dimensions"], output_path=decode["output_path"],
            sha256=decode["sha256"],
            video=dict(frames=FRAMES, width=WIDTH, height=HEIGHT, fps=float(FPS)),
        ))
    write(OUT / ARM / "generation_manifest.json", dict(
        schema_version=1, status="passed", method=ARM, sample_count=len(records),
        settings=dict(seed=SEED, steps=STEPS, frames=FRAMES,
                      height=HEIGHT, width=WIDTH, fps=FPS), records=records,
    ))


def quality():
    cfg = dict(
        work_dir=str(ROOT / "quality" / ARM),
        reference_manifest=str(OUT / "dense/generation_manifest.json"),
        candidate_manifest=str(OUT / ARM / "generation_manifest.json"),
        method=ARM, cases=list(CASES), workers=len(GPUS),
    )
    config_path = ROOT / "quality/config.json"
    write(config_path, cfg)
    subprocess.run([str(PYTHON), str(SCRIPTS / "h3_quality_video_pair.py"),
                    "run", "--config", str(config_path)], check=True,
                   env={**os.environ, **ENV, "H3_NUM_GPUS": str(len(GPUS))})
    result = read(ROOT / "quality" / ARM / f"{ARM}_quality_results.json")
    write(ROOT / "quality_results.json", result)


def vbench_setup():
    (ROOT / "vbench_code").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPTS / "vbench_core5_cached_module.py", ROOT / "vbench.py")
    shutil.copy2(SCRIPTS / "h3_vbench_queue.py", ROOT / "h3_vbench_queue.py")
    shutil.copy2(SCRIPTS / "_impl_bootstrap.py", ROOT / "_impl_bootstrap.py")
    for name in ("score_vbench20pct_768p10s.py", "score_vbench20pct_aesthetic.py"):
        text = (SCRIPTS / name).read_text().replace(
            "REPO = Path(__file__).resolve().parents[1]", f"REPO = Path({str(BENCH)!r})", 1)
        (ROOT / "vbench_code" / name).write_text(text)
    write(ROOT / "vbench/module_config.json", dict(
        cases=list(CASES), labels=LABELS,
        video_sources=[dict(method=arm, **{"from": "manifest"},
                            path=str(OUT / arm / "generation_manifest.json")) for arm in ARMS],
        historical_scores=[], model_weights=str(MODEL_WEIGHTS),
    ))
    subprocess.run([str(PYTHON), "-c",
                    "import sys; sys.path.insert(0, '.'); import vbench; vbench.prepare()"],
                   cwd=ROOT, check=True, env={**os.environ, **ENV})


def vbench():
    subprocess.run([str(PYTHON), str(ROOT / "h3_vbench_queue.py"), "run",
                    "--experiment", str(ROOT), "--gpus", *map(str, GPUS)],
                   check=True, env={**os.environ, **ENV})


def report():
    quality_result = read(ROOT / "quality_results.json")["summary"]
    vbench_result = read(ROOT / "vbench/results.json")["scores_percent"]
    records = [read(ROOT / "records" / f"{ARM}_{case:02}.json") for case in CASES]
    times = [row["denoise_seconds"] for row in records]
    warmed_times = [row["denoise_seconds"] for row in records
                    if row["case"] not in {1, 2, 3, 4}]
    dimensions = ("subject_consistency", "background_consistency", "motion_smoothness",
                  "imaging_quality", "aesthetic_quality")
    result = dict(
        status="complete", quality=quality_result, vbench_percent=vbench_result,
        denoise_seconds=dict(mean=statistics.mean(times), median=statistics.median(times),
                             warmed_mean=statistics.mean(warmed_times),
                             warmed_median=statistics.median(warmed_times), values=times),
        peak_memory_gib=[row["peak_memory_bytes"] / 2**30 for row in records],
        attention_audit=[row["attention_summary"] for row in records],
    )
    write(ROOT / "results.json", result)
    q = quality_result
    lines = [
        f"# {NAME}", "",
        "VBench core-five 25 prompts; seed 42; 10 s, 240 frames at 24 fps; "
        "1344x768; 20 requested steps (19 DiT evaluations). Spark-H3 retains 10%, "
        "uses four dense warmup evaluations and keeps layer 0 dense. LMv2 uses "
        "fanout 64 and 64 midpoint landmarks.", "",
        "## Paired RGB metrics against Dense", "",
        "| PSNR ↑ | SSIM ↑ | LPIPS ↓ |", "|---:|---:|---:|",
        f"| {q['psnr_db']:.2f} | {q['ssim']:.2f} | {q['lpips']:.2f} |", "",
        "## VBench core five (percent)", "",
        "| Arm | Subject ↑ | Background ↑ | Motion ↑ | Imaging ↑ | Aesthetic ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        lines.append("| " + LABELS[arm] + " | " +
                     " | ".join(f"{vbench_result[arm][dim]:.2f}" for dim in dimensions) + " |")
    lines += ["", "## Denoise runtime", "",
              f"Mean {statistics.mean(times):.2f} s; median {statistics.median(times):.2f} s. "
              f"Excluding the first case on each GPU: mean {statistics.mean(warmed_times):.2f} s; "
              f"median {statistics.median(warmed_times):.2f} s.", ""]
    text = "\n".join(lines)
    ROOT.joinpath("REPORT.md").write_text(text)
    REPORT.mkdir(parents=True, exist_ok=True)
    REPORT.joinpath("REPORT.md").write_text(text)
    for name in ("protocol.json", "source_manifest.json", "quality_results.json", "results.json"):
        shutil.copy2(ROOT / name, REPORT / name)
    shutil.copy2(ROOT / "vbench/results.json", REPORT / "vbench_results.json")
    write(ROOT / "status.json", dict(stage="complete"))
    print(text, flush=True)


def spawn(stage):
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"{stage}_gpu{gpu}.log").open("a")
        process = subprocess.Popen([sys.executable, __file__, stage, str(rank)],
                                   env={**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                   stdout=log, stderr=subprocess.STDOUT)
        jobs.append((gpu, process, log))
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    if any(code for _, code in codes):
        raise RuntimeError(f"{stage} failed: {codes}")


def run():
    os.environ.update(ENV)
    prepare()
    write(ROOT / "status.json", dict(stage="generating"))
    spawn("generate_worker")
    write(ROOT / "status.json", dict(stage="decoding"))
    spawn("decode_worker")
    candidate_manifest()
    write(ROOT / "status.json", dict(stage="quality"))
    quality()
    write(ROOT / "status.json", dict(stage="vbench"))
    vbench_setup()
    vbench()
    report()


if __name__ == "__main__":
    os.environ.update(ENV)
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command in {"generate_worker", "decode_worker"}:
        globals()[command](int(sys.argv[2]))
    elif command == "run":
        run()
    elif command in globals():
        globals()[command]()
    else:
        raise SystemExit(f"unknown command: {command}")
