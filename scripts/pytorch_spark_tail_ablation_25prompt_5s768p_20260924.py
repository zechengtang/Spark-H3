#!/usr/bin/env python3
"""PyTorch 25-prompt ablation of legacy, dense, and mean-Q-pad Spark tails."""
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


NAME = "pytorch_spark_tail_ablation_25prompt_5s768p_seed42_20260924"
IMPL = Path(__file__).resolve().parents[1]
BENCH = IMPL.parent / "MiniMax-H3-Benchmark"
SCRIPTS = BENCH / "scripts"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
OUT = Path("/autodl-fs/data/h3_outputs") / NAME
SAMPLES = BENCH / "vbench_core5_percent_subsets/10pct/samples.json"
CONDITIONING_SOURCE = Path(
    "/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913"
)
METHODS = ("dense", "spark_legacy", "spark_tail_dense", "spark_tail_pad")
SPARSE_METHODS = METHODS[1:]
GPUS = (0, 1, 2, 3)
SOURCE_CASES = tuple(range(1, 26))
CASES = tuple(range(1, 16))
SEED, STEPS, FRAMES, FPS = 42, 20, 120, 24
HEIGHT, WIDTH, INTERNAL_FRAMES = 768, 1344, 124
PYTHON = Path("/root/miniconda3/bin/python")
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
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"
    )
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


def parse_args():
    return pipeline().build_parser().parse_args([
        "denoise",
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


def spark_config(method):
    pipeline()
    from h3_sparse_attention import H3SparseAttentionConfig

    common = dict(
        landmark_tree_v2_fanout=64,
        landmark_tree_v2_landmark_count=64,
        sol_log_density=False,
    )
    if method == "spark_legacy":
        return H3SparseAttentionConfig.spark(
            STEPS, sol_video_tail_mode="dense", sol_legacy_full_query=True, **common
        )
    if method == "spark_tail_dense":
        return H3SparseAttentionConfig.spark(
            STEPS, sol_video_tail_mode="dense", **common
        )
    if method == "spark_tail_pad":
        return H3SparseAttentionConfig.spark(
            STEPS, sol_video_tail_mode="pad", **common
        )
    raise ValueError(method)


def prepare():
    if ROOT.exists() or OUT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT} or {OUT}")
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True)
    for method in METHODS:
        (ROOT / "records" / method).mkdir(parents=True)
        (ROOT / "warmup" / method).mkdir(parents=True)
        (OUT / method / "latents").mkdir(parents=True)

    p = pipeline()
    cases = p.load_cases(SAMPLES, list(CASES), expected_indices=SOURCE_CASES)
    for name in ("conditioning_cache", "conditioning_manifest.json"):
        source = CONDITIONING_SOURCE / name
        target = OUT / name
        target.symlink_to(source, target_is_directory=source.is_dir())

    configs = {method: dataclasses.asdict(spark_config(method)) for method in SPARSE_METHODS}
    snapshot = ROOT / "snapshot"
    for name in ("h3_sparse_attention", "sol_attn"):
        shutil.copytree(
            IMPL / name,
            snapshot / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(__file__, ROOT / "runner_source.py")
    write(ROOT / "source_manifest.json", {
        str(path.relative_to(snapshot)): sha(path)
        for path in sorted(snapshot.rglob("*.py"))
    })
    write(ROOT / "protocol.json", dict(
        name=NAME,
        purpose=(
            "Native PyTorch/Diffusers ablation of pre-fix full-query behavior, "
            "dense video tail, and mean-Q padded video tail"
        ),
        dataset="first 15 prompts of the VBench core-five 10% subset",
        cases=cases,
        methods=list(METHODS),
        spark_configs=configs,
        seed=SEED,
        requested_steps=STEPS,
        transformer_evaluations=STEPS - 1,
        requested_frames=FRAMES,
        internal_vae_aligned_frames=INTERNAL_FRAMES,
        fps=FPS,
        height=HEIGHT,
        width=WIDTH,
        latent_video_shape=[1, 24, 37, 48, 84],
        video_tokens=37296,
        video_token_remainder=48,
        missing_tail_rows=16,
        gpus=list(GPUS),
        torch_compile=False,
        timing=(
            "synchronized pipeline denoise only; model loading, conditioning, "
            "CPU latent transfer and saving excluded"
        ),
        warmup="one excluded full 20-step generation per method per GPU",
        pairing=(
            "every method for a prompt runs on the same GPU; method order rotates "
            "by prompt and GPU"
        ),
        spark_policy=dict(
            dense_warmup_evaluations=4,
            dense_layer=0,
            retained_ratio=0.10,
            fanout=64,
            landmark_count=64,
        ),
        environment=ENV,
    ))
    # Verify all cached conditioning is present before concurrent model loads.
    p.configure_denoise_workflow(parse_args(), cases)
    write(ROOT / "status.json", dict(stage="prepared", completed_records=0))
    print("PREPARED", ROOT, flush=True)


def method_order(case_index, rank):
    offset = (case_index - 1 + rank) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def run_one(pipe, p, original_state, case, method, placement, *, warmup=False):
    import torch
    from h3_sparse_attention import install_h3_sparse_attention

    state = p.clone_state(original_state)
    state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
    if method == "dense":
        context = contextlib.nullcontext(None)
        config = None
    else:
        config = spark_config(method)
        context = install_h3_sparse_attention(pipe.transformer, config)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with context as plugin, torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
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
        summary = None if plugin is None else plugin.summary()

    if method != "dense":
        calls = summary["processor_calls"]
        assert summary["completed_evaluations"] == 19, summary
        assert summary["dense_evaluations"] == 4, summary
        assert calls.get("dense:warmup") == 200, calls
        assert calls.get("dense:dense_layer") == 15, calls
        assert calls.get("sparse:sol") == 735, calls
        assert calls.get("sol_landmark_preprocess_calls") == 735, calls

    if warmup:
        del result, state
        return dict(method=method, seconds=seconds)

    payload = {
        key: result[key].detach().cpu().contiguous()
        for key in ("latents", "audio_latents")
    }
    assert all(torch.isfinite(value).all() for value in payload.values())
    latent_path = OUT / method / "latents" / f"{p.case_stem(case)}.pt"
    p.atomic_torch_save(payload, latent_path)
    record = dict(
        method=method,
        case=case["index"],
        sample_id=case["sample_id"],
        prompt_sha256=case["prompt_sha256"],
        config=None if config is None else dataclasses.asdict(config),
        attention_summary=summary,
        latent_path=str(latent_path),
        latent_sha256=sha(latent_path),
        seed=SEED,
        steps=STEPS,
        frames=FRAMES,
        height=HEIGHT,
        width=WIDTH,
        denoise_seconds=seconds,
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
        placement=placement,
    )
    write(ROOT / "records" / method / f"case_{case['index']:02}.json", record)
    del result, payload, state
    return record


def worker(rank):
    rank = int(rank)
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = pipeline()
    all_cases = p.load_cases(SAMPLES, list(CASES), expected_indices=SOURCE_CASES)
    cases = all_cases[rank::len(GPUS)]
    args = parse_args()
    workflow, states = p.configure_denoise_workflow(args, cases)
    pipe, manager, acceleration, placement = p.load_denoiser(args, workflow)
    try:
        warm_case, warm_state = cases[0], states[0]
        for method in method_order(warm_case["index"], rank):
            result = run_one(
                pipe, p, warm_state, warm_case, method, placement, warmup=True
            )
            write(ROOT / "warmup" / method / f"gpu{rank}.json", result)
            print("WARMUP", rank, method, f"{result['seconds']:.3f}s", flush=True)

        for case, original_state in zip(cases, states, strict=True):
            for method in method_order(case["index"], rank):
                record_path = ROOT / "records" / method / f"case_{case['index']:02}.json"
                if record_path.exists():
                    record = read(record_path)
                    if Path(record["latent_path"]).exists() and sha(record["latent_path"]) == record["latent_sha256"]:
                        print("REUSE", rank, method, case["index"], flush=True)
                        continue
                record = run_one(
                    pipe, p, original_state, case, method, placement, warmup=False
                )
                print(
                    "MEASURED", rank, method, case["index"],
                    f"{record['denoise_seconds']:.3f}s", flush=True,
                )
        write(ROOT / f"worker_gpu{rank}.json", dict(status="complete", rank=rank))
    finally:
        acceleration.remove()
        del pipe, manager
        p.release_cpu_arenas()


def tensor_metrics(reference, candidate):
    import torch

    diff = candidate.double() - reference.double()
    ref_norm = torch.linalg.vector_norm(reference.double())
    return dict(
        equal=bool(torch.equal(reference, candidate)),
        max_abs=float(diff.abs().max()),
        mae=float(diff.abs().mean()),
        relative_l2=float(torch.linalg.vector_norm(diff) / ref_norm.clamp_min(1e-30)),
        cosine=float(torch.nn.functional.cosine_similarity(
            reference.flatten().double(), candidate.flatten().double(), dim=0
        )),
    )


def aggregate():
    import torch

    records = {
        method: [read(ROOT / "records" / method / f"case_{case:02}.json") for case in CASES]
        for method in METHODS
    }
    summary = {}
    for method, rows in records.items():
        values = [row["denoise_seconds"] for row in rows]
        summary[method] = dict(
            mean_seconds=statistics.mean(values),
            median_seconds=statistics.median(values),
            stdev_seconds=statistics.stdev(values),
            min_seconds=min(values),
            max_seconds=max(values),
            mean_peak_memory_gib=statistics.mean(
                row["peak_memory_bytes"] / 2**30 for row in rows
            ),
            values=values,
        )

    paired_runtime = {}
    for method in SPARSE_METHODS:
        deltas = [
            records[method][i]["denoise_seconds"]
            - records["spark_legacy"][i]["denoise_seconds"]
            for i in range(len(CASES))
        ]
        paired_runtime[method] = dict(
            versus="spark_legacy",
            mean_delta_seconds=statistics.mean(deltas),
            median_delta_seconds=statistics.median(deltas),
            mean_ratio=(
                summary[method]["mean_seconds"]
                / summary["spark_legacy"]["mean_seconds"]
            ),
            faster_prompts=sum(delta < 0 for delta in deltas),
            values=deltas,
        )

    comparisons = []
    for index, case in enumerate(CASES):
        loaded = {
            method: torch.load(
                records[method][index]["latent_path"], map_location="cpu", weights_only=True
            )
            for method in METHODS
        }
        for method in SPARSE_METHODS:
            row = dict(case=case, reference="dense", candidate=method)
            for key in ("latents", "audio_latents"):
                row[key] = tensor_metrics(loaded["dense"][key], loaded[method][key])
            comparisons.append(row)
        row = dict(case=case, reference="spark_tail_dense", candidate="spark_tail_pad")
        for key in ("latents", "audio_latents"):
            row[key] = tensor_metrics(
                loaded["spark_tail_dense"][key], loaded["spark_tail_pad"][key]
            )
        comparisons.append(row)

    result = dict(
        status="complete",
        summary=summary,
        paired_runtime=paired_runtime,
        latent_comparisons=comparisons,
    )
    write(ROOT / "results.json", result)
    write(ROOT / "status.json", dict(stage="complete", completed_records=60))
    write(IMPL / "tail_exp_status.json", dict(
        status="complete",
        complete=True,
        experiment=NAME,
        protocol_path=str(ROOT / "protocol.json"),
        results_path=str(ROOT / "results.json"),
        output_path=str(OUT),
        completed_records=60,
        total_records=60,
        summary=summary,
        paired_runtime=paired_runtime,
        message="Experiment and final aggregation completed successfully.",
        updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    ))
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(json.dumps(result["paired_runtime"], indent=2), flush=True)


def spawn_workers():
    jobs = []
    for rank, gpu in enumerate(GPUS):
        log = (ROOT / f"worker_gpu{gpu}.log").open("a")
        process = subprocess.Popen(
            [str(PYTHON), str(__file__), "worker", str(rank)],
            env={**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, process, log))
    write(ROOT / "pids.json", {str(gpu): process.pid for gpu, process, _ in jobs})
    codes = [(gpu, process.wait()) for gpu, process, _ in jobs]
    for _, _, log in jobs:
        log.close()
    write(ROOT / "exit_codes.json", dict(codes))
    if any(code for _, code in codes):
        raise RuntimeError(f"workers failed: {codes}")


def run():
    os.environ.update(ENV)
    prepare()
    write(ROOT / "status.json", dict(stage="running", completed_records=0))
    spawn_workers()
    aggregate()


if __name__ == "__main__":
    os.environ.update(ENV)
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "worker":
        worker(int(sys.argv[2]))
    elif command == "run":
        run()
    elif command in globals():
        globals()[command]()
    else:
        raise SystemExit(f"unknown command: {command}")
