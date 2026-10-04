"""Wait for generation, then run Table 3/4 quality and VBench scoring."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
EXP = Path("/autodl-fs/data/h3_experiments/chunk5_chunk10_table34_50prompt_20260929")
REPORT = REPO / "reports" / EXP.name
OUT = Path("/autodl-fs/data/h3_outputs") / EXP.name / "videos"
ARMS = ("chunk5_topk10_reblock", "chunk10_topk10_reblock")
CASES = list(range(1, 51))
GPUS = list(range(4))
DENSE_MANIFEST = Path(
    "/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913/dense/generation_manifest.json"
)
QUALITY_WORKER = REPO / "scripts/quality_batch_worker_chunk5_chunk10_table34_50prompt_20260929.py"
VBENCH_MODULE = REPO / "scripts/vbench_chunk5_chunk10_table34_50prompt_20260929.py"
VBENCH_PYTHON = Path("/root/h3_local/venvs/vbench_core5/bin/python")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def wait_generation():
    while True:
        protocol = read(EXP / "protocol.json")
        status = protocol["status"]
        if status == "generation_complete":
            print("GENERATION COMPLETE; STARTING POSTPROCESS", flush=True)
            return
        if status == "needs_attention":
            raise RuntimeError(f"generation failed: {protocol.get('exit_codes')}")
        print("WAITING FOR GENERATION", status, flush=True)
        time.sleep(30)


def quality_prepare():
    references = {row["index"]: row for row in read(DENSE_MANIFEST)["records"]}
    rows = []
    for case in CASES:
        for arm in ARMS:
            record_path = EXP / "records" / f"{arm}_{case:02}.json"
            rows.append(
                dict(
                    arm=arm,
                    case=case,
                    denoise_record=read(record_path),
                    denoise_record_path=str(record_path),
                    reference=references[case],
                )
            )
    (EXP / "quality_work").mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = EXP / "historical_metric_cache.json"
    if not cache.exists():
        write(cache, {"cache": {}})
    write(
        EXP / "quality_work/protocol.json",
        dict(
            input_rows=rows,
            cases=CASES,
            arms=list(ARMS),
            metrics=[
                "PSNR pooled RGB MSE",
                "SSIM Gaussian11 sigma1.5 valid RGB frame mean",
                "LPIPS AlexNet v0.1 RGB[-1,1] aligned frame mean",
            ],
        ),
    )
    write(EXP / "quality_protocol.json", dict(cases=CASES, arms=list(ARMS)))
    print("QUALITY PREPARED", len(rows), flush=True)


def quality_run():
    jobs = []
    for slot in range(8):
        log = (EXP / f"quality_batch_{slot:02}.log").open("a")
        process = subprocess.Popen(
            [sys.executable, str(QUALITY_WORKER), str(slot)],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(GPUS[slot % len(GPUS)])},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((process, log))
    codes = [process.wait() for process, _ in jobs]
    for _, log in jobs:
        log.close()
    if any(codes):
        raise RuntimeError(f"quality worker failures: {codes}")


def quality_aggregate():
    scores = []
    for arm in ARMS:
        for case in CASES:
            row = read(EXP / "quality_work/quality" / f"{arm}_{case:02}.json")
            assert row["status"] == "complete" and len(row["frame_lpips"]) == 240
            record = read(EXP / "records" / f"{arm}_{case:02}.json")
            assert row["latent_sha256"] == record["latent_sha256"]
            scores.append(row)
    summary = {
        arm: {
            metric: sum(row[metric] for row in scores if row["arm"] == arm) / len(CASES)
            for metric in ("psnr_db", "ssim", "lpips", "denoise_seconds")
        }
        for arm in ARMS
    }
    result = dict(
        status="complete",
        cases=CASES,
        summary=summary,
        rows=scores,
        note="50 VBench core-five 20% prompts; paired Dense-reference RGB metrics.",
    )
    write(EXP / "quality_results.json", result)
    generation = read(EXP / "results.json")
    generation["quality"] = result
    write(EXP / "results.json", generation)
    lines = [
        "# Fixed latent-frame chunks: Table 3/4 protocol",
        "",
        "50 VBench core-five 20% prompts, seed 42, 1344x768, 240 output frames, 20 requested steps (19 denoiser evaluations), TopK=10, per-block torch.compile. One full first-prompt generation per GPU and arm was excluded from timing.",
        "",
        "Global reweight matches Table 3/4 (levels_up=99). The original fixed-chunk permutation is unchanged; a compatibility patch publishes its permutation-invariant global video root.",
        "",
        "## Generation timing",
        "",
        "| Arm | N | Mean s | Median s | Min s | Max s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = generation["summary"][arm]
        lines.append(
            f"| {arm} | {item['n']} | {item['mean_seconds']:.3f} | {item['median_seconds']:.3f} | {item['min_seconds']:.3f} | {item['max_seconds']:.3f} |"
        )
    lines += [
        "",
        "## Paired Dense-reference RGB metrics",
        "",
        "PSNR uses pooled RGB MSE; SSIM uses an 11x11 Gaussian window (sigma 1.5); LPIPS uses AlexNet v0.1. Means cover all 50 prompts and 240 aligned frames.",
        "",
        "| Arm | Denoise s | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = summary[arm]
        lines.append(
            f"| {arm} | {item['denoise_seconds']:.3f} | {item['psnr_db']:.4f} | {item['ssim']:.6f} | {item['lpips']:.6f} |"
        )
    (EXP / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("QUALITY COMPLETE", json.dumps(summary), flush=True)


def vbench_setup():
    source = Path("/autodl-fs/data/h3_experiments/strict_splitter_tau1_20260918")
    shutil.copytree(
        source / "vbench_code",
        EXP / "vbench_code",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(source / "h3_vbench_queue.py", EXP / "h3_vbench_queue.py")
    shutil.copy2(VBENCH_MODULE, EXP / "vbench.py")
    print("VBENCH SETUP DONE", flush=True)


def vbench_run():
    subprocess.run([str(VBENCH_PYTHON), str(EXP / "vbench.py"), "prepare"], check=True)
    subprocess.run(
        [
            str(VBENCH_PYTHON),
            str(EXP / "h3_vbench_queue.py"),
            "run",
            "--experiment",
            str(EXP),
            "--gpus",
            *map(str, GPUS),
        ],
        check=True,
    )


def sync_reports():
    REPORT.mkdir(parents=True, exist_ok=True)
    for name in (
        "REPORT.md",
        "protocol.json",
        "results.json",
        "quality_protocol.json",
        "quality_results.json",
    ):
        source = EXP / name
        if source.exists():
            shutil.copy2(source, REPORT / name)
    vbench = EXP / "vbench/results.json"
    if vbench.exists():
        shutil.copy2(vbench, REPORT / "vbench_results.json")
    print("REPORTS SYNCED", REPORT, flush=True)


def all():
    wait_generation()
    quality_prepare()
    quality_run()
    quality_aggregate()
    vbench_setup()
    vbench_run()
    sync_reports()


if __name__ == "__main__":
    globals()[sys.argv[1]]()
