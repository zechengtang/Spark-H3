#!/usr/bin/env python3
"""Replay three remapped and three non-remapped Spark-H3 20% cases on GPU 0."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import repro_blog_table3_speed_content_2gpu_20261004 as base


base.NAME = "spark20_remap_nonremap_6prompt_gpu0_20261004"
base.ROOT = Path("/autodl-fs/data/h3_experiments") / base.NAME
base.OUT = Path("/autodl-fs/data/h3_outputs") / base.NAME
base.CASE_INDICES = (4, 18, 21, 28, 36, 47)
base.CASE_GROUPS = {
    4: "remapped", 21: "remapped", 47: "remapped",
    18: "non_remapped", 28: "non_remapped", 36: "non_remapped",
}
base.GPUS = (0,)
base.METHODS = ("spark20",)


def prepare() -> None:
    base.prepare()
    protocol_path = base.ROOT / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update({
        "purpose": (
            "Spark-H3 20% historical-latent replay on three newly sampled remapped "
            "and three newly sampled non-remapped prompts"
        ),
        "sampling": {
            "selection": "fixed balanced sample",
            "without_replacement": True,
            "excluded_previous_cases": [14, 20],
            "remapped": [4, 21, 47],
            "non_remapped": [18, 28, 36],
        },
        "gpus": [0],
        "methods": ["spark20"],
        "method_order": {"gpu0": ["spark20"]},
        "warmup": {
            "exactly_once_per_gpu_per_method": True,
            "spark20": {"steps": 3, "evaluations": 2, "dense": 1, "sparse": 1},
            "excluded_from_comparison": True,
        },
    })
    base.write(protocol_path, protocol)
    (base.ROOT / "runner_source.py").write_text(Path(__file__).read_text())


def worker() -> None:
    import torch

    torch.set_num_threads(4)
    pipeline = base.base.pipeline()
    selected = base.cases()
    workflow, states = pipeline.configure_denoise_workflow(base.args(), selected)
    pipe, manager, acceleration, placement = pipeline.load_denoiser(base.args(), workflow)
    try:
        warm_seconds, warm_summary, _ = base.run_once(
            pipe, pipeline, states[0], "spark20", steps=3, save_path=None
        )
        base.write(base.ROOT / "records" / "gpu0_spark20_warmup.json", {
            "phase": "discarded_warmup", "gpu": 0, "method": "spark20",
            "case": selected[0]["index"], "steps": 3, "evaluations": 2,
            "seconds": warm_seconds, "attention_summary": warm_summary,
        })
        print("WARMUP spark20", round(warm_seconds, 3), flush=True)

        for case, state in zip(selected, states):
            index = case["index"]
            latent_path = (
                base.OUT / "spark20" / "latents" /
                f"{index:02d}_{case['sample_id']}.pt"
            )
            seconds, summary, payload = base.run_once(
                pipe, pipeline, state, "spark20", steps=base.STEPS,
                save_path=latent_path,
            )
            reference = base.reference_latent("spark20", case)
            comparison = base.compare_payload(payload, reference)
            record = {
                "phase": "replay", "gpu": 0, "method": "spark20",
                "case": index, "sample_id": case["sample_id"],
                "group": base.CASE_GROUPS[index],
                "prompt_sha256": case["prompt_sha256"],
                "placement": placement, "steps": base.STEPS,
                "evaluations": base.STEPS - 1, "seconds": seconds,
                "attention_summary": summary,
                "latent_path": str(latent_path),
                "latent_sha256": base.sha(latent_path),
                "reference_latent_path": str(reference),
                "reference_latent_sha256": base.sha(reference),
                "file_sha256_equal": base.sha(latent_path) == base.sha(reference),
                "content_comparison": comparison,
            }
            base.write(base.ROOT / "records" / f"case_{index:02d}.json", record)
            print(
                "REPLAY", index, base.CASE_GROUPS[index], round(seconds, 3),
                "video_equal", comparison["latents"]["torch_equal"],
                "audio_equal", comparison["audio_latents"]["torch_equal"],
                flush=True,
            )
            del payload
        base.write(base.ROOT / "worker_gpu0.json", {"status": "complete"})
    finally:
        acceleration.remove()
        del pipe, manager
        pipeline.release_cpu_arenas()


def summarize() -> None:
    rows = [
        json.loads(path.read_text())
        for path in sorted((base.ROOT / "records").glob("case_*.json"))
    ]
    groups = {}
    for group in ("remapped", "non_remapped"):
        chosen = [row for row in rows if row["group"] == group]
        groups[group] = {
            "count": len(chosen),
            "video_equal": sum(
                row["content_comparison"]["latents"]["torch_equal"] for row in chosen
            ),
            "audio_equal": sum(
                row["content_comparison"]["audio_latents"]["torch_equal"] for row in chosen
            ),
            "both_equal": sum(
                row["content_comparison"]["latents"]["torch_equal"]
                and row["content_comparison"]["audio_latents"]["torch_equal"]
                for row in chosen
            ),
        }
    result = {
        "status": "complete",
        "gpu": 0,
        "method": "spark20",
        "cases": rows,
        "groups": groups,
        "all_latent_tensors_equal": all(
            row["content_comparison"]["latents"]["torch_equal"]
            and row["content_comparison"]["audio_latents"]["torch_equal"]
            for row in rows
        ),
    }
    base.write(base.ROOT / "results.json", result)
    protocol_path = base.ROOT / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["status"] = "complete"
    base.write(protocol_path, protocol)
    print(json.dumps(result, indent=2), flush=True)


def run() -> None:
    prepare()
    worker()
    summarize()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        run()
    elif command == "worker":
        worker()
    elif command == "summarize":
        summarize()
    else:
        raise ValueError(command)
