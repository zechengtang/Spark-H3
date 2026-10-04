#!/usr/bin/env python3
"""Decode/archive and score the 5s/10s shared-layout M2 matrix."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

import reblock_priority_10s768p_quality as quality


DURATION = os.environ.get("REB_MATRIX_DURATION", "5s")
if DURATION not in ("5s", "10s"):
    raise ValueError("REB_MATRIX_DURATION must be 5s or 10s")
quality.base.FRAMES = 120 if DURATION == "5s" else 240
SCRIPT = Path(__file__).resolve()


if DURATION == "5s":
    # Keep the paired metric implementation unchanged. Only video packaging is
    # delegated to Benchmark's detached, resume-safe archive spool.
    sys.path.insert(0, str(quality.base.BENCH_SCRIPTS))
    import h3_quality_batch_worker as archive_worker

    _score_worker = quality.worker
    _archive_process_started = False

    def archive_media_async(experiment: Path, arm: str, case: dict, pixels, audio, rate: int):
        global _archive_process_started
        folder = experiment / "quality_videos" / arm
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"{case['index']:02d}.mkv"
        evidence_path = output.with_suffix(".archive.json")
        if output.is_file() and evidence_path.is_file():
            evidence = quality.base.read_json(evidence_path)
            digest = quality.sha256(output)
            target = archive_worker.archive_job_path(
                experiment / "quality_work", arm, case["index"]
            )
            job = archive_worker.read(target) if target.is_file() else {}
            if (evidence.get("file_sha256") == digest
                    or (job.get("status") == "complete"
                        and job.get("video_sha256") == digest)):
                evidence["file_sha256"] = digest
                return output, evidence
            raise RuntimeError(f"stale archive evidence: {output}")

        spool = experiment / "quality_work"
        target = archive_worker.archive_job_path(spool, arm, case["index"])
        if target.is_file():
            job = archive_worker.read(target)
            if job.get("status") == "exhausted" and all(
                Path(job[key]).is_file() for key in ("raw_path", "audio_source")
            ):
                job.update(status="pending", attempts=0, error=None)
                archive_worker.write(target, job)
            if job.get("status") not in ("pending", "running", "failed", "complete"):
                raise RuntimeError(f"unrecoverable archive job: {target}")
        else:
            raw = folder / f".{case['index']:02d}.rgb-{os.getpid()}.npy"
            wav = folder / f".{case['index']:02d}.audio-{os.getpid()}.wav"
            np.save(raw, pixels)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate),
                 "-ac", str(audio.shape[0]), "-i", "pipe:0", "-c:a", "pcm_f32le", str(wav)],
                input=np.ascontiguousarray(audio.T).tobytes(), check=True,
            )
            slot = int(os.environ["CUDA_VISIBLE_DEVICES"])
            rate_sample = archive_worker.DEFAULT_VERIFY_SAMPLE_RATE
            archive_worker.write(target, {
                "schema_version": 1, "status": "pending", "attempts": 0,
                "slot": slot, "arm": arm, "case": case["index"], "fps": 24,
                "raw_path": str(raw), "audio_source": str(wav), "video_path": str(output),
                "verify_pixels": archive_worker.sample_full_verification(
                    arm, case["index"], rate_sample),
                "verify_sample_rate": rate_sample,
            })
        if not _archive_process_started:
            archive_worker.launch_archive_worker(
                SimpleNamespace(experiment=experiment, slot=int(os.environ["CUDA_VISIBLE_DEVICES"]),
                                workers=quality.base.WORKERS), spool)
            _archive_process_started = True
        return output, {"file_sha256": None}

    def worker_async(experiment: Path, slot: int) -> None:
        global _archive_process_started
        spool = experiment / "quality_work"
        for target in sorted((spool / "archive_jobs").glob("*.json")):
            job = archive_worker.read(target)
            if job.get("slot") != slot or job.get("status") == "complete":
                continue
            if not all(Path(job[key]).is_file() for key in ("raw_path", "audio_source")):
                continue
            if job.get("status") == "exhausted":
                job.update(status="pending", attempts=0, error=None)
                archive_worker.write(target, job)
            if not _archive_process_started:
                archive_worker.launch_archive_worker(
                    SimpleNamespace(experiment=experiment, slot=slot,
                                    workers=quality.base.WORKERS), spool)
                _archive_process_started = True
        _score_worker(experiment, slot)
        archive_worker.write(
            spool / f"batch_worker_{slot:02d}.json",
            {"status": "complete", "slot": slot},
        )

    def wait_and_finalize_archives(experiment: Path, protocol: dict) -> None:
        spool = experiment / "quality_work"
        expected = (len(protocol["arms"]) + 1) * len(protocol["cases"])
        deadline = time.monotonic() + 60 * 60
        last_complete = -1
        last_launch = 0.0
        while True:
            jobs = archive_worker.archive_job_files(spool)
            statuses = [archive_worker.read(path).get("status") for path in jobs]
            markers = [
                spool / f"archive_worker_{slot:02d}.json"
                for slot in range(protocol["workers"])
            ]
            if any(path.is_file() and archive_worker.read(path).get("status") == "failed"
                   for path in markers) or "exhausted" in statuses:
                raise RuntimeError("asynchronous archive worker reported a failed job")
            if (len(jobs) == expected and all(status == "complete" for status in statuses)
                    and all(path.is_file() and archive_worker.read(path).get("status") == "complete"
                            for path in markers)):
                break
            complete_count = statuses.count("complete")
            now = time.monotonic()
            if complete_count != last_complete:
                last_complete = complete_count
                last_launch = now
            elif now - last_launch >= 30:
                pending_slots = {
                    archive_worker.read(path)["slot"] for path in jobs
                    if archive_worker.read(path).get("status") != "complete"
                }
                for slot in pending_slots:
                    archive_worker.launch_archive_worker(
                        SimpleNamespace(experiment=experiment, slot=slot,
                                        workers=protocol["workers"]), spool)
                last_launch = now
            quality.base.atomic_json(
                quality.base.quality_root(experiment) / "archive_status.json",
                {"status": "running", "complete": complete_count, "expected": expected},
            )
            if time.monotonic() > deadline:
                raise TimeoutError(f"archive queue did not finish: {statuses.count('complete')}/{expected}")
            time.sleep(10)

        evidence_by_key = {}
        for path in jobs:
            job = archive_worker.read(path)
            output = Path(job["video_path"])
            digest = quality.sha256(output)
            if digest != job["video_sha256"]:
                raise RuntimeError(f"archive hash changed: {output}")
            evidence = job["evidence"]
            evidence["file_sha256"] = digest
            quality.base.atomic_json(output.with_suffix(".archive.json"), evidence)
            evidence_by_key[job["arm"], job["case"]] = (digest, evidence)
        for arm in protocol["arms"]:
            for case in protocol["cases"]:
                target = quality.base.result_path(experiment, arm, case)
                row = quality.base.read_json(target)
                row["video_sha256"] = evidence_by_key[arm, case["index"]][0]
                row["dense_video_sha256"] = evidence_by_key["dense", case["index"]][0]
                row["archive_status"] = "complete"
                row["archive_verification_mode"] = evidence_by_key[arm, case["index"]][1]["verification_mode"]
                quality.base.atomic_json(target, row)
        quality.base.atomic_json(
            quality.base.quality_root(experiment) / "archive_status.json",
            {"status": "complete", "complete": expected, "expected": expected},
        )

    quality.archive_media = archive_media_async
    quality.base.worker = worker_async

    _quality_summarize = quality.summarize

    def summarize_async(experiment: Path):
        experiment = experiment.resolve()
        protocol = quality.base.read_json(quality.base.quality_root(experiment) / "protocol.json")
        wait_and_finalize_archives(experiment, protocol)
        return _quality_summarize(experiment)

    quality.base.summarize = summarize_async


def run(experiment: Path, requested_arms) -> None:
    experiment = experiment.resolve()
    protocol = quality.prepare(experiment, requested_arms)
    root = quality.base.quality_root(experiment)
    if DURATION == "5s":
        archive_protocol = {
            "mode": "detached_archive_spool",
            "benchmark_worker": str(Path(archive_worker.__file__).resolve()),
            "benchmark_worker_sha256": quality.sha256(Path(archive_worker.__file__)),
            "full_pixel_verify_sample_rate": archive_worker.DEFAULT_VERIFY_SAMPLE_RATE,
            "score_uses_pre_encoding_pixels": True,
        }
        archive_protocol_path = root / "async_archive_protocol.json"
        if archive_protocol_path.is_file():
            if quality.base.read_json(archive_protocol_path) != archive_protocol:
                raise RuntimeError("5s asynchronous archive protocol changed after preparation")
        else:
            quality.base.atomic_json(archive_protocol_path, archive_protocol)
    quality.base.atomic_json(
        root / "status.json", {"status": "running", "arms": protocol["arms"]}
    )
    quality.base.atomic_json(
        root / "matrix_adapter.json",
        {
            "duration": DURATION,
            "frames": quality.base.FRAMES,
            "path": str(SCRIPT),
            "sha256": quality.sha256(SCRIPT),
        },
    )
    jobs = []
    for slot in range(protocol["workers"]):
        log = (root / f"worker_{slot}.log").open("a")
        env = {
            **os.environ,
            "REB_MATRIX_DURATION": DURATION,
            "CUDA_VISIBLE_DEVICES": str(slot),
            "H3_IMPL_REPO": str(quality.base.REPO),
            "H3_DIFFUSERS_DIR": str(quality.base.MODEL),
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "PYTHONPATH": str(quality.base.LPIPS_PACKAGE)
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
            "TORCH_HOME": str(quality.base.LPIPS_CACHE),
        }
        process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "worker",
                "--experiment",
                str(experiment),
                "--slot",
                str(slot),
            ],
            cwd=quality.base.REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((slot, process, log))
    failures = []
    for slot, process, log in jobs:
        code = process.wait()
        log.close()
        if code:
            failures.append({"slot": slot, "exit_code": code})
    if failures:
        quality.base.atomic_json(
            root / "status.json", {"status": "failed", "workers": failures}
        )
        raise RuntimeError(f"quality worker failures: {failures}")
    result = quality.base.summarize(experiment)
    print(json.dumps(result["summaries"], indent=2))


quality.base.run = run


if __name__ == "__main__":
    quality.base.main()
