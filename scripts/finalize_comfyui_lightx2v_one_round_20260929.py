"""Honor the revised scope: keep seed 42 only and stop after its last result."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import time


ROOT = Path("/autodl-fs/data/h3_experiments/comfyui_lightx2v_sparse_warm_speed_20260929")
METHODS = ("official_sol", "spark_10pct", "spark_114blocks")
DURATIONS = ("10s", "14p4s")


def record(duration: str, method: str, phase: str, seed: int) -> dict:
    path = ROOT / "records" / f"{duration}_{method}_{phase}_seed{seed}.json"
    return json.loads(path.read_text())


def main() -> None:
    pid = int((ROOT / "runner.pid").read_text())
    targets = [ROOT / "records" / f"14p4s_{method}_timed_seed42.json" for method in METHODS]
    deadline = time.monotonic() + 3600
    while not all(path.is_file() for path in targets):
        if time.monotonic() > deadline:
            raise TimeoutError("seed 42 results did not complete in one hour")
        if not Path(f"/proc/{pid}").exists():
            raise RuntimeError("runner stopped before seed 42 results completed")
        time.sleep(0.25)
    print("All three 14.4s seed 42 measurements complete; stopping extra rounds", flush=True)
    os.kill(pid, signal.SIGINT)
    # The runner's finally block terminates its ComfyUI child.
    time.sleep(3)

    rows = []
    for duration in DURATIONS:
        sol = record(duration, "official_sol", "timed", 42)["sampler_seconds"]
        for method in METHODS:
            warmup = record(duration, method, "warmup", 41)
            timed = record(duration, method, "timed", 42)
            rows.append({
                "duration": duration, "frames": timed["frames"], "method": method,
                "warmup_seconds_excluded": warmup["sampler_seconds"],
                "timed_seconds": timed["sampler_seconds"], "seed": 42,
                "speedup_vs_sol": sol / timed["sampler_seconds"],
            })
    result = {
        "status": "complete", "scope": "one measured run after sparse warmup per mode and duration",
        "model": "lightx2v", "durations": DURATIONS, "methods": METHODS,
        "seed": 42, "rows": rows, "completed_unix": time.time(),
    }
    (ROOT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = ["# LightX2V: one measured run after first sparse step warmup", "",
             "The requested 15s case is the existing 14.4s / 345-frame prompt. "
             "Each method and duration first ran one uncounted 8-step warmup containing its first sparse step. "
             "Reported times use seed 42. The 10s seed 43 and 44 runs completed before the scope was reduced and are excluded below.", "",
             "| Duration | Method | Uncounted warmup (s) | Timed denoise (s) | Speed vs Sol |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        label = "14.4s / 345f" if row["duration"] == "14p4s" else "10s / 243f"
        lines.append(f"| {label} | {row['method']} | {row['warmup_seconds_excluded']:.2f} | "
                     f"{row['timed_seconds']:.2f} | {row['speedup_vs_sol']:.3f}× |")
    lines += ["", "Times cover the ComfyUI sampler node only, excluding model loading, prompt encoding, and latent save."]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")
    failure = ROOT / "failure.json"
    if failure.is_file():
        entry = json.loads(failure.read_text())
        if entry.get("type") == "KeyboardInterrupt":
            failure.rename(ROOT / "intentional_stop_after_one_round.json")
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol["status"] = "complete_one_round"
    protocol["measured_seeds_used"] = [42]
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    print(ROOT / "report.md", flush=True)


if __name__ == "__main__":
    main()
