"""Current-code ComfyUI Sol/Spark 4-prompt sampler-only speed retest.

Reuses the paired protocol and source-hash capture from the original run,
but writes to a fresh directory and intentionally skips decoding/scoring.
"""

import os
from pathlib import Path

# PO may export HTTP(S)_PROXY without a localhost exemption. ComfyUI's
# orchestrator only calls loopback HTTP APIs, which must bypass that proxy.
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import comfyui_latest_spark_4prompt_20260926 as trial


trial.NAME = "comfyui_spark_speed_retest_4prompt_20260927"
trial.ROOT = Path("/autodl-fs/data/h3_experiments") / trial.NAME
trial.OUT = Path("/autodl-fs/data/h3_outputs") / trial.NAME
ARCHIVED_PROTOCOL = (
    Path("/autodl-fs/data/h3_experiments")
    / "comfyui_latest_spark_4prompt_5s10s768p_20260926/protocol.json"
)
# The original 10-prompt source experiment was cleaned up. The four cases
# and their conditioning hashes remain in the archived paired protocol.
trial.source.cases = lambda: trial.read(ARCHIVED_PROTOCOL)["cases"]


def main() -> None:
    trial.prepare()
    protocol_path = trial.ROOT / "protocol.json"
    protocol = trial.read(protocol_path)
    protocol["settings"].update(
        midpoint_direction_mode="legacy",
        spark_tail_granularities=["block", "query"],
        scope="sampler speed only; no decode, PSNR or VBench",
    )
    trial.write(protocol_path, protocol)
    for path, expected in protocol["source_hashes"].items():
        if trial.base.sha256(Path(path)) != expected:
            raise RuntimeError(f"Implementation changed since preparation: {path}")
    trial.denoise()
    trial.write(trial.ROOT / "status.json", {"status": "complete", "stage": "denoise"})


if __name__ == "__main__":
    main()
