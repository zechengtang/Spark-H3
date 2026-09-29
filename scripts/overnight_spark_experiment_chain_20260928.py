"""Start TopK30 only after the running block8x8 experiment fully completes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


BLOCK_ROOT = Path(
    "/autodl-fs/data/h3_experiments/"
    "diffusers_spark_block8x8_25prompt_10s768p_20260928"
)
BLOCK_PID = 13076
TOPK30_SCRIPT = Path(__file__).with_name(
    "diffusers_spark_topk30_25prompt_20260928.py"
)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def complete(path):
    try:
        return json.loads(path.read_text()).get("status") == "complete"
    except (OSError, ValueError):
        return False


def main():
    while alive(BLOCK_PID):
        time.sleep(30)
    deadline = time.monotonic() + 2 * 60 * 60
    while not complete(BLOCK_ROOT / "vbench" / "results.json"):
        if time.monotonic() >= deadline:
            raise RuntimeError("block8x8 VBench did not complete; TopK30 will not start")
        time.sleep(30)
    result = json.loads((BLOCK_ROOT / "results.json").read_text())
    if result.get("status") != "quality_complete":
        raise RuntimeError("block8x8 quality is incomplete; TopK30 will not start")
    print("BLOCK8X8_COMPLETE starting TopK30", flush=True)
    subprocess.run([sys.executable, str(TOPK30_SCRIPT), "run"], check=True)
    print("TOPK30_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
