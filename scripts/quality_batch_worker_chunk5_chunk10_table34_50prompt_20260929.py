"""Thin configuration wrapper around the validated Table 3/4 quality worker."""

import importlib.util
from pathlib import Path
import sys


SOURCE = Path(
    "/autodl-fs/data/h3_repos/MiniMax-H3-Experiments/scripts/quality_batch_worker.py"
)
spec = importlib.util.spec_from_file_location("table34_quality_batch_worker", SOURCE)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

worker.EXP = Path(
    "/autodl-fs/data/h3_experiments/chunk5_chunk10_table34_50prompt_20260929"
)
worker.ROOT = worker.EXP / "quality_work"
worker.OUT = (
    Path("/autodl-fs/data/h3_outputs") / worker.EXP.name / "videos"
)
worker.ARMS = ("chunk5_topk10_reblock", "chunk10_topk10_reblock")
worker.CASES = list(range(1, 51))
worker.WORKERS = 8


if __name__ == "__main__":
    worker.main(int(sys.argv[1]))
