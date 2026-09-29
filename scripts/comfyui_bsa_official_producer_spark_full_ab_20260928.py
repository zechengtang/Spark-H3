"""Separate full ComfyUI A/B for official sol_producer_kernel + Spark downstream.

Reuses the validated full-sampler runner and keeps these results separate from
the earlier BSA tile BF16 materializer experiment.
"""

from pathlib import Path

import comfyui_bsa_spark_full_ab_20260928 as trial


trial.ROOT = Path('/autodl-fs/data/h3_experiments/comfyui_bsa_official_producer_spark_full_ab_20260928')


if __name__ == '__main__':
    trial.main()
