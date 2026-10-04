#!/usr/bin/env python3
"""Duration-selectable wrapper for Benchmark decoding and RGB quality scoring."""
from __future__ import annotations

import os

import reblock_priority_10s768p_quality as quality


quality.base.FRAMES = int(os.environ.get("EXACT_QUALITY_FRAMES", "240"))


if __name__ == "__main__":
    quality.base.main()
