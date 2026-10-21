"""Preserve chunk-phase selections used by Ref2VA comparison scripts."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ref2va_temporal import _chunk_phase_indices


def test_chunk_phase_selection_handles_partial_tail():
    assert _chunk_phase_indices(12, 2) == [0, 1, 3, 5, 6, 8, 10, 11]
    assert _chunk_phase_indices(12, 1, single_normal_index=2) == [0, 3, 5, 8, 10]
    assert _chunk_phase_indices(12, 4, head_tokens=0) == [1, 2, 3, 4, 6, 7, 8, 9, 11]


def test_explicit_phases_are_sorted_and_validated():
    assert _chunk_phase_indices(7, 2, normal_indices="3,0") == [0, 1, 4, 5, 6]
    with pytest.raises(ValueError, match="unique values"):
        _chunk_phase_indices(7, 2, normal_indices="0,0")
