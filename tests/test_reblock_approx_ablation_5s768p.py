"""CPU invariants for the diagnostic's capacity-preserving local refinement."""
import importlib.util
from pathlib import Path

import torch


_PATH = Path(__file__).resolve().parents[1] / "scripts/ablate_reblock_approx_5s768p_20261002.py"
_SPEC = importlib.util.spec_from_file_location("reblock_approx_screen", _PATH)
screen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(screen)


def test_swap_preserves_permutation_tail_and_reduces_objective():
    generator = torch.Generator().manual_seed(42)
    features = torch.randn(2, 272, 8, generator=generator)
    permutation = torch.stack([torch.randperm(272, generator=generator) for _ in range(2)])
    result, diagnostics = screen.swap_refine(features, permutation)
    assert torch.equal(result.sort(-1).values, permutation.sort(-1).values)
    assert torch.equal(result[:, 256:], permutation[:, 256:])
    assert all(b <= a + 1e-6 for a, b in zip(diagnostics["objective"], diagnostics["objective"][1:]))
    assert diagnostics["token_swaps"] > 0


def test_swap_repairs_mixed_adjacent_blocks_without_changing_capacity():
    features = torch.zeros(1, 160, 2)
    features[:, :48, 0] = 1
    features[:, 48:64, 1] = 1
    features[:, 64:80, 0] = 1
    features[:, 80:, 1] = 1
    permutation = torch.arange(160)[None]
    result, diagnostics = screen.swap_refine(features, permutation)
    packed = features.gather(1, result[..., None].expand_as(features))
    assert torch.equal(packed[:, :64, 0], torch.ones(1, 64))
    assert torch.equal(packed[:, 64:128, 1], torch.ones(1, 64))
    assert torch.equal(result[:, 128:], permutation[:, 128:])
    assert diagnostics["objective"][0] > .1
    assert diagnostics["objective"][-1] < 1e-6
    assert diagnostics["token_swaps"] == 16


def test_homogeneous_blocks_need_no_exchange():
    features = torch.ones(1, 128, 4)
    permutation = torch.arange(128)[None]
    result, diagnostics = screen.swap_refine(features, permutation)
    assert torch.equal(result, permutation)
    assert diagnostics["token_swaps"] == 0


def test_log_domain_mixing_handles_overflow_and_underflow():
    values = torch.tensor([[2., 0.], [0., 3.]])
    exact_value = torch.tensor([[7., 7.]])
    for shift in (-1000., 1000.):
        logmass = torch.tensor([[shift, shift-1]])
        exact_logmass = torch.tensor([shift])
        out,error,log_error = screen.mix_summary(logmass,values,exact_logmass,exact_value)
        weights = torch.tensor([0., 0., -1.]).softmax(0)
        expected = weights[0]*exact_value+weights[1]*values[0]+weights[2]*values[1]
        assert torch.allclose(out,expected,atol=3e-4)
        assert torch.isfinite(error).all() and torch.isfinite(log_error).all()
