"""Weighted-key prefetch must preserve routing and pipeline phase boundaries."""
import functools

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('mode', ['threshold', 'hybrid', 'explicit', 'exact', 'skipped', 'alternating'])
def test_weighted_key_prefetch_exact_parity(monkeypatch, mode):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip('SM120 required')
    import h3_sparse_attention.spark_reweight_sm120 as kernel
    import h3_sparse_attention.sol_numerator_virtual_q as rw

    torch.manual_seed(917)
    b, t, h = 2, 8329, 2
    n = (t+63)//64
    q, k, v = [torch.randn(b, t, h, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    ranges = torch.tensor([[0, 8192], [8192, t]], device='cuda')
    mapping = torch.tensor([0]*128+[1]*(n-128), device='cuda')
    anchors = rw.build_virtual_anchors(q, ranges)
    kc = rw.reduce_virtual_key_centroids(k)
    threshold = torch.full((b, n, h), .02, device='cuda') if mode in ('threshold', 'hybrid') else None
    route = None if mode == 'threshold' else torch.randint(2, (b, n, h, n), device='cuda', dtype=torch.uint8)
    if mode == 'hybrid':
        threshold[:, ::2] = torch.nan
    elif mode == 'exact':
        route.fill_(1)
    elif mode == 'skipped':
        route.zero_()
    elif mode == 'alternating':
        # An all-exact group followed by all-skipped, then a partial exact
        # group exercises both draining speculative loads and phase handoff.
        route.fill_(1)
        route[..., 64:128] = 0
    original = kernel.SparkReweightForwardSm120
    results = []
    for prefetch, shared_mass in ((False, False), (True, False), (True, True)):
        monkeypatch.setattr(kernel, 'SparkReweightForwardSm120', functools.partial(
            original, prefetch_approx_k=prefetch, shared_log_mass=shared_mass))
        monkeypatch.setattr(rw, '_FUSED_COMPILED', {})
        results.append(rw._fused_virtual(q, k, v, anchors, ranges, mapping, kc,
                                       threshold, route, t, 0, export_route=True,
                                       force_local_blocks=False))
    for result in results[1:]:
        for reference, actual in zip(results[0], result):
            assert torch.equal(reference, actual)
        assert torch.isfinite(result[0]).all()
