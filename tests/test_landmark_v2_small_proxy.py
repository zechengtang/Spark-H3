"""Small-proxy launch arithmetic, determinism, and graph replay coverage."""
import pytest
import torch
from h3_sparse_attention import landmark_v2_cosine_fast as cosine

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason='SM120 small-proxy tuning')

@pytest.mark.parametrize('children', [14, 15, 16])
@pytest.mark.parametrize('kind', ['random', 'zero', 'tied', 'scaled', 'unequal'])
def test_small_proxy_against_legacy(monkeypatch, children, kind):
    torch.manual_seed(123 + children)
    x = torch.randn(128, 32, 128, device='cuda', dtype=torch.bfloat16)
    caps = (64,) * children
    weights = torch.full((128, 32), 2 * children, device='cuda', dtype=torch.long)
    if kind == 'zero':
        x.zero_()
    elif kind == 'tied':
        x[:] = x[:, :1].clone()
    elif kind == 'scaled':
        x *= torch.logspace(-4, 4, 32, device='cuda').reshape(1, 32, 1)
    elif kind == 'unequal':
        caps = tuple(32 if i % 2 else 96 for i in range(children))
        weights.zero_()
        weights[:, :16] = sum(caps) // 16
    monkeypatch.setenv('H3_LMV2_SMALL_PROXY_FAST', '0')
    reference = cosine.build_cosine_directions(x, weights, caps)
    monkeypatch.setenv('H3_LMV2_SMALL_PROXY_FAST', '1')
    actual = cosine.build_cosine_directions(x, weights, caps)
    torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-6)
    assert torch.equal(actual, cosine.build_cosine_directions(x, weights, caps))
    if kind in ('zero', 'tied'):
        assert torch.equal(actual, reference)
    # Returning leaf weights must still execute the final writes.
    partitioned, _, active = cosine.build_cosine_directions(x, weights, caps, return_partition=True)
    assert torch.equal(partitioned, actual)
    from h3_sparse_attention.landmark_v2_terminal import split_topology
    for node, row in enumerate(split_topology(caps)):
        left = active[:, 2 * node + 1]
        right = active[:, 2 * node + 2]
        assert torch.equal(left.sum(-1), torch.full((128,), row[1], device='cuda'))
        assert torch.equal(right.sum(-1), torch.full((128,), row[2], device='cuda'))


def test_small_proxy_graph_changed_inputs(monkeypatch):
    monkeypatch.setenv('H3_LMV2_SMALL_PROXY_FAST', '1')
    torch.manual_seed(15)
    x = torch.randn(32, 32, 128, device='cuda', dtype=torch.bfloat16)
    weights = torch.full((32, 32), 30, device='cuda', dtype=torch.long)
    caps = (64,) * 15
    for _ in range(3):
        cosine.build_cosine_directions(x, weights, caps)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = cosine.build_cosine_directions(x, weights, caps)
    for _ in range(2):
        x.copy_(torch.randn_like(x))
        graph.replay()
        assert torch.equal(actual, cosine.build_cosine_directions(x, weights, caps))


@pytest.mark.parametrize('caps,dtype', [((128,) * 15, torch.bfloat16),
                                      ((64,) * 8, torch.bfloat16),
                                      ((64,) * 15, torch.float32)])
def test_other_shapes_keep_legacy_arithmetic(monkeypatch, caps, dtype):
    x = torch.randn(8, 32, 128, device='cuda', dtype=dtype)
    weights = torch.full((8, 32), sum(caps) // 32, device='cuda', dtype=torch.long)
    monkeypatch.setenv('H3_LMV2_SMALL_PROXY_FAST', '0')
    expected = cosine.build_cosine_directions(x, weights, caps)
    monkeypatch.setenv('H3_LMV2_SMALL_PROXY_FAST', '1')
    assert torch.equal(expected, cosine.build_cosine_directions(x, weights, caps))
