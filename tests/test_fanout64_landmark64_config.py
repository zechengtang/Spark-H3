import pytest
import torch

from h3_sparse_attention.landmark_tree_v2 import (
    PreparedLandmarkTreeV2Permutation,
    _normalize_children,
)
from h3_sparse_attention.processor import H3SparseAttentionConfig
from h3_sparse_attention.reblock_hierarchy import tree_frontiers


def test_spark_accepts_fanout64_landmark64():
    config = H3SparseAttentionConfig.spark(
        20,
        landmark_tree_v2_fanout=64,
        landmark_tree_v2_landmark_count=64,
    )
    assert config.landmark_tree_v2_children == 64
    assert config.landmark_tree_v2_fanout == 64
    assert config.landmark_tree_v2_landmark_count == 64


def test_fanout64_hierarchy_covers_leaves_exactly():
    assert _normalize_children(64) == 64
    levels = tree_frontiers(582, 64)
    leaves = levels[-1]
    assert leaves[0][0] == 0
    assert leaves[-1][1] == 582
    assert len(leaves) == 582
    assert all(end - start == 1 for start, end in leaves)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fanout64_landmark64_cuda_graph_is_a_permutation():
    torch.manual_seed(42)
    tokens = 1134 * 64 + 17
    samples = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)
    plan = PreparedLandmarkTreeV2Permutation(
        batch=1,
        tokens=tokens,
        dim=128,
        grid_shape=(1, 1, tokens),
        device="cuda",
        fanout=64,
        landmark_count=64,
    )
    plan.run(samples)
    permutation, inverse = plan.run(samples)
    expected = torch.arange(tokens, device="cuda")[None]
    assert plan.graph_active
    assert plan.hierarchy.metadata()["fanout"] == 64
    assert torch.equal(permutation.gather(1, inverse), expected)
