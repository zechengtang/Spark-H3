import torch
from h3_sparse_attention.landmark_chunk import chunk_frame_lengths, chunk_permutation


def test_chunk_remainder():
    assert chunk_frame_lengths(67,10)==(10,)*6+(7,)
    assert chunk_frame_lengths(72,10)==(10,)*6+(12,)
    assert chunk_frame_lengths(72,5)==(5,)*13+(7,)
    assert chunk_frame_lengths(3,10)==(3,)


def test_chunk_tokens_and_leaf_membership(monkeypatch):
    monkeypatch.setenv('H3_TEMPORAL_MIN_FRAMES','0')
    torch.manual_seed(7)
    x=torch.randn(2,72*16,8)
    for size in [5,10]:
        perm, inv, _, info=chunk_permutation(x,chunk_frames=size,grid_shape=(72,4,4),
            validate=True,optimized_means=False,reuse_group4=False,distance='euclidean',
            order_mode='parent_order',group_size=1,max_children=(16,16),landmark_mode='midpoint',landmark_count=32,aggregation='linear')
        expected=torch.arange(x.shape[1]).expand(2,-1)
        assert torch.equal(perm.sort(-1).values,expected)
        assert torch.equal(perm.gather(1,inv),expected)
        offset=0;cursor=0
        for length in info['latent_frame_lengths']:
            n=length*16;complete=n//64*64
            part=perm[:,cursor:cursor+complete]
            assert ((part>=offset)&(part<offset+n)).all()
            cursor+=complete;offset+=n
        assert x.shape[1]-cursor==info['tail_tokens']
