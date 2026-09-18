import sys,types
from pathlib import Path
import pytest
import torch
from torch import nn
import h3_sparse_attention.processor as proc

@pytest.mark.parametrize('fused',[False,True])
@pytest.mark.parametrize('context',[0,3])
@pytest.mark.parametrize('reblock',[False,True])
def test_sol_layout_roundtrip_exact(monkeypatch,fused,context,reblock):
    torch.manual_seed(42)
    n=128+context;heads=2;hidden=256
    perm=torch.randperm(n);inverse=perm.argsort()
    layout=proc.PackedLayout(perm,inverse,(2,8,8),128,n,torch.zeros(128,3,dtype=torch.long))
    config=proc.H3SparseAttentionConfig.sol(2,warmup_percent=0,sol_dense_layers=0,sol_log_density=False,sol_landmark_preprocess=reblock,sol_landmark_preprocess_version='v2')
    controller=proc._Controller(config);controller.layout=layout;controller.evaluation_index=0
    def fake_sol(q,k,v,**kw):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        return q*.25+k*.5+v
    package=types.ModuleType('sol_attn');package.sol_attn=fake_sol;package.get_sol_attn_backend=lambda device:'test'
    monkeypatch.setitem(sys.modules,'sol_attn',package)
    if reblock:
        qp=torch.stack([torch.randperm(128) for _ in range(heads)])[None]
        kp=torch.stack([torch.randperm(128) for _ in range(heads)])[None]
        monkeypatch.setattr(__import__('h3_sparse_attention.spark_integration',fromlist=['_landmark_tree_v2_qk_block_permutations']),'_landmark_tree_v2_qk_block_permutations',lambda *a:(qp,qp.argsort(-1),kp,kp.argsort(-1),{},{}))
    linear=lambda size:nn.Linear(hidden,size,bias=False,dtype=torch.bfloat16)
    attn=types.SimpleNamespace(heads=heads,fused_projections=fused,to_qkv=linear(3*hidden),to_q=linear(hidden),to_k=linear(hidden),to_v=linear(hidden),norm_q=nn.Identity(),norm_k=nn.Identity(),to_out=[linear(hidden),nn.Identity()])
    processor=proc._H3SparseProcessor(0,None,controller)
    x=torch.randn(1,n,hidden,dtype=torch.bfloat16)
    with torch.no_grad():
        monkeypatch.setattr(proc,'_SOL_LAYOUT_FAST',False)
        before=processor(attn,x)
        monkeypatch.setattr(proc,'_SOL_LAYOUT_FAST',True)
        after=processor(attn,x)
    assert torch.equal(before,after)
