"""SM90/SM100 fused virtual-query (Spark reweight) kernel tests.

The wiring tests run anywhere (backend probing never constructs a kernel).
The numerical tests are gated on SM90/SM100 hardware and compare the fused
CuTe kernel against the existing fallback (exact_attention + streamed virtual
merge), which shares the same Spark semantics.
"""
from __future__ import annotations

import pytest
import torch

FUSED_CAPS = ((9, 0), (10, 0), (12, 0))

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
requires_new_fused_backend = pytest.mark.skipif(
    not torch.cuda.is_available()
    or tuple(torch.cuda.get_device_capability(0)) not in ((9, 0), (10, 0)),
    reason="SparkReweightForwardSm90/Sm100 require SM90/SM100 hardware",
)


# --- wiring tests (run on any arch) ---------------------------------------


@requires_cuda
@pytest.mark.parametrize(
    "cap,expected",
    [
        ((9, 0), "sm90_fused_virtual_query"),
        ((10, 0), "sm100_fused_virtual_query"),
        ((12, 0), "sm120_fused_virtual_query"),
        ((8, 0), "stock_exact+triton_virtual_query_skipped"),
    ],
)
def test_virtual_q_backend_strings(monkeypatch, cap, expected):
    import h3_sparse_attention.sol_numerator_virtual_q as rw

    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "1")
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *a, **k: cap
    )
    q = torch.zeros(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
    assert rw.virtual_q_backend(q) == expected


@requires_cuda
def test_virtual_q_backend_auto_length_gate(monkeypatch):
    import h3_sparse_attention.sol_numerator_virtual_q as rw

    monkeypatch.delenv("H3_SPARK_REWEIGHT_FUSED", raising=False)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *a, **k: (10, 0)
    )
    short = torch.zeros(1, 8192, 2, 128, device="cuda", dtype=torch.bfloat16)
    long = torch.zeros(1, 8193, 2, 128, device="cuda", dtype=torch.bfloat16)
    assert rw.virtual_q_backend(short) == "stock_exact+triton_virtual_query_skipped"
    assert rw.virtual_q_backend(long) == "sm100_fused_virtual_query"


def test_fused_kernel_modules_importable():
    import h3_sparse_attention.spark_reweight_sm90 as m90
    import h3_sparse_attention.spark_reweight_sm100 as m100

    assert hasattr(m90, "SparkReweightForwardSm90")
    assert hasattr(m100, "SparkReweightForwardSm100")


# --- numerical tests (SM90/SM100 hardware only) ---------------------------


def _topology(t, parents=2):
    """Split the leaf blocks into equal contiguous virtual parents."""
    from h3_sparse_attention.sol_numerator_virtual_q import validate_virtual_layout

    n = (t + 63) // 64
    per = n // parents
    ranges = [[i * per * 64, (i + 1) * per * 64] for i in range(parents)]
    ranges[-1][1] = t
    mapping = [min(i // per, parents - 1) for i in range(n)]
    ranges_host, mapping_host = validate_virtual_layout(ranges, mapping, t)
    return (
        torch.tensor(ranges_host, device="cuda", dtype=torch.int64),
        torch.tensor(mapping_host, device="cuda", dtype=torch.int64),
    )


def _virtual_q_inputs(b=2, t=512, h=4, d=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (
        torch.randn(b, t, h, d, generator=g, device="cuda", dtype=torch.float32)
        .to(torch.bfloat16)
        for _ in range(3)
    )
    from sol_attn.preprocess import _reduce_kv

    kc, vs = _reduce_kv(k, v)
    return q, k, v, kc, vs


@requires_new_fused_backend
@pytest.mark.parametrize("mode", ["threshold", "external_route"])
def test_fused_virtual_query_matches_fallback(monkeypatch, mode):
    import h3_sparse_attention.sol_numerator_virtual_q as rw

    torch.manual_seed(11)
    q, k, v, kc, vs = _virtual_q_inputs()
    b, t, h, _ = q.shape
    n = kc.shape[1]
    ranges, mapping = _topology(t)
    g = torch.Generator(device="cuda").manual_seed(5)
    if mode == "threshold":
        threshold = torch.randn(
            b, n, h, generator=g, device="cuda", dtype=torch.float32
        ) * 0.5
        route = None
    else:
        threshold = None
        route = (
            torch.rand(b, n, h, n, generator=g, device="cuda") > 0.5
        ).to(torch.uint8)
        route[:, 0, :, 0] = 1

    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "1")
    fused = rw.virtual_q_attention(
        q, k, v, virtual_ranges=ranges, leaf_to_virtual=mapping,
        key_centroids=kc, value_sums=vs, threshold=threshold, route=route,
        sink_tokens=64,
    )
    assert rw.virtual_q_backend(q).endswith("fused_virtual_query")

    monkeypatch.setenv("H3_SPARK_REWEIGHT_FUSED", "0")
    fallback = rw.virtual_q_attention(
        q, k, v, virtual_ranges=ranges, leaf_to_virtual=mapping,
        key_centroids=kc, value_sums=vs, threshold=threshold, route=route,
        sink_tokens=64,
    )
    # Fused kernel and streamed fallback accumulate in different orders;
    # tolerances are wider than the SDPA parity tests and may need
    # calibration on real hardware.
    torch.testing.assert_close(fused.float(), fallback.float(), atol=0.02, rtol=0.03)
