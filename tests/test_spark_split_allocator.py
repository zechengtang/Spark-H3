"""Persistent split statistics must not escape temporary malloc-graph scopes."""
import os
from pathlib import Path
import subprocess
import sys


def test_split_statistics_with_comfy_malloc_graph():
    import pytest
    torch = pytest.importorskip("torch")
    pytest.importorskip("comfy_aimdo.control")
    if not torch.cuda.is_available() or not os.environ.get("H3_SPARK_SPLIT_LIBRARY"):
        pytest.skip("CUDA and built split producer required")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    # AIMDO must install its allocator hooks before torch loads CUDA libraries.
    result = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _exercise_allocator():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    import comfy_aimdo.control as control
    assert control.init()
    import torch
    assert control.init_devices([0])
    import comfy_aimdo.malloc_graph as malloc_graph
    from comfy_kitchen import set_allocation_context
    from comfyui_backend import ComfyPackedLayout, ComfySparkConfig, ComfySparkController
    from comfyui_spark_split import spark_split_attention

    torch.set_grad_enabled(False)
    torch.manual_seed(31)
    tokens, heads = 2048, 4
    qkv = torch.randn(tokens, 3 * heads * 128, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(1, tokens, 1, 32, 2, 2, device="cuda")
    weights = (torch.ones(128, device="cuda", dtype=torch.bfloat16),) * 2
    indices = torch.arange(tokens, device="cuda")
    layout = ComfyPackedLayout(
        indices, indices, (2, 32, 32), tokens, tokens,
        torch.zeros(tokens, 3, device="cuda"), 0,
    )
    controller = ComfySparkController(ComfySparkConfig(topk_mode="topk_ratio", topk_ratio=.2))
    controller.direct_output = True
    stats = {i: None for i in range(4)}

    def provider(kind, *, indices=None, start=None, stop=None):
        raw = qkv.index_select(0, indices) if indices is not None else qkv[start:stop]
        q, k, v = raw.split(heads * 128, dim=-1)
        return {"q": q, "k": k, "kv": raw[:, heads * 128:], "qkv": raw, "qv": (q, v)}[kind]

    def attention(layer):
        out, stats[layer] = spark_split_attention(
            controller, provider, tokens, heads, rope, weights, 1e-6, layout,
            stats[layer], scheme="drop_k", chunk=512,
        )
        return out

    # Use the same number of steps: quantized statistics can evolve on replay.
    for _ in range(12):
        for layer in range(4):
            out = attention(layer)
    expected = (out.cpu(), *(s.cpu() for s in stats[3]))
    del out
    stats = {i: None for i in range(4)}
    torch.cuda.synchronize()
    graph = malloc_graph.record(torch.cuda.current_stream())

    class PersistentAllocation:
        def __enter__(self):
            graph.pause()

        def __exit__(self, *args):
            graph.resume()

    set_allocation_context(PersistentAllocation())
    records = []
    try:
        for step in range(12):
            if step:
                graph.push()
            for layer in range(4):
                graph.iterate("block")
                out = attention(layer)
                if step == 11 and layer == 3:
                    actual = (out.cpu(), *(s.cpu() for s in stats[layer]))
                del out
            graph.iterate(None)
            graph.pop()
            torch.cuda.synchronize()
            records.append((graph.physical_bytes, graph.rogue_count))
    finally:
        set_allocation_context(None)
        stats.clear()
        del graph
        torch.cuda.synchronize()
        control.deinit()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert all(rogues == 0 for _, rogues in records), records
    assert len({size for size, _ in records[2:]}) == 1, records
    print(records)


if __name__ == "__main__":
    _exercise_allocator()
