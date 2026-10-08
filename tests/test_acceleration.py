from types import SimpleNamespace

import torch

from h3_sparse_attention import H3AccelerationConfig, install_h3_acceleration


class _Attention:
    def forward(self, x):
        return x


class _Block:
    def __init__(self):
        self.attn = _Attention()

    def forward(self, x):
        return self.attn.forward(x)


def test_compile_default_installs_and_restores_per_block(monkeypatch):
    blocks = [_Block(), _Block()]
    pipe = SimpleNamespace(transformer=SimpleNamespace(transformer_blocks=blocks))
    compile_calls = []
    disable_calls = []

    def fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return lambda *args, **kw: fn(*args, **kw)

    def fake_disable(fn):
        disable_calls.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.compiler, "disable", fake_disable)
    monkeypatch.setattr(torch._inductor, "list_mode_options", lambda mode: {})

    config = H3AccelerationConfig()
    assert config.torch_compile is True
    assert config.compile_reproducibility_fix is True
    assert config.vae_fp16 is False
    with install_h3_acceleration(pipe, config) as plugin:
        assert plugin.config is config
        assert len(compile_calls) == len(disable_calls) == 2
        assert all(call["options"]["deterministic"] for call in compile_calls)
        assert all("forward" in block.__dict__ for block in blocks)
        assert all("forward" in block.attn.__dict__ for block in blocks)
    assert all("forward" not in block.__dict__ for block in blocks)
    assert all("forward" not in block.attn.__dict__ for block in blocks)


def test_reproducibility_fix_can_restore_legacy_whole_block_compile(monkeypatch):
    blocks = [_Block(), _Block()]
    pipe = SimpleNamespace(transformer=SimpleNamespace(transformer_blocks=blocks))
    compile_calls = []

    def fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return lambda *args, **kw: fn(*args, **kw)

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        torch.compiler,
        "disable",
        lambda fn: (_ for _ in ()).throw(
            AssertionError("legacy mode must not install the eager attention boundary")
        ),
    )
    monkeypatch.setattr(
        torch._inductor,
        "list_mode_options",
        lambda mode: (_ for _ in ()).throw(
            AssertionError("legacy mode must not install deterministic options")
        ),
    )

    config = H3AccelerationConfig(compile_reproducibility_fix=False)
    with install_h3_acceleration(pipe, config):
        assert all("forward" in block.__dict__ for block in blocks)
        assert all("forward" not in block.attn.__dict__ for block in blocks)
    assert compile_calls == [
        {"mode": None, "dynamic": None},
        {"mode": None, "dynamic": None},
    ]


def test_explicit_no_compile_is_not_a_noop_fallback(monkeypatch):
    block = _Block()
    pipe = SimpleNamespace(transformer=SimpleNamespace(transformer_blocks=[block]))
    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("compile should not be called")
    ))
    with install_h3_acceleration(pipe, H3AccelerationConfig(torch_compile=False)):
        assert "forward" not in block.__dict__
