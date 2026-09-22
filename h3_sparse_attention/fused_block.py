"""Inference-only kernels for the pointwise parts of a DiT block: the two pre-norms,
the two AdaLN affines, the two gated residual adds, and the feed-forward's SwiGLU
activation.

Ported from vdn-minimax-h3's src/models/ops/fused_block.py; see PORT_MANIFEST.json.

WHY THIS FILE EXISTS. Every one of those steps is pure elementwise work on the
[~15k, 5376] residual stream. Written the obvious way they run as seven separate
kernels that between them move several times more memory than the arithmetic needs.
Nothing here is clever: it is the same expressions, handed to inductor in one piece
so the intermediates stay in registers -- one read and one write per stage.

The AdaLN affine is the worst of them: `scale.index_select(0, adaln_indices)` and
`shift.index_select(0, adaln_indices)` each materialise a full-size gathered copy of
a table with a handful of distinct rows, purely so a broadcast can consume it. Fused,
the gather happens per element in the kernel and never reaches memory.

INFERENCE ONLY, installed by `install_fused_blocks`. Two reasons it is not simply the
block's forward:

  * `dynamic=False`. The whole point is that inductor gets to specialise on the real
    shape; a training run varies caption length every step and would recompile.
  * NOT BITWISE. Inductor holds the RMSNorm reciprocal and the affine chain in fp32
    and rounds once at the store, where eager rounds after every op. Same direction
    and the same size as fp8's caveat -- an improvement in accuracy, but a
    difference, and a latent_sha256 protocol must not silently take it.

The eager bodies stay here next to the fused ones as the reference, so what is being
claimed equal is visible in one screen.
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# The bodies. `_ref` is what diffusers' MiniMaxH3TransformerBlock.forward does inline.
# --------------------------------------------------------------------------------------

def _pre_ref(hidden, weight, eps, scale, shift, indices):
    """RMSNorm, then the AdaLN affine: x_hat * (1 + scale[i]) + shift[i]."""
    normed = F.rms_norm(hidden, (hidden.shape[-1],), weight, eps)
    return normed * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def _post_ref(residual, gate, indices, branch_out):
    """The gated residual add: residual + gate[i] * branch_out."""
    return residual + gate.index_select(0, indices) * branch_out


_CACHE = {}


def _compiled(name, body):
    """One compiled callable per body, built on first use.

    Separate entries rather than one compile of the whole block: the two halves are
    called with different tensors around an attention that is itself several compiled
    regions, and keeping the graphs small keeps a recompile cheap and its cause obvious.
    """
    if name not in _CACHE:
        _CACHE[name] = torch.compile(body, dynamic=False)
    return _CACHE[name]


def _swiglu_ref(h):
    """diffusers SwiGLU.forward after its projection: a * silu(gate)."""
    a, gate = h.chunk(2, dim=-1)
    return a * F.silu(gate)


def _fast_ff_forward(self, hidden_states):
    """Drop-in for diffusers' FeedForward.forward ([SwiGLU, Dropout, Linear]) under
    inference: the projection GEMM, the activation as ONE kernel, the down GEMM. Eager,
    `chunk -> silu -> mul` over the [T, 28672] projection output runs well below
    bandwidth; compiled it sits at the bandwidth floor. With the down projection in fp8
    the activation is fused with its quantisation instead, so the bf16 [T, 14336]
    intermediate is never written. Same non-bitwise caveat as everything else in this
    file."""
    swiglu, _, down = self.net
    shape = hidden_states.shape
    h = swiglu.proj(hidden_states.reshape(-1, shape[-1]))

    if hasattr(down, "forward_quantized"):
        from .fp8_linear import swiglu_quantize_activation
        out = down.forward_quantized(*swiglu_quantize_activation(h), out_dtype=h.dtype)
    else:
        out = down(_compiled("swiglu", _swiglu_ref)(h))

    return out.reshape(*shape[:-1], -1)


def _fast_block_forward(self, hidden_states, temb, adaln_indices, rotary_emb,
                        attention_mask=None):
    """Drop-in for MiniMaxH3TransformerBlock.forward under inference.

    Line for line the same as the original; only the two pointwise stretches around each
    sub-layer are handed to one kernel instead of four.
    """
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
    pre, post = _compiled("pre", _pre_ref), _compiled("post", _post_ref)

    normed = pre(hidden_states, self.norm1.weight, self.norm1.eps,
                 scale_msa, shift_msa, adaln_indices)
    hidden_states = post(hidden_states, gate_msa, adaln_indices,
                         self.attn(normed, rotary_emb, attention_mask))

    normed = pre(hidden_states, self.norm2.weight, self.norm2.eps,
                 scale_mlp, shift_mlp, adaln_indices)
    hidden_states = post(hidden_states, gate_mlp, adaln_indices, self.ff(normed))
    return hidden_states


def install_fused_blocks(pipe_or_transformer):
    """Patch every block in `transformer_blocks` with the fused inference forward, and
    every diffusers-shaped FeedForward with the single-kernel SwiGLU one. One way, like
    `install_fp8`; call it BEFORE the pipeline's torch.compile so the patched forwards
    are what gets traced. Compose-safe with `install_fp8` in either order: fp8 blocks
    get their `ff` replaced by Fp8SwiGLUFeedForward outright, and the block fusion only
    wraps the pointwise stages around whatever `attn`/`ff` are present. Returns how
    many blocks were patched."""
    transformer = getattr(pipe_or_transformer, "transformer", pipe_or_transformer)
    blocks = getattr(transformer, "transformer_blocks", None)
    if not blocks:
        raise ValueError("install_fused_blocks expects a model with a non-empty "
                         "`transformer_blocks` ModuleList")
    for block in blocks:
        block.forward = types.MethodType(_fast_block_forward, block)
        ff = getattr(block, "ff", None)
        net = getattr(ff, "net", None)
        if (net is not None and len(net) == 3
                and isinstance(getattr(net[0], "proj", None), (nn.Linear,))
                and getattr(net[1], "p", 0) == 0):
            ff.forward = types.MethodType(_fast_ff_forward, ff)
    return len(blocks)
