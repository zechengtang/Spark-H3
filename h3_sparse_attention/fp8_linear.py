"""fp8 for the DiT's big Linears. OPT-IN, NEVER A DEFAULT.

Ported from vdn-minimax-h3's src/models/ops/fp8_linear.py; see PORT_MANIFEST.json.

WHY IT IS HERE. Once the pointwise work is fused, a block's time is dominated by GEMMs
and attention already running near tensor-core peak; fp8 e4m3 is the remaining lever,
roughly doubling each of the block's GEMMs.

WHY IT IS NOT THE DEFAULT. fp8 does not degrade the output -- it CHANGES it. A single
step's velocity prediction stays close to bf16's (cosine ~0.998), but a denoising
trajectory is chaotic: perturb it early and it settles into a different mode, so the
final latents and the decoded clip are a different (not worse) sample of the same
prompt. fp8 IS NOT A DROP-IN: anything that depends on reproducing a previous render
(the latent_sha256 experiment protocols in this repo's scripts/, a checkpoint
comparison, a user re-rolling a seed) stops working.

WHAT IS QUANTISED. Only Linears at or above `min_width` on BOTH sides, and only inside
`model.transformer_blocks` -- the qkv projections, to_out, and the two feed-forward
GEMMs (the model is hidden 5376 / inner 7168 / ffn 14336, so the default 4096 keeps
exactly those). The scope is the blocks alone, not the whole module tree: the
context_embedder is wide enough to pass the width test but is a condition embedder,
not a timed GEMM, and the token_refiner's blocks sit outside `transformer_blocks`
where `skip_end_blocks` could not protect them. Everything narrow stays in bf16 on
purpose: the output gates are low-rank, the adaln table is tiny, and the fp32 islands
in `_keep_in_fp32_modules` are places precision matters more than throughput.

THE QUANTISER IS A TRITON KERNEL, one program per row: absmax, then the scaled cast,
read once from HBM (the second sweep over the row comes back out of L2); the compiled
torch spelling was several times slower than the bandwidth floor. Two callers avoid
extra quantisation passes: `Fp8AttnProcessor` quantises x once for the q/k/v
projections (`forward_qkv_shared`), and `Fp8SwiGLUFeedForward` fuses the SwiGLU
activation with the quantisation of its output (`swiglu_quantize_activation`), so the
bf16 intermediate is never written. Both are installed by the conversion entry points;
a bare `Fp8Linear.forward` quantises per call and needs neither.

WHERE THE ERROR COMES FROM. The WEIGHTS, not the activations: rowwise activation
scaling is no more accurate than per-tensor, while fp8 weights alone account for most
of the error. So the lever for accuracy, if anyone wants one, is finer weight
granularity (per-block) or keeping outlier channels in bf16 -- not a better activation
scale.

`skip_end_blocks` leaves the first and last N blocks in bf16; it keeps most of the
speedup and removes a disproportionate share of the error.

    from h3_sparse_attention.fp8_linear import install_fp8
    install_fp8(pipe.transformer)   # after load_components, before the render

The swap is ONE WAY: the bf16 weight is released as each Linear is replaced, so the
quantised model is roughly half the weight of the one it came from and there is nothing
to put back. Rendering bf16 again means loading the model again.

SCALE GRANULARITY FOLLOWS THE CARD, and this is a dispatch fact, not a numerics
preference. sm90 keeps rowwise activation scales: they are free there. sm100+ (this
repo targets sm120) uses PER-TENSOR scales on BOTH sides, because the torch builds
that support those cards route rowwise-scaled `_scaled_mm` to a generic CUTLASS
kernel that is barely faster than bf16 at these shapes -- on torch 2.12 / sm120 the
rowwise combination is refused outright ("Invalid scaling configuration") -- while
per-tensor x per-tensor dispatches to a cuBLAS kernel at the expected ~2x. The
accuracy cost is negligible on the real weights, because e4m3's dynamic range covers
their channel outliers without underflow. The granularity follows the capability
alone -- it is not a knob.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(FP8_DTYPE).max
MIN_WIDTH = 4096
SKIP_END_BLOCKS = 4


@triton.jit
def _quantize_rows_kernel(X, Y, S, K, FP8_MAX: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + row * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    scale = tl.maximum(tl.max(amax, axis=0) / FP8_MAX, 1e-12)
    tl.store(S + row, scale)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + row * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        y = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
        tl.store(Y + row * K + cols, y.to(Y.dtype.element_ty), mask=cols < K)


@triton.jit
def _swiglu_quantize_kernel(H, Y, S, K, FP8_MAX: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(H + row * 2 * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        g = tl.load(H + row * 2 * K + K + cols, mask=cols < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(a * g * tl.sigmoid(g)))
    scale = tl.maximum(tl.max(amax, axis=0) / FP8_MAX, 1e-12)
    tl.store(S + row, scale)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(H + row * 2 * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        g = tl.load(H + row * 2 * K + K + cols, mask=cols < K, other=0.0).to(tl.float32)
        y = tl.minimum(tl.maximum(a * g * tl.sigmoid(g) / scale, -FP8_MAX), FP8_MAX)
        tl.store(Y + row * K + cols, y.to(Y.dtype.element_ty), mask=cols < K)


def _rows(x):
    if not x.is_cuda:
        raise ValueError("the fp8 quantiser is a Triton kernel; the input must be on CUDA")
    if x.dim() != 2:
        raise ValueError(f"expected [M, K], got {tuple(x.shape)}")
    return x.contiguous()


def quantize_rows(x):
    """bf16 [M, K] -> (fp8 [M, K], fp32 scale [M, 1]). One absmax per row."""
    x = _rows(x)
    M, K = x.shape
    y = torch.empty_like(x, dtype=FP8_DTYPE)
    scale = torch.empty(M, 1, device=x.device, dtype=torch.float32)
    _quantize_rows_kernel[(M,)](x, y, scale, K, FP8_MAX=_FP8_MAX, BLOCK_K=1024, num_warps=4)
    return y, scale


def swiglu_quantize(h):
    """[M, 2K] projection output -> quantize_rows(a * silu(gate)) with a, gate = h.chunk(2),
    without writing the bf16 activation. Matches diffusers' SwiGLU chunk order
    (value first, gate second)."""
    h = _rows(h)
    M, K2 = h.shape
    K = K2 // 2
    y = torch.empty(M, K, device=h.device, dtype=FP8_DTYPE)
    scale = torch.empty(M, 1, device=h.device, dtype=torch.float32)
    _swiglu_quantize_kernel[(M,)](h, y, scale, K, FP8_MAX=_FP8_MAX, BLOCK_K=2048, num_warps=16)
    return y, scale


def quantize_rows_reference(x):
    """The eager spelling of `quantize_rows`, for the tests."""
    scale = (x.float().abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(1e-12)
    return (x.float() / scale).to(FP8_DTYPE), scale


_PER_TENSOR = None


def per_tensor_gemm():
    """sm100+ (incl. sm120) uses per-tensor scales (see the header); sm90 keeps rowwise."""
    global _PER_TENSOR
    if _PER_TENSOR is None:
        _PER_TENSOR = (torch.cuda.is_available()
                       and torch.cuda.get_device_capability(0)[0] >= 10)
    return _PER_TENSOR


@triton.jit
def _absmax_kernel(X, OUT, N, BLOCK: tl.constexpr):
    """Partial absmax + one atomic per program: a full reduction in ONE read pass,
    where eager `x.abs().amax()` materialises the abs (a full-size write at the qkv
    shape) before it reduces."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N, other=0.0).to(tl.float32)
    tl.atomic_max(OUT, tl.max(tl.abs(x), axis=0))


@triton.jit
def _cast_scaled_kernel(X, Y, S, N, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    scale = tl.load(S)
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_rowmax_kernel(H, Y, RM, K, BLOCK_K: tl.constexpr):
    """a * silu(gate) per row into bf16 Y, plus the row's absmax -- the per-tensor
    spelling of `_swiglu_quantize_kernel`, split so the global scale can be taken
    before the cast."""
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        mask = cols < K
        a = tl.load(H + row * 2 * K + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(H + row * 2 * K + K + cols, mask=mask, other=0.0).to(tl.float32)
        act = (a * g * tl.sigmoid(g)).to(Y.dtype.element_ty)
        # amax of the ROUNDED value: the bf16 intermediate is the tensor being
        # quantised, so its own absmax is the scale.
        amax = tl.maximum(amax, tl.abs(act.to(tl.float32)))
        tl.store(Y + row * K + cols, act, mask=mask)
    tl.store(RM + row, tl.max(amax, axis=0))


def quantize_tensor(x):
    """bf16 [M, K] -> (fp8 [M, K], fp32 scale [1, 1]). One absmax for the whole tensor;
    the scale never leaves the device (no sync)."""
    x = _rows(x)
    n = x.numel()
    amax = torch.zeros(1, device=x.device, dtype=torch.float32)
    _absmax_kernel[(triton.cdiv(n, 8192),)](x.view(-1), amax, n, BLOCK=8192,
                                            num_warps=8)
    scale = (amax / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
    y = torch.empty_like(x, dtype=FP8_DTYPE)
    _cast_scaled_kernel[(triton.cdiv(n, 8192),)](x.view(-1), y.view(-1), scale, n,
                                                 FP8_MAX=_FP8_MAX, BLOCK=8192,
                                                 num_warps=8)
    return y, scale


def quantize_tensor_reference(x):
    """The eager spelling of `quantize_tensor`, for the tests."""
    scale = (x.float().abs().amax() / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
    return (x.float() / scale).to(FP8_DTYPE), scale


def swiglu_quantize_tensor(h):
    """[M, 2K] projection output -> per-tensor quantised a * silu(gate). One extra
    bf16 write against the fused rowwise kernel; the fast GEMM it unlocks on sm100+
    repays it many times over."""
    h = _rows(h)
    M, K2 = h.shape
    K = K2 // 2
    act = torch.empty(M, K, device=h.device, dtype=h.dtype)
    rowmax = torch.empty(M, device=h.device, dtype=torch.float32)
    _swiglu_rowmax_kernel[(M,)](h, act, rowmax, K, BLOCK_K=2048, num_warps=16)
    scale = (rowmax.amax() / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
    y = torch.empty_like(act, dtype=FP8_DTYPE)
    n = act.numel()
    _cast_scaled_kernel[(triton.cdiv(n, 8192),)](act.view(-1), y.view(-1), scale, n,
                                                 FP8_MAX=_FP8_MAX, BLOCK=8192,
                                                 num_warps=8)
    return y, scale


def quantize_activation(x):
    """What the GEMM callers use: the scale granularity the current card's fast
    `_scaled_mm` path accepts."""
    return quantize_tensor(x) if per_tensor_gemm() else quantize_rows(x)


def swiglu_quantize_activation(h):
    return swiglu_quantize_tensor(h) if per_tensor_gemm() else swiglu_quantize(h)


class Fp8Linear(nn.Module):
    """A bias-optional Linear whose GEMM runs in fp8 e4m3.

    The weight is quantised ONCE at construction (per output channel on sm90, per tensor
    on sm100+); the activation is quantised on every call, or by the caller
    (`forward_quantized`) when one activation feeds several of these. Only the fp8 copy
    and the bias are kept -- the bf16 weight goes as soon as the caller drops the Linear
    it came from.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight
        if per_tensor_gemm():
            # sm100+: both scales must be scalar for the fast cuBLAS path; on the real
            # weights this costs nothing measurable over per-channel (see the header).
            scale = (weight.abs().amax().float() / _FP8_MAX).clamp_min(1e-12)
            self.register_buffer("weight_fp8",
                                 (weight / scale.to(weight.dtype)).to(FP8_DTYPE))
            self.register_buffer("weight_scale", scale.reshape(1, 1).contiguous())
        else:
            scale = (weight.abs().amax(dim=1, keepdim=True).float()
                     / _FP8_MAX).clamp_min(1e-12)
            self.register_buffer("weight_fp8",
                                 (weight / scale.to(weight.dtype)).to(FP8_DTYPE))
            self.register_buffer("weight_scale", scale.reshape(1, -1).contiguous())
        self.register_parameter("bias", linear.bias)

    def forward_quantized(self, x_fp8, x_scale, out_dtype=torch.bfloat16):
        """[M, K] fp8 rows and their scales ([M, 1] rowwise / [1, 1] per-tensor) -> [M, N]."""
        out = torch._scaled_mm(x_fp8, self.weight_fp8.t(),
                               scale_a=x_scale, scale_b=self.weight_scale,
                               out_dtype=out_dtype, use_fast_accum=True)
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x):
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        out = self.forward_quantized(*quantize_activation(rows), out_dtype=rows.dtype)
        return out.reshape(*shape[:-1], -1)


def forward_qkv_shared(attn, hidden_states):
    """One quantisation of hidden_states feeds the q/k/v projections when all three are
    fp8; returns None when any of them is bf16, so the caller falls back to the plain
    path. Bitwise identical to three separate `Fp8Linear.forward` calls (same kernel,
    same input), it just runs the quantiser once instead of three times."""
    projections = (attn.to_q, attn.to_k, attn.to_v)
    if not all(isinstance(p, Fp8Linear) for p in projections):
        return None
    rows = hidden_states.reshape(-1, hidden_states.shape[-1])
    x_fp8, x_scale = quantize_activation(rows)
    return tuple(
        p.forward_quantized(x_fp8, x_scale, out_dtype=hidden_states.dtype)
         .reshape(*hidden_states.shape[:-1], -1)
        for p in projections
    )


class Fp8SwiGLUFeedForward(nn.Module):
    """diffusers FeedForward([SwiGLU, Dropout(0), Linear]) with both GEMMs in fp8.

    The SwiGLU activation is fused with the quantisation of its output, so the bf16
    [M, ffn_dim] intermediate is never written. Installed by `convert_linear_to_fp8`
    in place of the original `ff` module; only valid at inference, the dropout must be
    a no-op for the fusion to be one kernel.
    """

    def __init__(self, ff):
        super().__init__()
        swiglu, dropout, down = ff.net
        if getattr(dropout, "p", 0) != 0:
            raise ValueError("Fp8SwiGLUFeedForward fuses the activation with the "
                             "quantisation; a live dropout cannot be fused")
        self.proj = swiglu.proj
        self.down = down

    def forward(self, hidden_states):
        shape = hidden_states.shape
        h = self.proj(hidden_states.reshape(-1, shape[-1]))
        out = self.down.forward_quantized(*swiglu_quantize_activation(h),
                                          out_dtype=h.dtype)
        return out.reshape(*shape[:-1], -1)


class Fp8AttnProcessor:
    """MiniMaxH3AttnProcessor with one quantisation of hidden_states shared by q/k/v.

    Installed per attention module by `install_fp8`, only where all three projections
    became Fp8Linear. The sparse plugin adopts whatever processor is current as its
    dense fallback, so installing this BEFORE the plugin makes the plugin's
    warmup/dense path fp8-shared as well, and the plugin's exit restores it.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        query, key, value = forward_qkv_shared(attn, hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _fuse_ff(block):
    """Replace block.ff with the fused fp8 FeedForward when the conversion above made
    both of its GEMMs fp8. No-op for the bf16 end blocks and for non-SwiGLU layouts."""
    ff = getattr(block, "ff", None)
    net = getattr(ff, "net", None)
    if net is None or len(net) != 3:
        return False
    swiglu, dropout, down = net
    if not (isinstance(getattr(swiglu, "proj", None), Fp8Linear)
            and isinstance(down, Fp8Linear)
            and getattr(dropout, "p", 0) == 0):
        return False
    block.ff = Fp8SwiGLUFeedForward(ff)
    return True


def convert_linear_to_fp8(model, min_width=MIN_WIDTH, skip_end_blocks=SKIP_END_BLOCKS):
    """Swap the wide Linears inside `model.transformer_blocks` for fp8 ones, in place,
    and fuse each converted block's SwiGLU FeedForward. Returns how many were swapped.

    Scoped to the blocks on purpose -- see the header for why the context_embedder and
    the token_refiner must stay bf16. `skip_end_blocks` keeps the first and last N
    blocks in bf16.

    One Linear at a time, and the bf16 one is unreferenced the moment its parent is
    repointed: peak memory is the model plus a single fp8 weight, and it falls from
    there. There is no way back -- see the header.
    """
    blocks = getattr(model, "transformer_blocks", None)
    if not blocks:
        raise ValueError("convert_linear_to_fp8 expects a model with a non-empty "
                         "`transformer_blocks` ModuleList")
    swapped = 0
    for index, block in enumerate(blocks):
        if index < skip_end_blocks or index >= len(blocks) - skip_end_blocks:
            continue
        for parent in list(block.modules()):
            for name, child in list(parent.named_children()):
                if not isinstance(child, nn.Linear):
                    continue
                if child.in_features < min_width or child.out_features < min_width:
                    continue
                setattr(parent, name, Fp8Linear(child).to(child.weight.device))
                swapped += 1
        _fuse_ff(block)
    return swapped


def install_fp8(pipe_or_transformer, min_width=MIN_WIDTH, skip_end_blocks=SKIP_END_BLOCKS):
    """Entry point for the inference scripts: accept the pipeline or the transformer
    itself, convert the Linears, fuse the FeedForwards, and install the shared-qkv
    attention processor on every converted block. Returns how many Linears were
    quantised. Call it AFTER the weights are loaded (pipe.load_components), BEFORE the
    sparse-attention plugin (so the plugin adopts the fp8 processor as its dense
    fallback), and before the render; converting first and loading later would
    silently overwrite the fp8 copies with bf16 weights."""
    transformer = getattr(pipe_or_transformer, "transformer", pipe_or_transformer)
    swapped = convert_linear_to_fp8(transformer, min_width=min_width,
                                    skip_end_blocks=skip_end_blocks)
    for block in transformer.transformer_blocks:
        attn = getattr(block, "attn", None)
        if (attn is not None and not attn.fused_projections
                and all(isinstance(getattr(attn, name, None), Fp8Linear)
                        for name in ("to_q", "to_k", "to_v"))):
            attn.set_processor(Fp8AttnProcessor())
    return swapped
