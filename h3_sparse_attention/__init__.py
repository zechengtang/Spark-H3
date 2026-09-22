"""Sol-Attn integration and Spark primitives for MiniMax-H3."""
from .processor import (
    H3SparseAttentionConfig,
    H3SparseAttentionPlugin,
    install_h3_sol_attn,
    install_h3_spark_attn,
    install_h3_sparse_attention,
)

__all__ = [
    "H3SparseAttentionConfig", "H3SparseAttentionPlugin",
    "install_h3_spark_attn", "install_h3_sol_attn", "install_h3_sparse_attention",
    "spark_reblock", "spark_reweight", "spark_block",
    "Fp8Linear", "convert_linear_to_fp8", "install_fp8", "install_fused_blocks",
]


def __getattr__(name):
    # Keep the optional CUDA/Triton Spark kernels lazy for Sol-only callers.
    if name in ("spark_reblock", "spark_reweight", "spark_block"):
        from . import spark
        return getattr(spark, name)
    # Same for the fp8 path: opt-in, Triton/CUDA-only.
    if name in ("Fp8Linear", "convert_linear_to_fp8", "install_fp8"):
        from . import fp8_linear
        return getattr(fp8_linear, name)
    if name == "install_fused_blocks":
        from .fused_block import install_fused_blocks
        return install_fused_blocks
    raise AttributeError(name)
