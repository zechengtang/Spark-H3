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
]


def __getattr__(name):
    # Keep the optional CUDA/Triton Spark kernels lazy for Sol-only callers.
    if name in ("spark_reblock", "spark_reweight", "spark_block"):
        from . import spark
        return getattr(spark, name)
    raise AttributeError(name)
