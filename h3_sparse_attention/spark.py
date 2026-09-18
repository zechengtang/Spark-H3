"""Public Spark names for the existing LMv2 and reweighting implementations.

Usage::

    from h3_sparse_attention.spark import spark_reblock, spark_reweight

``spark_reblock`` is the landmark-tree-v2 block builder. For arbitrary
scheduling, ``root_fanout=None`` and ``final_fanout=None`` both inherit
``fanout`` (default8). Set either explicitly to change the first nonfinal
or final round. ``max_children`` remains a compatibility alias.
The result includes the permutation and inverse permutation.

``spark_reweight(a, k, v)`` consumes query representatives ``a[B,P,H,D]``
and keys/values ``k,v[B,T,H,D]``. It returns query-conditioned weighted
keys, weighted values, and the log-mass correction ``(ak, av, lm)``.
It does not build query representatives, select routes, compute the exact
attention branch, or merge attention outputs.

The two tensor functions are direct aliases, not wrappers.
`install_h3_spark_attn` is the model-level context-manager installer; its
configuration defaults to minimum-10, power-of-two fanout-16 reblock and target-189 reweight.
See docs/SPARK_INSTALLER.md.

For the tensor aliases, Original names, signatures, internal
call sites, kernel names and configuration fields remain unchanged. Function
introspection therefore still reports the original implementation names.
"""

from .landmark_tree_v2 import recursive_landmark_tree_v2_blocks as spark_reblock
from .sol_numerator_virtual_q import virtual_summaries as spark_reweight
from .processor import install_h3_spark_attn

# Preserve callers of the initial public name.
spark_block = spark_reblock

__all__ = ["spark_reblock", "spark_reweight", "spark_block", "install_h3_spark_attn"]
