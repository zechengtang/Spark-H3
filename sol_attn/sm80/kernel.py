"""Native Triton kernel entry point for Ampere SM80.

SM80 has no TMA, so this backend intentionally uses the pointer-based Triton
mainloop and preprocessing kernels.  It is selected before launch and never
serves as a recovery path for a failed CuTe kernel.
"""

from ..triton_ref.fwd import sol_attn

__all__ = ["sol_attn"]
