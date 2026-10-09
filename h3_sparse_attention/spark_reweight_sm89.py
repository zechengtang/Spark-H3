"""Ada SM89 fused Spark kernel.

Ada supports the warp MMA and cp.async instructions used by the SM80
implementation. CuTe compiles the inherited kernel for the active SM89 device;
the route, summary and online-softmax contract matches the SM120 kernel.
"""

from .spark_reweight_sm80 import SparkReweightForwardSm80


class SparkReweightForwardSm89(SparkReweightForwardSm80):
    """Compile the cp.async Spark mainloop for Ada SM89."""


__all__ = ["SparkReweightForwardSm89"]
