"""
Provides convenient aliases for operator types.

This module exposes the various Operation types from okapi.operation
with simpler names for easier importing and usage.

The deprecated node wrappers are also re-exported for backward compatibility.
"""

from okapi.operation import CloseThresholdOp, FarThresholdOp, MaxOp, MeanOp, MinOp, WeightedMeanOp

# Operation types available for use in trees
MIN = MinOp
MAX = MaxOp
MEAN = MeanOp
WEIGHTED_MEAN = WeightedMeanOp
FAR_THRESHOLD = FarThresholdOp
CLOSE_THRESHOLD = CloseThresholdOp
