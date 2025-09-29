"""
Provides convenient aliases for operator node types.

This module exposes the various operator node types from giraffe.node
with simpler names for easier importing and usage.
"""

from giraffe.node import CloseThresholdNode, FarThresholdNode, MaxNode, MeanNode, MinNode, WeightedMeanNode

# Operator node types available for use in trees
MIN = MinNode
MAX = MaxNode
MEAN = MeanNode
WEIGHTED_MEAN = WeightedMeanNode
FAR_THRESHOLD = FarThresholdNode
CLOSE_THRESHOLD = CloseThresholdNode
