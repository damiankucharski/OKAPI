"""
Provides convenient aliases for operator node types.

This module exposes the various operator node types from okapi.node
with simpler names for easier importing and usage.
"""

from okapi.node import (
    CloseThresholdNode,
    FarThresholdNode,
    LogitMeanNode,
    MaxNode,
    MeanNode,
    MinNode,
    WeightedLogitMeanNode,
    WeightedMeanNode,
)

# Operator node types available for use in trees
MIN = MinNode
MAX = MaxNode
MEAN = MeanNode
WEIGHTED_MEAN = WeightedMeanNode
FAR_THRESHOLD = FarThresholdNode
CLOSE_THRESHOLD = CloseThresholdNode
LOGIT_MEAN = LogitMeanNode
WEIGHTED_LOGIT_MEAN = WeightedLogitMeanNode
