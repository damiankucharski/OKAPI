"""Prediction-level shrinkage helpers for OKAPI ensembles.

The functions in this module are deliberately independent from the evolutionary
loop. They can be used for post-hoc saved-front analysis now and for a later
fitness-wrapped in-training condition without adding a new tree node type.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def shrink_prediction(tree_prediction: T, provider_prediction: T, alpha: float) -> T:
    """Blend a candidate prediction with a robust provider prediction.

    ``alpha=0`` returns the original candidate prediction and ``alpha=1`` returns
    the provider prediction. The function works with NumPy arrays, PyTorch
    tensors, and other array-like objects supporting scalar arithmetic.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return (1.0 - alpha) * tree_prediction + alpha * provider_prediction


def average_predictions(predictions: Sequence[T]) -> T:
    """Return the arithmetic mean of prediction tensors/arrays."""
    if not predictions:
        raise ValueError("cannot average an empty prediction sequence")
    total = predictions[0]
    for prediction in predictions[1:]:
        total = total + prediction
    return total / len(predictions)
