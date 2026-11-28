"""A paired lower-confidence fitness for evaluating every program during native search."""

from collections.abc import Callable, Mapping
from numbers import Integral, Real

import numpy as np

from okapi.node import MeanNode, ValueNode
from okapi.tree import Tree


def _as_array(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _loss_array(values):
    values = _as_array(values)
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("Losses must be finite real numbers")
    return values.astype(np.float64, copy=False)


def _group_index(groups, rows):
    # Preserve first appearance so renaming groups leaves a seeded sign draw unchanged.
    ids = np.asarray(groups, dtype=object)
    if ids.shape != (rows,):
        raise ValueError("Supply one group ID per row")
    order = {}
    membership = np.empty(rows, dtype=np.intp)
    for i, key in enumerate(ids):
        if not isinstance(key, (str, Integral, Real)) or (isinstance(key, Real) and not np.isfinite(key)):
            raise ValueError("Group IDs must be strings or finite real numbers")
        membership[i] = order.setdefault(key, len(order))
    return membership, np.bincount(membership)


def arithmetic_mean_tree(tensors: Mapping) -> Tree:
    """Build the native equal-weight arithmetic mean using every provider once."""
    if not tensors:
        raise ValueError("The arithmetic mean needs at least one provider")
    operands = [ValueNode(None, value, key) for key, value in tensors.items()]
    root, *others = operands
    if others:
        root.add_child(MeanNode(others))
    return Tree(root)


class PairedGainFitness:
    """Score a program by mean paired gain minus z times the standard error of its gain.

    Bind one instance per data split to its fixed arithmetic-mean prediction,
    ordered labels and lower-is-better row_loss(prediction, labels) -> (N,).
    The callable fits the native GP, pattern-screen and head-optimization objective
    interfaces. Every score uses the same rows that those routines already use.

    Gains are the baseline's row losses minus the program's. With group IDs,
    average within each group, giving independent groups equal weight. For U
    units the penalty uses the sample standard deviation divided by sqrt(U).
    At U=1, z>0 yields the neutral baseline score for every program, while
    z=0 yields empirical gain. U=0 provides no fitting data.

    The standard error assumes independent units. Adaptive search can favor
    noisy improvements even after penalization. This is a search objective
    in loss units. Its score is not a calibrated confidence bound.
    """

    def __init__(self, baseline_prediction, labels, row_loss: Callable, *, z: float = 1.0, groups=None):
        if isinstance(z, (bool, np.bool_)) or not isinstance(z, Real) or not np.isfinite(z) or z < 0:
            raise ValueError("z must be a finite nonnegative real number")
        if not callable(row_loss):
            raise TypeError("row_loss must be callable")
        self.z = float(z)
        self._labels = _as_array(labels).copy()
        if self._labels.ndim == 0 or not len(self._labels) or len(baseline_prediction) != len(self._labels):
            raise ValueError("Supply nonempty aligned baseline predictions and labels")
        self._row_loss = row_loss
        self._baseline_losses = self._losses(baseline_prediction, labels).copy()
        self._groups = None if groups is None else _group_index(groups, len(self._labels))
        self.n_units = len(self._labels) if self._groups is None else len(self._groups[1])

    def _losses(self, prediction, labels):
        if len(prediction) != len(self._labels):
            raise ValueError("Prediction rows must match the bound baseline")
        losses = _loss_array(self._row_loss(prediction, labels))
        if losses.shape != (len(self._labels),):
            raise ValueError("row_loss must return one scalar per row with shape (N,)")
        return losses

    def __call__(self, prediction, labels) -> float:
        if not np.array_equal(_as_array(labels), self._labels):
            raise ValueError("Labels differ from the bound split or row order")
        gains = self._baseline_losses - self._losses(prediction, labels)
        if self._groups is not None:
            membership, counts = self._groups
            gains = np.bincount(membership, weights=gains) / counts
        average = float(gains.mean())
        if self.z == 0:
            return average
        if self.n_units == 1:
            return 0.0
        return average - float(self.z * gains.std(ddof=1) / np.sqrt(self.n_units))
