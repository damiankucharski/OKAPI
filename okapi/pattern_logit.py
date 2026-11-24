"""Value-only coordinate search over native signed logit fusion weights."""

from collections.abc import Callable, Sequence

import numpy as np

from okapi.functions import scale_vector_to_sum_1
from okapi.globals import BACKEND as B
from okapi.globals import _passthrough, use_postprocessing_function
from okapi.node import BinaryAffineNode, ValueNode, WeightedLogitMeanNode
from okapi.tree import Tree


def _native_sum(columns, weights):
    # Both native operators accumulate in provider order. The first product
    # owns its storage and fixes the accumulator dtype for the multiclass node.
    total = columns[0] * float(weights[0])
    for col, weight in zip(columns[1:], weights[1:], strict=True):
        total += col * float(weight)
    return total


def _search(columns, labels, objective, multiclass):
    count = len(columns)
    evaluations = 0

    def score(slope, bias):
        nonlocal evaluations
        if multiclass:
            prediction = scale_vector_to_sum_1(WeightedLogitMeanNode._from_sum(slope, False))
        else:
            prediction = BinaryAffineNode._probability(slope + bias)
        evaluations += 1
        value = float(objective(prediction, labels))
        if not np.isfinite(value):
            raise ValueError("objective must return a finite scalar")
        return value

    weights = np.zeros(count)
    bias = 0.0
    slope = _native_sum(columns, weights)
    current = score(slope, bias)
    first = np.zeros(count)
    first[0] = 1.0
    starts = [first, *(np.full(count, total / count) for total in (1.0, 0.25, 0.5, 2.0, 4.0, count))]
    seen = {tuple(weights)}
    for candidate in starts:
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        trial = _native_sum(columns, candidate)
        value = score(trial, bias)
        if value > current:
            weights, slope, current = candidate, trial, value

    def poll(weight_direction, slope_direction, bias_direction, distances):
        """Evaluate both offsets from the same point, accepting strict improvement."""
        nonlocal weights, slope, bias, current
        best = current
        selected = None
        for distance in distances:
            trial = slope if slope_direction is None else slope + distance * slope_direction
            trial_bias = bias + distance * bias_direction
            value = score(trial, trial_bias)
            if value > best:
                best = value
                selected = (distance, trial, trial_bias)
        if selected is not None:
            distance, slope, bias = selected
            if weight_direction is not None:
                weights = weights + distance * weight_direction
            current = best
        return selected is not None

    champion = (current, weights, bias)
    sweep = 0
    axis = np.zeros(count)
    for step in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
        for _ in range(3 if count <= 16 else 1):
            changed = False
            order = range(count) if sweep % 2 == 0 else reversed(range(count))
            sweep += 1
            for i in order:
                axis[i] = 1.0
                changed |= poll(axis, columns[i], 0.0, (step, -step))
                axis[i] = 0.0
            if not multiclass:
                changed |= poll(None, None, 1.0, (step, -step))
            if np.any(weights):
                changed |= poll(weights, slope, 0.0, (-0.5, 1.0))
            if not changed:
                break
            # Incremental updates can round differently from native replay,
            # particularly for float32 inputs and discontinuous objectives.
            # Compare finalists only after rebuilding in native slot order.
            slope = _native_sum(columns, weights)
            current = score(slope, bias)
            if current > champion[0]:
                champion = (current, weights, bias)
    value, weights, bias = champion
    return weights.tolist(), bias, {"evaluations": evaluations, "score": value}


def fit_pattern_tree(roots: Sequence[ValueNode], labels, *, objective: Callable, multiclass: bool = False):
    """Fit a native tree against a scalar, higher-is-better fit-set objective.

    Supply a nonempty ordered sequence of ValueNodes bound to fit probabilities,
    shaped (N, 1) for binary or (N, C) for multiclass with C >= 2. Each root
    may itself be a program. Labels are integer class indices of length N.
    The callback receives (native probability tensor, original labels).

    The first-root-only start gives additional bases zero initial weights.
    Other starts include zero evidence, the log-evidence sum and temperatures
    of the equal-weight pool. Coordinate polls fit signed slopes and, in binary,
    an intercept. Every objective invocation counts toward info["evaluations"].

    Returns (Tree, info), including the native-order winning fit score. Use
    identity binary or scale_vector_to_sum_1 multiclass postprocessing for
    subsequent native predictions, including after rebinding to fresh rows.
    """
    roots = tuple(roots)
    if not roots or not all(isinstance(root, ValueNode) for root in roots):
        raise ValueError("Supply a nonempty sequence of ValueNode roots")
    if not callable(objective):
        raise TypeError("objective must be callable")
    y = np.asarray(B.to_numpy(labels) if hasattr(labels, "detach") else labels)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or not y.size or y.dtype.kind not in "biuf" or not np.isfinite(y).all() or np.any(y != np.floor(y)):
        raise ValueError("labels must be a nonempty vector of integer class indices")

    copies = [root.copy_subtree() for root in roots]
    columns = []
    shape = None
    binary = BinaryAffineNode(None, [1.0])
    with B.no_grad(), use_postprocessing_function(scale_vector_to_sum_1 if multiclass else _passthrough):
        for root in copies:
            prediction = Tree(root).predict()
            dims = tuple(B.shape(prediction))
            if len(dims) != 2 or dims[0] != len(y) or dims[1] < 1 or (dims[1] > 1) != multiclass:
                raise ValueError("Each root must return (N, 1) binary or (N, C) multiclass probabilities")
            if shape is None:
                shape = dims
                classes = dims[1] if multiclass else 2
                if np.any((y < 0) | (y >= classes)):
                    raise ValueError("labels contain an out-of-range class index")
            elif dims != shape:
                raise ValueError("All roots must have matching probability shapes")
            # Validate on the native device; only a scalar crosses back to CPU.
            if not bool(((prediction >= 0) & (prediction <= 1)).all()):
                raise ValueError("Each root must return finite probabilities in [0, 1]")
            if multiclass:
                eps = WeightedLogitMeanNode._EPS
                columns.append(B.log(B.clip(prediction, eps, 1.0 - eps)))
            else:
                columns.append(binary._feature(prediction))
        weights, bias, info = _search(columns, labels, objective, multiclass)

    parent, *children = copies
    head = WeightedLogitMeanNode(children, weights) if multiclass else BinaryAffineNode(children, weights, bias)
    parent.add_child(head)
    return Tree(parent), info
