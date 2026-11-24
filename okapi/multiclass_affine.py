"""Fit a native softmax of affine functions of multiclass probability vectors."""

from collections.abc import Callable, Sequence
from numbers import Integral, Real

import numpy as np

from okapi.functions import scale_vector_to_sum_1
from okapi.globals import BACKEND as B
from okapi.globals import use_postprocessing_function
from okapi.node import MulticlassAffineNode, ValueNode
from okapi.tree import Tree


def _fit_inputs(roots, labels):
    if not roots or not all(isinstance(root, ValueNode) for root in roots):
        raise ValueError("Provide at least one ValueNode root")
    y = np.asarray(B.to_numpy(labels) if hasattr(labels, "detach") else labels)
    if y.ndim != 1 or len(y) == 0 or y.dtype.kind not in "iu":
        raise ValueError("labels must be a nonempty integer vector")
    values = []
    reference = None
    for root in roots:
        # Copies start with empty evaluation caches, including nested subtrees.
        prediction = Tree(root.copy_subtree()).predict()
        column = np.asarray(B.to_numpy(prediction), dtype=np.float64)
        if column.ndim != 2 or column.shape[0] != len(y) or column.shape[1] < 2:
            raise ValueError("Each root must produce a multiclass probability matrix (N, C)")
        if values and column.shape != values[0].shape:
            raise ValueError("All roots must have identical prediction shapes")
        if not np.all((column >= 0) & (column <= 1)) or not np.allclose(column.sum(axis=1), 1.0, atol=1e-5, rtol=0):
            raise ValueError("Each root must produce finite probabilities summing to one per row")
        if reference is None:
            reference = prediction
        values.append(column)
    if np.any((y < 0) | (y >= values[0].shape[1])):
        raise ValueError("labels include an out-of-range class")
    return np.stack(values, axis=1), reference


def _softmax(logits):
    mass = np.exp(logits - logits.max(axis=1, keepdims=True))
    return mass / mass.sum(axis=1, keepdims=True)


def _tree(roots, weights, bias, mode):
    first, *others = (root.copy_subtree() for root in roots)
    first.add_child(MulticlassAffineNode(others, weights, bias, mode=mode))
    return Tree(first)


def _axes(providers, classes, mode):
    """Visit intercepts, matching-class slopes, then successive cyclic class offsets."""
    for out in range(classes):
        yield None, out
    offsets = range(classes) if mode == "full" else (0,)
    for offset in offsets:
        for provider in range(providers):
            for input_class in range(classes):
                index = (provider, input_class)
                out = (input_class + offset) % classes
                yield index if mode == "diagonal" else (*index, out), out


def fit_multiclass_affine_tree(
    roots: Sequence[ValueNode],
    labels,
    *,
    objective: Callable,
    mode: str = "diagonal",
    l2_strength: float = 0.0,
    max_evaluations: int = 2400,
):
    """Fit and return (Tree, info) using a higher-is-better scalar callback.

    Each ordered root supplies probabilities shaped (N, C), C >= 2. A root can
    be a whole native subtree. For mode='diagonal', score c uses input c from
    each provider. For mode='full', every input class may score every output.
    The deployed node has signed raw-probability coefficients and a class bias.

    Search starts at uniform output and tests shared slopes. It polls both signs
    of each intercept and individual slope at decreasing, fit-standardized
    step sizes. Fit-only centering of a slope direction is absorbed into the
    deployed bias. Finalists are replayed with the native Tree. Every call to
    objective(native_probabilities, original_labels) counts against the cap.
    Only the original uniform tree is evaluated when max_evaluations=1.
    The callback must be deterministic. Search uses CPU float64 and supplies
    callback predictions on the input backend and device.

    The penalty is l2_strength / N times sum(variance_mi * W_mio**2), summed
    over stored slopes, using fit-row population variance. Bias is unpenalized.
    info reports raw score, penalty, penalized_score, evaluations, mode and cap.
    Subsequent subtree replay uses the ordinary multiclass postprocessor.
    """
    if mode not in ("diagonal", "full"):
        raise ValueError("mode must be 'diagonal' or 'full'")
    if not callable(objective):
        raise TypeError("objective must be callable")
    if isinstance(l2_strength, (bool, np.bool_)) or not isinstance(l2_strength, Real) or not np.isfinite(l2_strength) or l2_strength < 0:
        raise ValueError("l2_strength must be a finite nonnegative scalar")
    if isinstance(max_evaluations, (bool, np.bool_)) or not isinstance(max_evaluations, Integral) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")

    roots = tuple(roots)
    with B.no_grad(), use_postprocessing_function(scale_vector_to_sum_1):
        inputs, reference = _fit_inputs(roots, labels)
        rows, providers, classes = inputs.shape
        # Center differences from the first row so exactly constant columns
        # have zero variance even when their decimal value is inexact.
        centered = inputs - inputs[0]
        offset = centered.mean(axis=0)
        center = inputs[0] + offset
        centered -= offset
        # Scale before squaring to handle valid, tiny provider probabilities.
        peaks = np.max(np.abs(centered), axis=0)
        scales = peaks * np.sqrt(np.square(centered / np.where(peaks > 0, peaks, 1.0)).mean(axis=0))
        penalty_scales = scales if mode == "diagonal" else scales[:, :, None]
        penalty_rate = float(l2_strength) / rows
        weights = np.zeros((providers, classes) if mode == "diagonal" else (providers, classes, classes))
        bias = np.zeros(classes)
        evaluations = 0

        def penalty(w):
            return penalty_rate * float(np.sum(np.square(w * penalty_scales))) if penalty_rate else 0.0

        def score(prediction):
            nonlocal evaluations
            evaluations += 1
            value = objective(prediction, labels)
            if np.ndim(value) != 0 or not np.isfinite(float(value)):
                raise ValueError("objective must return a finite scalar")
            return float(value)

        initial_tree = _tree(roots, weights, bias, mode)
        initial = score(initial_tree.predict())
        best_tree, best_raw, best_penalty = initial_tree, initial, 0.0
        rank = initial
        logits = np.zeros((rows, classes))
        class_ids = np.arange(classes)

        def trial_rank(trial_logits, trial_penalty):
            probabilities = _softmax(trial_logits)
            # Match the final native multiclass postprocessor on the softmax.
            if not isinstance(reference, np.ndarray):
                probabilities = B.to_device(B.tensor(probabilities), reference)
            return score(scale_vector_to_sum_1(probabilities)) - trial_penalty

        # Reserve a true native replay, which also handles score ties or
        # discontinuities sensitive to differences in floating-point summation.
        search_cap = max_evaluations - 1
        mean = inputs.mean(axis=1)
        for total_slope in (1.0, 4.0, 16.0, 2.0, 8.0, 32.0):
            if evaluations >= search_cap:
                break
            trial = np.zeros_like(weights)
            if mode == "diagonal":
                trial.fill(total_slope / providers)
            else:
                trial[:, class_ids, class_ids] = total_slope / providers
            trial_logits = total_slope * mean
            value = trial_rank(trial_logits, penalty(trial))
            if value > rank:
                weights, logits, rank = trial, trial_logits, value

        axes = list(_axes(providers, classes, mode))
        # A full sweep touches at most 2*(C + M*C*C) scores. Use more sweeps
        # on small models while applying the same hard cap to both modes.
        for step in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            if evaluations >= search_cap:
                break
            for index, out in axes:
                if evaluations >= search_cap:
                    break
                if index is None:
                    direction, scale = 1.0, 1.0
                else:
                    source = index[:2]
                    scale = scales[source]
                    if scale == 0:
                        continue
                    direction = centered[:, source[0], source[1]] / scale
                previous = logits[:, out].copy()
                current_penalty = penalty(weights)
                best_rank, best_distance = rank, 0.0
                for distance in (step, -step):
                    if evaluations >= search_cap:
                        break
                    deduction = current_penalty
                    if index is not None:
                        with np.errstate(over="ignore"):
                            proposed = weights[index] + distance / scale
                        if not np.isfinite(proposed):
                            continue
                        deduction += penalty_rate * distance * (2 * (weights[index] * scale) + distance)
                    logits[:, out] = previous + distance * direction
                    value = trial_rank(logits, deduction)
                    if value > best_rank:
                        best_rank, best_distance = value, distance
                logits[:, out] = previous + best_distance * direction
                if best_distance:
                    rank = best_rank
                    if index is None:
                        bias[out] += best_distance
                    else:
                        delta = best_distance / scale
                        weights[index] += delta
                        bias[out] -= delta * center[source]

        if rank > initial:
            candidate = _tree(roots, weights, bias, mode)
            raw = score(candidate.predict())
            deduction = penalty(weights)
            if raw - deduction > initial:
                best_tree, best_raw, best_penalty = candidate, raw, deduction
    return best_tree, {
        "mode": mode,
        "max_evaluations": int(max_evaluations),
        "evaluations": evaluations,
        "score": best_raw,
        "penalty": best_penalty,
        "penalized_score": best_raw - best_penalty,
    }
