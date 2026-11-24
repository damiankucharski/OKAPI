"""Native pairwise minima of raw discovery-set provider probabilities."""

from collections.abc import Callable, Sequence
from itertools import combinations
from math import isfinite
from numbers import Real

from okapi.functions import scale_vector_to_sum_1
from okapi.globals import BACKEND as B
from okapi.globals import _passthrough, use_postprocessing_function
from okapi.node import MinNode, ValueNode
from okapi.tree import Tree


def build_min_pair_basis(roots: Sequence[ValueNode], labels, *, objective: Callable, multiclass: bool = False):
    """Return (tree, native prediction) pairs and accounting on discovery rows.

    Supply ordered raw ValueNodes bound to matrices of shape (N, 1) for
    binary or (N, C) for multiclass. With more than twelve providers, retain
    the twelve highest individual objective(prediction, labels) scores. Ties
    favor earlier providers. Each pair (i, j) is emitted in original index
    order. Both the returned program and its discovery prediction use native
    MIN. Replay with identity binary or native sum-normalized multiclass
    postprocessing after rebinding provider IDs to new rows.

    info["evaluations"] counts individual objective calls and info["programs"]
    counts pair predictions. info["pairs"] aligns with archive positions and
    uses indices into the original provider sequence.
    """
    roots = tuple(roots)
    if not roots or not all(isinstance(root, ValueNode) and not root.children and root.value is not None for root in roots):
        raise ValueError("Supply a nonempty sequence of bound raw provider ValueNodes")
    if not callable(objective):
        raise TypeError("objective must be callable")
    shape = tuple(B.shape(roots[0].value))
    if len(shape) != 2 or shape[0] == 0 or shape[0] != len(labels) or shape[1] < 1 or (shape[1] > 1) != multiclass:
        raise ValueError("Discovery rows must align with labels and have shape (N, 1) for binary or (N, C) for multiclass")
    for root in roots:
        p = root.value
        if tuple(B.shape(p)) != shape or not bool(((p >= 0) & (p <= 1)).all()):
            raise ValueError("All discovery providers must have matching shapes and finite probabilities in [0, 1]")
        if multiclass and not bool((abs(B.sum(p, axis=-1) - 1) <= 1e-6).all()):
            raise ValueError("Multiclass provider probabilities must sum to one on each discovery row")

    indices = list(range(len(roots)))
    evaluations = 0
    archive = []
    pairs = []
    with B.no_grad(), use_postprocessing_function(scale_vector_to_sum_1 if multiclass else _passthrough):
        if len(roots) > 12:
            scores = []
            for root in roots:
                value = objective(B.clone(root.value), labels)
                if getattr(value, "ndim", None) == 0:
                    value = value.item()
                if not isinstance(value, Real) or not isfinite(value):
                    raise ValueError("objective must return a finite real scalar")
                scores.append(float(value))
            evaluations = len(scores)
            indices = sorted(sorted(indices, key=scores.__getitem__, reverse=True)[:12])

        # Each candidate has independent nodes. Isolate the bound data from
        # the callers' tensors once per retained provider, then reuse that data.
        operands = {}
        for i in indices:
            operand = roots[i].copy_subtree()
            operand.value = B.clone(operand.value)
            operands[i] = operand
        for i, j in combinations(indices, 2):
            parent = operands[i].copy_subtree()
            parent.add_child(MinNode([operands[j].copy_subtree()]))
            tree = Tree(parent)
            archive.append((tree, tree.predict()))
            pairs.append([i, j])
    return archive, {"provider_indices": indices, "pairs": pairs, "evaluations": evaluations, "programs": len(archive)}
