"""Choose a native candidate across environments and shrink fitted subtree weights."""

from collections.abc import Mapping, Sequence
from typing import TypeVar

import numpy as np

from okapi.globals import BACKEND as B
from okapi.interventions import copy_for_variation
from okapi.node import BinaryAffineNode, WeightedLogitMeanNode
from okapi.tree import Tree

Candidate = TypeVar("Candidate")


def _finite_scalar(value, name):
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype.kind not in "iuf" or not np.isfinite(arr):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(arr)


def select_transfer_candidate(scores: Mapping[Candidate, Sequence[float]], *, clean_tolerance: float = 0.01) -> Candidate:
    """Return a candidate key using higher-is-better scores in metric units.

    Give every candidate the same ordered environments, clean first followed
    by one or more stresses. Candidates within the inclusive tolerance of the
    best clean score compete on their minimum score across all environments.
    Ties favor higher clean score, then earlier mapping order.
    """
    tolerance = _finite_scalar(clean_tolerance, "clean_tolerance")
    if tolerance < 0:
        raise ValueError("clean_tolerance must be nonnegative")
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("scores must be a nonempty mapping")

    ranks = {}
    expected = None
    for candidate, row in scores.items():
        values = np.asarray(row)
        if values.ndim != 1 or values.size < 2 or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("Each candidate requires a finite clean score and at least one finite stress score")
        if expected is None:
            expected = len(values)
        elif len(values) != expected:
            raise ValueError("All candidates must have scores for the same environments")
        ranks[candidate] = (float(values.min()), float(values[0]))
    threshold = max(clean for _, clean in ranks.values()) - tolerance
    eligible = (candidate for candidate, (_, clean) in ranks.items() if clean >= threshold)
    return max(eligible, key=ranks.__getitem__)


def shrink_subtree_weights(tree: Tree, provider_count: int, factor: float = 0.5) -> Tree:
    """Return a private tree with scaled appended features in its final logit head.

    The original providers must occupy the first provider_count slots, including
    the incoming parent in slot zero. Appended subtrees follow in head order.
    All earlier updates, nested parameters and any binary intercept are retained.
    The result has fresh evaluation caches and independent bound tensors.
    """
    factor = _finite_scalar(factor, "factor")
    if not 0 <= factor <= 1:
        raise ValueError("factor must be in [0, 1]")
    if not isinstance(tree, Tree) or not tree.root.children:
        raise ValueError("tree must have a fitted logit head as its last root update")
    head = tree.root.children[-1]
    if not isinstance(head, (BinaryAffineNode, WeightedLogitMeanNode)):
        raise ValueError("tree must have a binary affine or weighted logit head as its last root update")
    if isinstance(provider_count, (bool, np.bool_)) or not isinstance(provider_count, (int, np.integer)):
        raise ValueError("provider_count must be an integer")
    if not 1 <= provider_count <= len(head.children) + 1:
        raise ValueError("provider_count must include the root anchor and fit within the head arity")

    candidate = copy_for_variation(tree)
    head = candidate.root.children[-1]
    weights = head.coefficients if isinstance(head, BinaryAffineNode) else head._weights
    if len(weights) != len(head.children) + 1:
        raise ValueError("The fitted head must have a weight for each input slot")
    weights[provider_count:] = [factor * w for w in weights[provider_count:]]
    # Native node copies share bound tensors. Isolate each distinct tensor
    # once, retaining repeated references within the copied tree.
    tensors = {}
    for node in candidate.value_nodes:
        if node.value is not None:
            identity = id(node.value)
            if identity not in tensors:
                tensors[identity] = B.clone(node.value)
            node.value = tensors[identity]
    return candidate
