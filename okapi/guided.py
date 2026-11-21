"""Build bounded fusion-tree libraries from fit-time provider correlations."""

from collections.abc import Mapping
from itertools import combinations

import numpy as np

from okapi.node import ValueNode
from okapi.operators import MAX, MEAN, MIN
from okapi.tree import Tree


def _correlation_distances(columns):
    # Subtracting a reference before centering keeps constant columns exactly
    # zero. Rescaling before taking norms also handles tiny score differences.
    columns = np.asarray(columns, dtype=np.float64)
    centered = columns - columns[:1]
    scale = np.max(np.abs(centered), axis=0)
    np.divide(centered, scale, out=centered, where=scale > 0)
    centered -= centered.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    np.divide(centered, norm, out=centered, where=norm > 0)
    return 1 - np.clip(centered.T @ centered, -1, 1)


def _average_link_cuts(distances):
    """Cut an ordered pairwise dissimilarity matrix at one through four groups."""
    distances = np.array(distances, dtype=np.float64, copy=True)
    groups = [(i,) for i in range(len(distances))]
    cuts = {}
    while len(groups) > 1:
        if len(groups) <= 4:
            cuts[len(groups)] = tuple(groups)
        left, right = np.triu_indices(len(groups), k=1)
        closest = np.argmin(distances[left, right])
        i, j = left[closest], right[closest]
        size_i, size_j = len(groups[i]), len(groups[j])
        # Average linkage weights the two previous means by their sizes.
        merged = (size_i * distances[i] + size_j * distances[j]) / (size_i + size_j)
        distances[i, :] = distances[:, i] = merged
        distances = np.delete(np.delete(distances, j, axis=0), j, axis=1)
        groups[i] = tuple(sorted((*groups[i], *groups[j])))
        del groups[j]
    cuts[1] = tuple(groups)
    return cuts


def _binary_compositions(roots):
    if len(roots) == 1:
        yield roots[0]
        return
    # The first group always belongs to the left subtree. This enumerates
    # each unordered bipartition once for both commutative operators.
    rest = range(1, len(roots))
    for count in range(len(roots) - 1):
        for chosen in combinations(rest, count):
            left_roots = [roots[i] for i in (0, *chosen)]
            right_roots = [roots[i] for i in rest if i not in chosen]
            for left in _binary_compositions(left_roots):
                for right in _binary_compositions(right_roots):
                    for operator in (MIN, MAX):
                        root = left.copy_subtree()
                        root.add_child(operator([right.copy_subtree()]))
                        yield root


def _templates_from_cuts(ids, values, cuts, shuffle_seed=None):
    """Build the same min/max library for prespecified ordered provider groups."""
    order = np.arange(len(ids)) if shuffle_seed is None else np.random.default_rng(shuffle_seed).permutation(len(ids))
    trees = []
    for count in range(1, 5):
        roots = []
        for group in cuts[count]:
            leaves = [ValueNode(None, values[order[i]], ids[order[i]]) for i in group]
            root = leaves[0]
            if len(leaves) > 1:
                root.add_child(MEAN(leaves[1:]))
            roots.append(root)
        trees.extend(Tree(root) for root in _binary_compositions(roots))
    return trees


def correlation_templates(
    providers: Mapping[int | str, np.ndarray], *, shuffle_seed: int | None = None,
) -> list[Tree]:
    """Return 135 OKAPI trees using only binary fit predictions.

    Supply four or more row-aligned positive-class probability arrays, each
    shaped (N,) or (N, 1), with at least two rows. Fit values follow the first
    provider's layout. Supply a consistent shape when rebinding new predictions.
    Insertion order breaks linkage ties. Average linkage uses signed Pearson
    distance 1 - r and treats constant providers as uncorrelated.

    Cut the hierarchy at one through four groups. Average each group's scores,
    then enumerate every unordered binary min/max composition over groups.
    The four blocks contain 1, 2, 12 and 120 trees including equivalent
    functions. Every tree owns its nodes. A shuffle seed replaces provider
    assignments with one fixed permutation across all cuts after clustering.
    """
    if len(providers) < 4:
        raise ValueError("At least four providers are required")
    ids = list(providers)
    arrays = [np.asarray(providers[key]) for key in ids]
    shape = arrays[0].shape
    for array in arrays:
        if array.ndim not in (1, 2) or (array.ndim == 2 and array.shape[1] != 1):
            raise ValueError("Expected binary positive-class scores shaped (N,) or (N, 1)")
        if array.shape[0] < 2 or array.shape[0] != shape[0]:
            raise ValueError("All providers must have the same length with at least two rows")
        if array.dtype.kind not in "biuf" or not np.all((array >= 0) & (array <= 1)):
            raise ValueError("Provider scores must be finite probabilities in [0, 1]")
    dtype = np.result_type(np.float32, *(array.dtype for array in arrays))
    values = [(array if array.shape == shape else array.reshape(shape)).astype(dtype, copy=False) for array in arrays]
    cuts = _average_link_cuts(_correlation_distances(np.column_stack(values)))
    return _templates_from_cuts(ids, values, cuts, shuffle_seed)
