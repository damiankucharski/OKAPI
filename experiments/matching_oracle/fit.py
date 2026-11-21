"""Exact empirical Brier fit for hidden pairs on rows with two high providers."""

from dataclasses import dataclass
from functools import cache

import numpy as np

from okapi.node import MaxNode, MinNode, ValueNode
from okapi.tree import Tree


@dataclass(frozen=True)
class MatchingFit:
    pairs: tuple[tuple[int, int], ...]
    fit_brier: float
    n_optimal: int
    n_states: int
    transitions: int


def fit_matching(probabilities, labels, *, low=1 / 64, high=63 / 64):
    """Minimize positive-channel Brier over disjoint pairs covering all providers.

    Each row has exactly two high entries. A matching predicts high on that
    row iff it contains the corresponding pair. The endpoints must sum to one.
    Break exact integer-score ties by lexicographic order of the sorted pairs.
    """
    x = np.asarray(probabilities)
    y = np.asarray(labels)
    if not (np.isscalar(low) and np.isscalar(high) and 0 <= low < high <= 1 and low + high == 1):
        raise ValueError("low and high must be complementary probability endpoints")
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2 or x.shape[1] % 2:
        raise ValueError("expected nonempty rows and an even number of at least two providers")
    if x.dtype.kind not in "biuf" or not np.all((x == low) | (x == high)):
        raise ValueError("every prediction must equal the low or high endpoint")
    active = x == high
    if not np.all(active.sum(axis=1) == 2):
        raise ValueError("each training row must have exactly two high providers")
    if y.shape != (len(x),) or y.dtype.kind not in "biuf" or not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be a binary vector aligned with the rows")
    y = y.astype(np.int64, copy=False)
    n_inputs = x.shape[1]
    left, right = np.nonzero(active)[1].reshape(-1, 2).T
    benefit = np.zeros((n_inputs, n_inputs), dtype=np.int64)
    np.add.at(benefit, (left, right), 2 * y - 1)
    transitions = 0

    @cache
    def solve(mask):
        nonlocal transitions
        if not mask:
            return 0, 1, ()
        first_bit = mask & -mask
        first = first_bit.bit_length() - 1
        remaining = mask ^ first_bit
        available = remaining
        best_score, n_best, best_pairs = None, 0, None
        while available:
            next_bit = available & -available
            available ^= next_bit
            second = next_bit.bit_length() - 1
            tail_score, tail_count, tail_pairs = solve(remaining ^ next_bit)
            transitions += 1
            score = int(benefit[first, second]) + tail_score
            if best_score is None or score > best_score:
                best_score, n_best = score, tail_count
                best_pairs = ((first, second), *tail_pairs)
            elif score == best_score:
                n_best += tail_count
        return best_score, n_best, best_pairs

    score, n_optimal, pairs = solve((1 << n_inputs) - 1)
    squared_error = np.sum((y - low) ** 2, dtype=float) - (high - low) * score
    return MatchingFit(pairs, float(squared_error / len(y)), n_optimal, solve.cache_info().currsize, transitions)


def matching_tree(pairs, tensors):
    """Compile a full matching to ordinary native MIN and MAX updates.

    Provider IDs are the zero-based positions in tensors. Each provider occurs
    exactly once. The caller supplies arrays from the desired evaluation split.
    """
    pairs = tuple(pairs)
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("expected nonempty pairs of provider indices")
    indices = [i for edge in pairs for i in edge]
    if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in indices)
            or sorted(indices) != list(range(len(tensors)))):
        raise ValueError("pairs must cover every provider index exactly once")
    branches = [
        ValueNode([MinNode([ValueNode(None, tensors[j], j)])], tensors[i], i)
        for i, j in pairs
    ]
    root, *rest = branches
    if rest:
        root.add_child(MaxNode(rest))
    return Tree(root)
