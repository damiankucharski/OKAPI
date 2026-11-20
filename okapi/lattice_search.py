"""Exhaustive empirical Brier selection of small monotone min/max programs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np

Term = tuple[int, ...]
Program = tuple[Term, ...]


@lru_cache(maxsize=4)
def _programs(n_inputs: int) -> tuple[Program, ...]:
    # A program is the max of its minimal sufficient conjunctions (their min).
    # A conjunction contained in another absorbs it, so enumerate antichains.
    terms = sorted(term for size in range(1, n_inputs + 1) for term in combinations(range(n_inputs), size))
    masks = {term: sum(1 << i for i in term) for term in terms}

    def extend(prefix, remaining):
        for index, term in enumerate(remaining):
            program = prefix + (term,)
            yield program
            mask = masks[term]
            incomparable = [other for other in remaining[index + 1 :] if mask & masks[other] not in (mask, masks[other])]
            yield from extend(program, incomparable)

    return tuple(sorted(extend((), terms), key=lambda p: (sum(map(len, p)), p)))


def _probabilities(values):
    array = np.asarray(values)
    if array.ndim != 2 or not 1 <= array.shape[1] <= 4:
        raise ValueError("probabilities must have shape (n, k) with 1 <= k <= 4")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("probabilities must be finite real numbers between 0 and 1")
    return array.astype(np.float64, copy=False)


@dataclass(frozen=True)
class SmallLattice:
    """A nonconstant monotone lattice function in max-of-mins normal form."""

    terms: Program
    n_inputs: int
    n_candidates: int

    def predict(self, probabilities):
        """Evaluate the selected program on positive-class probability columns."""
        scores = _probabilities(probabilities)
        if scores.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} probability columns")
        return np.maximum.reduce([scores[:, term].min(axis=1) for term in self.terms])


def fit_small_lattice(probabilities, y) -> SmallLattice:
    """Choose a min/max program on 1–4 binary probability columns by empirical Brier loss.

    The candidates are all nonconstant monotone Boolean functions lifted to probabilities
    by replacing OR with max and AND with min. Exact loss ties favor fewer literals, then
    lexicographic order of the sorted zero-based index tuples. Distinct input columns are
    treated as separate variables even if all their observed values coincide.
    """
    scores = _probabilities(probabilities)
    target = np.asarray(y)
    if len(scores) == 0 or target.shape != (len(scores),) or target.dtype.kind not in "biuf" or not np.isin(target, (0, 1)).all():
        raise ValueError("y must contain one binary label per row and at least one row")
    candidates = _programs(scores.shape[1])
    minima = {term: scores[:, term].min(axis=1) for term in set().union(*candidates)}

    def brier(terms):
        prediction = np.maximum.reduce([minima[term] for term in terms])
        return np.mean(np.square(prediction - target))

    return SmallLattice(min(candidates, key=brier), scores.shape[1], len(candidates))
