"""Exact Hamming-loss minimizers among nonconstant monotone Boolean functions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from okapi.monotone_envelope import _binary_matrix


def _indices(bits):
    return bits.astype(np.intp, copy=False) @ (1 << np.arange(bits.shape[1], dtype=np.intp))


@lru_cache(maxsize=12)
def _cover_edges(dimension):
    """The Hasse diagram adds exactly one previously absent coordinate."""
    flags = 1 << np.arange(dimension)
    masks = np.arange(1 << dimension)
    start, bit = np.nonzero((masks[:, None] & flags) == 0)
    return start, start | flags[bit]


@dataclass(frozen=True)
class MonotoneERM:
    """The least and greatest empirical minimizers, indexed by Boolean masks."""

    n_inputs: int
    min_errors: int
    lower_by_mask: np.ndarray
    upper_by_mask: np.ndarray

    def bounds(self, query_bits):
        """Return Boolean predictions of the least and greatest optimal rules."""
        query = _binary_matrix(query_bits)
        if query.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} input columns")
        indices = _indices(query)
        return self.lower_by_mask[indices], self.upper_by_mask[indices]

    def predict_positive(self, query_bits):
        """Return the midpoint of the two optimal predictions on each query."""
        lower, upper = self.bounds(query_bits)
        return (lower.astype(float) + upper) / 2


def fit_monotone_erm(bits, y) -> MonotoneERM:
    """Fit every Hamming-optimal rule through the extremal cuts of one flow.

    Repeated binary observations contribute separately, including conflicting
    labels. The empty set must predict zero and the full set one. Positive
    Boolean cells form the source side of a cut. Its finite cost counts
    disagreed labels, and infinite upward cover edges enforce monotonicity.
    The union and intersection of optimal source sides are also optimal. They
    bound all minimizers, without implying independent choices per cell.

    The full hypercube has 2**d vertices and d*2**(d-1) cover edges; this API
    accepts 1 through 12 columns.
    """
    bits = _binary_matrix(bits)
    dimension = bits.shape[1]
    if dimension > 12:
        raise ValueError("exact monotone ERM requires between 1 and 12 binary input columns")
    labels = np.asarray(y)
    if labels.shape != (len(bits),) or labels.dtype.kind not in "biuf" or not np.isin(labels, (0, 1)).all():
        raise ValueError("y must contain one binary label per row")

    indices = _indices(bits)
    cells = 1 << dimension
    positives = np.bincount(indices[labels == 1], minlength=cells)
    negatives = np.bincount(indices[labels == 0], minlength=cells)
    # Cancel contradictory repeats at each cell. Their paired contribution is
    # fixed for every rule, as are erroneous labels at the two anchored cells.
    fixed = int(positives[0] + negatives[-1])
    balance = positives - negatives
    fixed += int(np.minimum(positives[1:-1], negatives[1:-1]).sum())
    balance[[0, -1]] = 0
    # Use the cheaper of the two feasible constant-interior cuts as a bound.
    # Hard-edge capacity strictly above it suffices regardless of fixed errors.
    limit = min(balance[balance > 0].sum(), -balance[balance < 0].sum()) + 1
    if limit > np.iinfo(np.int32).max:
        raise ValueError("minimum-cut capacity exceeds SciPy's signed 32-bit flow limit")
    positive = np.flatnonzero(balance > 0)
    negative = np.flatnonzero(balance < 0)
    source, sink = cells, cells + 1
    cover_from, cover_to = _cover_edges(dimension)
    rows = np.r_[cover_from, np.full(len(positive), source), negative, source, 0]
    columns = np.r_[cover_to, positive, np.full(len(negative), sink), cells - 1, sink]
    # Unary costs above limit cannot occur on any minimum cut, so clipping
    # them is exact and keeps individual capacities within the backend limit.
    capacities = np.r_[np.full(len(cover_from), limit), np.minimum(limit, balance[positive]), np.minimum(limit, -balance[negative]), limit, limit]
    graph = csr_matrix((capacities, (rows, columns)), shape=(cells + 2, cells + 2))
    result = maximum_flow(graph, source, sink)
    residual = graph - result.flow
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    to_sink = breadth_first_order(residual.T.tocsr(), sink, directed=True, return_predecessors=False)
    lower, upper = np.zeros(cells + 2, dtype=bool), np.ones(cells + 2, dtype=bool)
    lower[reachable] = True
    upper[to_sink] = False
    return MonotoneERM(dimension, fixed + int(result.flow_value), lower[:cells], upper[:cells])
