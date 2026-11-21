"""Pointwise bounds for noiseless, nonconstant monotone binary rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _binary_matrix(bits):
    array = np.asarray(bits)
    if array.ndim != 2 or array.shape[1] < 1 or array.dtype.kind not in "biuf" or not np.isin(array, (0, 1)).all():
        raise ValueError("bits must be a binary matrix with at least one column")
    return array


def _masks(bits):
    # Python integers retain all bits, including indices above 63.
    packed = np.packbits(bits.astype(bool, copy=False), axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), byteorder="little") for row in packed]


def _minimal_sets(masks):
    kept = []
    for mask in sorted(set(masks), key=lambda key: (key.bit_count(), key)):
        if not any((other & mask) == other for other in kept):
            kept.append(mask)
    return tuple(kept)


@dataclass(frozen=True)
class MonotoneEnvelope:
    """Positive and negative antichains encoded as unbounded integer masks."""

    n_inputs: int
    minimal_positive: tuple[int, ...]
    maximal_negative: tuple[int, ...]

    def bounds(self, query_bits):
        """Return (lower, upper), the forced and possible positives for each row."""
        query = _binary_matrix(query_bits)
        if query.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} input columns")
        masks = _masks(query)
        bounds = {
            mask: (
                any((positive & mask) == positive for positive in self.minimal_positive),
                not any((negative & mask) == mask for negative in self.maximal_negative),
            )
            for mask in set(masks)
        }
        values = np.asarray([bounds[mask] for mask in masks], dtype=bool).reshape(-1, 2)
        return values[:, 0], values[:, 1]

    def predict_positive(self, query_bits):
        """Return 0 or 1 where all compatible rules agree, and 0.5 elsewhere."""
        lower, upper = self.bounds(query_bits)
        return (lower.astype(float) + upper) / 2


def fit_monotone_envelope(bits, y) -> MonotoneEnvelope:
    """Fit exact lower and upper label bounds without enumerating the hypercube.

    A positive observation forces all supersets positive. A negative observation
    forces all subsets negative. The two closure boundaries are both compatible
    full rules when no positive is contained in any negative. Nonconstancy fixes
    the empty set's label to zero and the full set's label to one, including
    when no rows have been observed. Any point between the boundaries can take
    either label in at least one compatible rule.
    """
    bits = _binary_matrix(bits)
    labels = np.asarray(y)
    if labels.shape != (len(bits),) or labels.dtype.kind not in "biuf" or not np.isin(labels, (0, 1)).all():
        raise ValueError("y must contain one binary label per row")
    n_inputs = bits.shape[1]
    full = (1 << n_inputs) - 1
    positives, negative_complements = {full}, {full}
    for mask, label in zip(_masks(bits), labels, strict=True):
        if label:
            positives.add(mask)
        else:
            negative_complements.add(full ^ mask)
    minimal_positive = _minimal_sets(positives)
    maximal_negative = tuple(full ^ mask for mask in _minimal_sets(negative_complements))
    if any((positive & negative) == positive for positive in minimal_positive for negative in maximal_negative):
        raise ValueError("contradictory monotone observations: a positive input is contained in a negative input")
    return MonotoneEnvelope(n_inputs, minimal_positive, maximal_negative)
