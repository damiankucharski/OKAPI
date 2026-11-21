"""Compile equal-weight binary logistic pairs into midpoint decision rules."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logsumexp


def midpoint_certificate(midpoints):
    """Return positive-class predictions and a mask for pairwise proven decisions."""
    centers = np.asarray(midpoints, dtype=float)
    if centers.ndim == 0 or centers.shape[-1] == 0:
        raise ValueError("At least one pair is required")
    positive = np.all(centers >= 0, axis=-1)
    return positive, positive | np.all(centers <= 0, axis=-1)


def _fallback(centers, differences):
    # Six ordinary sigmoid evaluations suffice for three uncertain pairs. Resolve
    # machine-level cancellation via the signed identity for the pair's margin.
    count = centers.shape[1]
    summed = expit(centers + differences).sum(1) + expit(centers - differences).sum(1)
    margin = summed - count
    result = margin >= 0
    ambiguous = np.abs(margin) <= 16 * np.finfo(float).eps * count
    if ambiguous.any():
        m, d = centers[ambiguous], differences[ambiguous]
        # Symmetric logits can give an exact tie across the original pairs even
        # when independently rounded log-sums disagree. Re-pair extremes first.
        logits = np.sort(np.concatenate([m + d, m - d], axis=1), axis=1)
        n = m.shape[1]
        extremes, proven = midpoint_certificate(logits[:, :n] / 2 + logits[:, ::-1][:, :n] / 2)
        pending = np.flatnonzero(ambiguous)
        result[pending[proven]] = extremes[proven]
        if proven.all():
            return result
        m, d = m[~proven], d[~proven]
        a, b = np.abs(m), np.abs(d)
        scale = np.maximum(a, b)
        with np.errstate(divide="ignore", over="ignore"):
            numerator = a - scale + np.log(-np.expm1(-2 * a))
            denominator = np.logaddexp(
                a - scale + np.log1p(np.exp(-2 * a)), b - scale + np.log1p(np.exp(-2 * b))
            )
        _, sign = logsumexp(numerator - denominator, b=np.sign(m), axis=1, return_sign=True)
        result[pending[~proven]] = sign >= 0
    return result


@dataclass
class BinaryPairRouter:
    center_weights: np.ndarray
    center_bias: np.ndarray
    difference_weights: np.ndarray
    difference_bias: np.ndarray

    def predict(self, x, *, return_certificate=False):
        """Classify raw inputs with the full equal-weight rule; sigmoid ties go to class one."""
        x = np.asarray(x, dtype=float)
        if x.ndim < 1 or x.shape[-1] != self.center_weights.shape[1] or not np.isfinite(x).all():
            raise ValueError("Inputs need the compiled raw feature dimension")
        rows = x.reshape(-1, x.shape[-1])
        centers = rows @ self.center_weights.T + self.center_bias
        labels, cert = midpoint_certificate(centers)
        uncertain = ~cert
        if uncertain.any():
            delta = rows[uncertain] @ self.difference_weights.T + self.difference_bias
            labels[uncertain] = _fallback(centers[uncertain], delta)
        labels = labels.reshape(x.shape[:-1])
        return (labels, cert.reshape(x.shape[:-1])) if return_certificate else labels


def compile_binary_pairs(weights, intercepts, pairs):
    """Compile an even, equally weighted ensemble of raw-input binary affine logits."""
    w, b = np.asarray(weights, dtype=float), np.asarray(intercepts, dtype=float)
    edges = np.asarray(pairs)
    if (
        w.ndim != 2
        or not w.size
        or b.shape != (len(w),)
        or not np.isfinite(w).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Use K finite logits with K by D weights and K biases")
    if (
        edges.shape != (len(w) // 2, 2)
        or edges.dtype.kind not in "iu"
        or not np.array_equal(np.sort(edges.ravel()), np.arange(len(w)))
    ):
        raise ValueError("Pairs must partition the model indices into two-element groups")
    left, right = edges.T
    return BinaryPairRouter(
        w[left] / 2 + w[right] / 2,
        b[left] / 2 + b[right] / 2,
        w[left] / 2 - w[right] / 2,
        b[left] / 2 - b[right] / 2,
    )
