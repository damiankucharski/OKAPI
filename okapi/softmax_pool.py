"""Compile an arithmetic pool of linear softmaxes into one affine softmax with TV certificates."""

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax

CURVATURE_FACTOR = 1 / (12 * np.sqrt(3))


@dataclass
class CompiledSoftmaxPool:
    weights: np.ndarray
    intercepts: np.ndarray
    bound_matrix: np.ndarray
    mixing: np.ndarray
    centered_residuals: np.ndarray | None

    def predict(self, x):
        return softmax(np.asarray(x) @ self.weights.T + self.intercepts, axis=-1)

    def error_bound(self, x):
        """Pointwise total-variation bound using only a raw-input quadratic form."""
        x = np.asarray(x, dtype=float)
        q = np.einsum("...d,de,...e->...", x, self.bound_matrix[:-1, :-1], x)
        return np.clip(q + 2 * (x @ self.bound_matrix[:-1, -1]) + self.bound_matrix[-1, -1], 0, 1)

    def oscillation_bound(self, x):
        """Tighter pointwise TV bound that evaluates all retained residual affine logits."""
        if self.centered_residuals is None:
            raise ValueError("Compile with keep_sources=True for the per-source oscillation bound")
        x = np.asarray(x, dtype=float)
        residual = self.centered_residuals
        z = np.einsum("...d,kcd->...kc", x, residual[:, :, :-1]) + residual[:, :, -1]
        return np.minimum(1, CURVATURE_FACTOR * (np.ptp(z, axis=-1) ** 2 @ self.mixing))


def compile_softmax_pool(weights, intercepts, mixing_weights, *, keep_sources=True):
    """Compile K positive-weight affine softmax experts with C classes and D shared raw features.

    Inputs have shapes (K,C,D), (K,C) and (K,). Common within-expert class logit shifts
    cancel before computing the bounds. The resulting classifier predicts using C affine logits.
    Set keep_sources=False to discard source parameters when only the quadratic bound is needed.
    """
    w, b, a = [np.asarray(v, dtype=float) for v in (weights, intercepts, mixing_weights)]
    if w.ndim != 3 or not w.size or w.shape[1] < 2 or b.shape != w.shape[:2] or a.shape != (len(w),):
        raise ValueError("Supply weights (K,C,D), biases (K,C) and mixing weights (K,), with C>=2")
    if not all(np.isfinite(v).all() for v in (w, b, a)) or (a < 0).any() or a.max() <= 0:
        raise ValueError("Use finite parameters and nonnegative mixing weights with positive total mass")
    a = a / a.max()
    a /= a.sum()
    active = a > 0
    a = a[active]
    theta = np.concatenate((w[active], b[active, :, None]), axis=2)
    theta -= theta.mean(axis=1, keepdims=True)
    mean = np.einsum("k,kcd->cd", a, theta)
    residual = theta - mean
    matrix = 2 * CURVATURE_FACTOR * np.einsum("k,kci,kcj->ij", a, residual, residual, optimize=True)
    return CompiledSoftmaxPool(
        mean[:, :-1].copy(), mean[:, -1].copy(), matrix, a, residual if keep_sources else None
    )
