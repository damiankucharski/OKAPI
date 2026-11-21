"""Fast sufficient tests for agreement between a compact softmax and its arithmetic pool."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space


@dataclass
class SoftmaxRouter:
    fused_weights: np.ndarray
    fused_bias: np.ndarray
    n_classes: int
    center: np.ndarray
    scale: np.ndarray
    tail_eigenvalue: float
    all_agree_factor: float

    @property
    def rank(self):
        return len(self.fused_bias) - self.n_classes

    def _scores_and_energy(self, x):
        x = np.asarray(x, dtype=float)
        projected = x @ self.fused_weights.T + self.fused_bias
        projections = projected[..., self.n_classes :]
        energy = np.sum(projections**2, axis=-1)
        if self.tail_eigenvalue:
            energy = energy + self.tail_eigenvalue * (
                1 + np.sum(((x - self.center) / self.scale) ** 2, axis=-1)
            )
        return projected[..., : self.n_classes], energy

    def residual_energy_upper(self, x):
        """Upper bound the weighted sum of squared centered logit residuals."""
        return self._scores_and_energy(x)[1]

    def certified(self, x):
        """Return compact top classes, the TV-based mask and the all-agree mask.

        An unproven input requires evaluation of the original pool for an exact top class.
        These masks certify labels, not the complete vector of probabilities.
        """
        scores, energy = self._scores_and_energy(x)
        largest = np.partition(scores, -2, axis=-1)[..., -2:]
        gap = largest[..., 1] - largest[..., 0]
        denominator = np.exp(scores - largest[..., 1, None]).sum(axis=-1)
        probability_gap = -np.expm1(-gap) / denominator
        by_tv = probability_gap > energy / (3 * np.sqrt(3))
        unanimous = gap**2 > self.all_agree_factor * energy
        return np.argmax(scores, axis=-1), by_tv, unanimous


def compile_softmax_router(weights, intercepts, mixing_weights, *, center, scale, rank=None):
    """Compile a label router; lower ranks use a conservative spectral tail bound.

    Each of K affine experts has C classes and D features. Positive scale and center are
    D-vectors used to choose a coordinate system for bounding the omitted directions.
    Returned linear projections accept raw features.
    """
    w, b, a, center, scale = [
        np.asarray(v, dtype=float) for v in (weights, intercepts, mixing_weights, center, scale)
    ]
    if w.ndim != 3 or not w.size or w.shape[1] < 2 or b.shape != w.shape[:2] or a.shape != (len(w),):
        raise ValueError("Supply K by C by D weights, K by C biases and K mixing weights")
    if center.shape != (w.shape[-1],) or scale.shape != center.shape or (scale <= 0).any():
        raise ValueError("Feature means and strictly positive scales must match the feature dimension")
    if not all(np.isfinite(v).all() for v in (w, b, a, center, scale)) or (a < 0).any() or a.max() <= 0:
        raise ValueError("Use finite inputs and nonnegative weights with positive mass")
    if rank is not None and (
        not isinstance(rank, (int, np.integer)) or isinstance(rank, (bool, np.bool_)) or rank < 0
    ):
        raise ValueError("Rank must be None or a nonnegative integer")
    a = a / a.max()
    a /= a.sum()
    active = a > 0
    a, w, b = a[active], w[active], b[active]
    b = b - b.mean(axis=1, keepdims=True)
    w = w - w.mean(axis=1, keepdims=True)
    mean_w = np.einsum("k,kcd->cd", a, w)
    mean_b = a @ b
    d_w = w - mean_w
    raw = np.concatenate((d_w * scale, (b - mean_b + d_w @ center)[:, :, None]), axis=2)
    # Centering removes exactly one direction on each of the two categorical axes.
    provider_basis = null_space(np.sqrt(a)[None, :]).T
    class_basis = null_space(np.ones((1, w.shape[1]))).T
    factor = np.einsum("ik,k,kcd,jc->ijd", provider_basis, np.sqrt(a), raw, class_basis, optimize=True)
    flat = factor.reshape((len(provider_basis) * len(class_basis), len(center) + 1))
    _, singular, directions = np.linalg.svd(flat, full_matrices=False)
    retained = len(singular) if rank is None else min(rank, len(singular))
    tail = singular[retained] ** 2 if retained < len(singular) else 0.0
    coeff = np.sqrt(np.maximum(0, singular[:retained] ** 2 - tail))[:, None] * directions[:retained]
    projection_w = coeff[:, :-1] / scale
    projection_b = coeff[:, -1] - projection_w @ center
    fused_w = np.concatenate([mean_w, projection_w])
    fused_b = np.concatenate([mean_b, projection_b])
    return SoftmaxRouter(
        fused_w,
        fused_b,
        w.shape[1],
        center.copy(),
        scale.copy(),
        float(tail),
        float(2 * (1 - a.min()) / a.min()),
    )
