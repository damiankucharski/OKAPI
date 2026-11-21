"""Smallest weighted Brier ball enclosing a finite pool of prediction vectors."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class BrierCenter:
    """Primal prediction, worst-case risk and dual witness on the candidate rows."""

    prediction: np.ndarray
    upper: float
    lower: float
    mixture: np.ndarray


def _fit_mixture(points):
    """Dual simplex weights for points in Euclidean coordinates."""
    size = len(points)
    if size == 1:
        return np.ones(1)
    if size == 2:
        return np.array([0.5, 0.5])
    if points.shape[1] == 1:
        mixture = np.zeros(size)
        mixture[[points[:, 0].argmin(), points[:, 0].argmax()]] = 0.5
        return mixture

    norms = np.einsum("ij,ij->i", points, points)
    initial = np.full(size, 1.0 / size)

    def negative_dual(mixture):
        center = mixture @ points
        return center @ center - mixture @ norms, 2 * (points @ center) - norms

    result = minimize(
        negative_dual,
        initial,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * size,
        constraints={"type": "eq", "fun": lambda m: m.sum() - 1.0, "jac": lambda m: np.ones_like(m)},
        options={"ftol": 1e-14, "maxiter": 500},
    )
    mixture = np.maximum(result.x, 0) if np.isfinite(result.x).all() else initial
    mixture = mixture / mixture.sum() if mixture.sum() > 0 else initial

    # SLSQP stops on objective changes. Where the certificate still has a gap,
    # move mass from the nearest active point to the farthest point. The exact
    # line maximum follows from the quadratic form of the dual.
    for _ in range(max(100, 2 * size)):
        residual = points - mixture @ points
        distances = np.einsum("ij,ij->i", residual, residual)
        far = distances.argmax()
        if distances[far] - mixture @ distances <= 1e-9:
            break
        near = np.argmin(np.where(mixture > 0, distances, np.inf))
        direction = points[far] - points[near]
        curvature = 2 * (direction @ direction)
        if curvature == 0:
            break
        step = min(mixture[near], (distances[far] - distances[near]) / curvature)
        mixture[near] -= step
        mixture[far] += step
    return mixture


def fit_brier_center(candidates, target_weights=None):
    """Minimize the largest target-weighted MSE against a finite set of rows.

    Each row of candidates holds probabilities on the same ordered coordinates.
    Nonnegative coordinate weights are normalized to sum to one. The returned
    prediction is the convex mixture across candidate rows, also on coordinates
    assigned zero risk weight. Rows identical on the weighted coordinates split
    their mass equally.

    For mixture lambda and its mean q, the dual certificate is the weighted
    candidate variance about q. It lower-bounds minimax risk because every
    prediction incurs at least this much lambda-average loss. The worst row at
    q gives an upper bound. These bounds, up to floating-point precision, apply
    to the supplied candidate set and coordinate weights.
    """
    raw = np.asarray(candidates)
    if raw.ndim != 2 or min(raw.shape) < 1 or raw.dtype.kind not in "biuf" or not ((raw >= 0) & (raw <= 1)).all():
        raise ValueError("candidates must be a nonempty matrix of finite probabilities in [0, 1]")
    values = raw.astype(np.float64, copy=False)
    weights = np.ones(values.shape[1]) if target_weights is None else np.asarray(target_weights)
    if weights.shape != (values.shape[1],) or weights.dtype.kind not in "biuf" or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("target_weights must contain one finite nonnegative weight per coordinate")
    weights = weights.astype(np.result_type(weights.dtype, np.float64), copy=False)
    largest = weights.max()
    if largest == 0:
        raise ValueError("target_weights must have positive total mass")
    weights = np.asarray(weights / largest, dtype=np.float64)
    weights /= weights.sum()

    positive = weights > 0
    distinct, inverse, counts = np.unique(values[:, positive], axis=0, return_inverse=True, return_counts=True)
    points = (distinct - distinct[0]) * np.sqrt(weights[positive])
    group_mix = _fit_mixture(points)
    inverse = inverse.reshape(-1)
    mixture = (group_mix / counts)[inverse]
    mixture /= mixture.sum()
    prediction = np.clip(mixture @ values, 0, 1)
    if len(distinct) == 1:
        prediction[positive] = distinct[0]
    errors = np.square(values - prediction) @ weights
    upper = float(errors.max())
    return BrierCenter(prediction=prediction, upper=upper, lower=min(upper, float(mixture @ errors)), mixture=mixture)
