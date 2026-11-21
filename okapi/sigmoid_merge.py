"""Merge a weighted sigmoid ensemble into a certified smaller hidden layer."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, log_expit, logsumexp

CURVATURE_FACTOR = 1 / (12 * np.sqrt(3))


@dataclass
class MergedSigmoids:
    weights: np.ndarray
    intercepts: np.ndarray
    mixing: np.ndarray
    bound_matrix: np.ndarray

    def predict(self, x):
        return expit(np.asarray(x) @ self.weights.T + self.intercepts) @ self.mixing

    def error_bound(self, x):
        """Pointwise upper bound on absolute probability error relative to the original pool."""
        x = np.asarray(x, dtype=float)
        quadratic = np.einsum("...i,ij,...j->...", x, self.bound_matrix[:-1, :-1], x)
        raw = quadratic + 2 * x @ self.bound_matrix[:-1, -1] + self.bound_matrix[-1, -1]
        return np.clip(raw, 0, 1)


def merge_sigmoids(coefficients, intercepts, mixing_weights, center, covariance, *, widths):
    """Build Ward merges in the logit metric and return models at requested width caps.

    The center and covariance govern which components to merge. Every returned model's
    error_bound applies to any finite input, including under a different distribution.
    """
    theta = np.asarray(coefficients, dtype=float)
    b, alpha, mu = [np.asarray(v, dtype=float) for v in (intercepts, mixing_weights, center)]
    cov = np.asarray(covariance, dtype=float)
    if (
        theta.ndim != 2
        or not theta.size
        or b.shape != (len(theta),)
        or alpha.shape != b.shape
        or mu.shape != (theta.shape[1],)
    ):
        raise ValueError(
            "Specify K by D coefficients, K intercepts and weights, and a D-dimensional center"
        )
    if cov.shape != (len(mu), len(mu)) or not all(
        np.isfinite(v).all() for v in (theta, b, alpha, mu, cov)
    ):
        raise ValueError("Inputs must be finite and covariance must be D by D")
    if (alpha < 0).any() or alpha.max() <= 0 or not np.allclose(cov, cov.T):
        raise ValueError("Use nonnegative active weights and a symmetric covariance")
    eig = np.linalg.eigvalsh(cov)
    if eig[0] < -1e-10 * max(1, float(eig[-1])):
        raise ValueError("Covariance must be positive semidefinite")
    widths = list(widths)
    if not widths or any(
        not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1 for k in widths
    ):
        raise ValueError("Provide positive integer width caps")

    alpha = alpha / alpha.max()
    alpha /= alpha.sum()
    active = alpha > 0
    alpha = alpha[active]
    theta = np.column_stack((theta[active], b[active]))
    constant = (theta[:, :-1] == 0).all(axis=1)
    if constant.sum() > 1:
        log_masses = np.log(alpha[constant])
        logits = theta[constant, -1]
        constant_logit = logsumexp(log_masses + log_expit(logits)) - logsumexp(
            log_masses + log_expit(-logits)
        )
        theta = np.vstack((theta[~constant], np.r_[np.zeros(len(mu)), constant_logit]))
        alpha = np.r_[alpha[~constant], alpha[constant].sum()]
    theta, groups = np.unique(theta, axis=0, return_inverse=True)
    alpha = np.bincount(groups, weights=alpha)

    augmented_mean = np.r_[mu, 1.0]
    moment = np.outer(augmented_mean, augmented_mean)
    moment[:-1, :-1] += cov
    pending = {k: min(k, len(theta)) for k in widths}
    models = {}
    bound = np.zeros_like(moment)
    while pending:
        for k in list(pending):
            if pending[k] == len(theta):
                models[k] = MergedSigmoids(
                    theta[:, :-1].copy(), theta[:, -1].copy(), alpha.copy(), bound.copy()
                )
                del pending[k]
        if not pending:
            break
        i, j = np.triu_indices(len(theta), k=1)
        delta = theta[i] - theta[j]
        masses = (alpha[i] / (alpha[i] + alpha[j])) * alpha[j]
        squared = np.einsum("ki,ij,kj->k", delta, moment, delta, optimize=True)
        best = np.argmin(masses * np.maximum(squared, 0))
        left, right = i[best], j[best]
        bound += CURVATURE_FACTOR * masses[best] * np.outer(delta[best], delta[best])
        total = alpha[left] + alpha[right]
        theta[left] = (alpha[left] / total) * theta[left] + (alpha[right] / total) * theta[right]
        alpha[left] = total
        theta = np.delete(theta, right, axis=0)
        alpha = np.delete(alpha, right)
    return models
