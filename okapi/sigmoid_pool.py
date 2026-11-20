"""Local, parameter-only approximation of a probability pool by one affine sigmoid."""

import numpy as np
from scipy.special import log_expit, logsumexp


def collapse_sigmoids(coefficients, intercepts, mixing_weights, center, *, covariance=None):
    """Return a sigmoid matching the pool at center and its local or symmetric-secant slope.

    Without covariance, the derivative also matches at center. Providing a feature covariance
    replaces the slope magnitude by a secant through center ± one score standard deviation.
    The returned coefficients and intercept use the input feature coordinates.
    """
    w = np.asarray(coefficients, dtype=float)
    b, alpha, center = [np.asarray(a, dtype=float) for a in (intercepts, mixing_weights, center)]
    if (
        w.ndim != 2
        or not w.size
        or b.shape != (len(w),)
        or alpha.shape != b.shape
        or center.shape != (w.shape[1],)
    ):
        raise ValueError(
            "Expected K by D coefficients, K intercepts and weights, and one D-dimensional center"
        )
    if (
        not all(np.isfinite(v).all() for v in (w, b, alpha, center))
        or (alpha < 0).any()
        or alpha.max() <= 0
    ):
        raise ValueError(
            "Inputs must be finite and mixing weights must be nonnegative with positive mass"
        )
    active = alpha > 0
    alpha = alpha[active] / alpha.max()
    log_alpha = np.log(alpha / alpha.sum())
    w, b = w[active], b[active]
    logits = w @ center + b

    def pool_logit(z):
        return logsumexp(log_alpha + log_expit(z)) - logsumexp(log_alpha + log_expit(-z))

    value = pool_logit(logits)
    # Work in log space so even a unanimous but extreme prediction retains its slope.
    lp = log_expit(logits)
    lq = log_expit(-logits)
    factors = np.exp(log_alpha + lp + lq - logsumexp(log_alpha + lp) - logsumexp(log_alpha + lq))
    weights = factors @ w
    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if (
            cov.shape != (len(center), len(center))
            or not np.isfinite(cov).all()
            or not np.allclose(cov, cov.T)
        ):
            raise ValueError("Covariance must be a finite symmetric D by D matrix")
        eigen = np.linalg.eigvalsh(cov)
        if eigen[0] < -1e-10 * max(1, float(eigen[-1])):
            raise ValueError("Covariance must be positive semidefinite")
        axis = cov @ weights
        length = np.sqrt(max(float(weights @ axis), 0))
        if length > 1e-12:
            shift = w @ (axis / length)
            weights = weights * (pool_logit(logits + shift) - pool_logit(logits - shift)) / (2 * length)
    return weights, float(value - center @ weights)
