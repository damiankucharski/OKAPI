"""Check whether deployment feature means fit an annotated two-context mixture."""

from dataclasses import dataclass

import numpy as np

from okapi.selection import estimate_context_prevalence


@dataclass(frozen=True)
class ContextFit:
    prevalence: float
    max_residual: float
    p_value: float
    status: str


def assess_context_mixture(target, reference, changed, fallback, *, alpha=0.05, draws=499, seed=0):
    """Approximate goodness-of-fit check for a two-context mean segment.

    Inputs are samples by features. Independent source and target examples
    are assumed. A multiplier Gaussian approximation propagates uncertainty
    from the target mean and *both* source means. Each null realization refits
    the prevalence, including the [0,1] constraint. The diagnostic checks only
    the supplied observable moments. A compatible fit cannot certify stable
    labels or stable risks. With dependent rows, aggregate to independent units.
    """
    a, b, x = [np.asarray(v, dtype=float) for v in (reference, changed, target)]
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("target requires at least two observation rows")
    if not 0 < alpha < 1 or draws < 1:
        raise ValueError("alpha must be in (0,1) and draws must be positive")
    p = estimate_context_prevalence(x.mean(0), a, b, fallback)
    if min(len(a), len(b)) < 2:
        return ContextFit(p, np.nan, np.nan, "insufficient_source")
    origin = a.mean(0)
    direction = b.mean(0) - origin
    length2 = direction @ direction
    if length2 < 1e-12:
        return ContextFit(p, np.nan, np.nan, "unidentified")
    fitted = origin + p * direction
    residual = x.mean(0) - fitted

    def mean_cov(v):
        d = v - v.mean(axis=0)
        return d.T @ d / (len(v) * (len(v) - 1))

    covariance = mean_cov(x) + (1 - p) ** 2 * mean_cov(a) + p**2 * mean_cov(b)
    scale = np.maximum(np.sqrt(np.maximum(covariance.diagonal(), 0)), 1e-12)
    score = np.max(abs(residual) / scale)
    eig, vectors = np.linalg.eigh(covariance)
    normal = np.random.default_rng(seed).normal(size=(draws, len(scale)))
    noise = (normal * np.sqrt(np.maximum(eig, 0))) @ vectors.T
    null_p = np.clip(p + noise @ direction / length2, 0, 1)
    null_deviation = noise + (p - null_p)[:, None] * direction
    null_scores = np.max(abs(null_deviation) / scale, axis=1)
    pvalue = (1 + (null_scores >= score - 1e-10).sum()) / (draws + 1)
    return ContextFit(
        float(p), float(score), float(pvalue), "incompatible" if pvalue <= alpha else "compatible"
    )


def invariant_mean_pvalue(reference, target, *, draws=399, seed=0):
    """Randomization test on feature columns prespecified as invariant.

    Source rows used to choose the columns must be disjoint from reference.
    Under exchangeable rows and the specified invariance, Monte Carlo
    permutation p-values control false alarms up to the tie convention.
    """
    a, b = [np.asarray(v, dtype=float) for v in (reference, target)]
    if a.ndim != 2 or b.ndim != 2 or a.shape[1:] != b.shape[1:] or a.shape[1] == 0:
        raise ValueError("reference and target must have matching feature columns")
    if min(len(a), len(b), draws) < 1 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("draws and sample sizes must be positive and features finite")
    values = np.concatenate([a, b])
    values -= values.mean(0)
    scale = values.std(0, ddof=1)
    selected = scale > 1e-12
    if not np.any(selected):
        return 1.0
    values = values[:, selected] / scale[selected]
    n = len(a)
    observed = np.max(np.abs(values[:n].mean(0)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        rand = rng.random((size, len(values)))
        rows = np.argpartition(rand, n - 1, axis=1)[:, :n]
        scores = np.max(abs(values[rows].mean(1)), axis=1)
        exceed += np.count_nonzero(scores >= observed - 1e-10)
    return float((exceed + 1) / (draws + 1))
