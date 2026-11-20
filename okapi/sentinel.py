"""Rank an observed audit statistic against independent clean reference batches."""

import numpy as np


def upper_tail_pvalues(reference, observed, uniforms):
    """Randomized upper-tail conformal ranks for arbitrary tied batch scores.

    Pass one independently uniform [0,1) variate per observed batch. For
    exchangeable clean scores, with higher more anomalous, each resulting p
    is uniform marginally over reference, observation and tie randomization.
    Several observations using one calibration set have dependent p-values.
    """
    clean = np.asarray(reference, dtype=float)
    x = np.asarray(observed, dtype=float)
    u = np.asarray(uniforms, dtype=float)
    if clean.ndim != 1 or len(clean) == 0 or not np.isfinite(clean).all():
        raise ValueError("reference must be a nonempty finite vector")
    if u.shape != x.shape or not np.isfinite(x).all():
        raise ValueError("supply one uniform per finite observation")
    if not (np.isfinite(u).all() and np.all((0 <= u) & (u < 1))):
        raise ValueError("uniforms must be in [0,1)")
    ordered = np.sort(clean)
    lo = np.searchsorted(ordered, x, side="left")
    hi = np.searchsorted(ordered, x, side="right")
    return (len(clean) - hi + (hi - lo + 1) * u) / (len(clean) + 1)
