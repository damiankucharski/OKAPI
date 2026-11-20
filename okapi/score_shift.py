"""Estimate target class prevalence and an additive shift of a scalar expert score."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreShift:
    prior: float
    offset: float
    cdf_rmse: float


def _sample(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Supply a nonempty one-dimensional array of finite scores")
    return x


class ReferenceScoreCDF:
    """Binary source score distributions. Offsets returned by fit are in original score units."""

    def __init__(self, negative, positive):
        negative, positive = _sample(negative), _sample(positive)
        source = np.concatenate((negative, positive))
        self.scale = source.std()
        if self.scale <= 0:
            raise ValueError("Reference scores must have positive pooled variance")
        lower, upper = np.quantile(source, [0.005, 0.995])
        self.grid = np.linspace(lower - 1.5 * self.scale, upper + 1.5 * self.scale, 151)
        self.offsets = np.linspace(-1.5, 1.5, 301) * self.scale
        values = self.grid[None, :] - self.offsets[:, None]
        self.negative = np.searchsorted(np.sort(negative), values, side="right") / len(negative)
        self.difference = (
            np.searchsorted(np.sort(positive), values, side="right") / len(positive) - self.negative
        )
        self.class_means = np.array([negative.mean(), positive.mean()])

    def fit(self, target, *, allow_offset=True):
        target = _sample(target)
        observed = np.searchsorted(np.sort(target), self.grid, side="right") / len(target)
        residual = observed - self.negative
        denom = (self.difference**2).sum(axis=1)
        priors = np.divide(
            (residual * self.difference).sum(axis=1),
            denom,
            out=np.full_like(denom, 0.5),
            where=denom > 0,
        )
        priors = np.clip(priors, 0.02, 0.98)
        loss = np.mean((residual - priors[:, None] * self.difference) ** 2, axis=1)
        if allow_offset:
            tied = np.flatnonzero(np.isclose(loss, loss.min(), rtol=0, atol=1e-14))
            best = tied[np.argmin(np.abs(self.offsets[tied]))]
        else:
            best = np.argmin(np.abs(self.offsets))
        return ScoreShift(float(priors[best]), float(self.offsets[best]), float(np.sqrt(loss[best])))

    def prior_from_mean(self, target):
        """Compare with a prior estimate that assumes fixed class conditional means."""
        target = _sample(target)
        difference = self.class_means[1] - self.class_means[0]
        if difference == 0:
            return 0.5
        return float(np.clip((target.mean() - self.class_means[0]) / difference, 0.02, 0.98))
