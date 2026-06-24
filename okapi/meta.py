"""Pool meta-analysis for prediction fusion.

Given a stack of ``K`` calibrated base-model probability predictions ``[N, K, C]``
and labels ``y``, these (pure-numpy) utilities compute the *meta* signals OKAPI can
use to tell a genuine specialist from a poisoned/colluding base model, **without
looking at test data**:

- :func:`bootstrap_fitness_ci` — a confidence interval on each model's competence,
  so trust can be a conservative *lower* bound rather than a point estimate;
- :func:`error_correlation` / :func:`redundancy` — which models make the *same*
  mistakes (a colluding majority) vs make complementary errors (real diversity);
- :func:`model_trust` — a per-model trust in ``[0, 1]`` (competence above chance,
  via the CI lower bound), consumed by trust-gated operators and trust-aware
  selection;
- :func:`diversity_groups` — a coarse grouping of behaviourally-redundant models,
  for diversity-aware seeding.

Fundamental limit (deliberate): a base model whose inflated fit competence is
*uniformly mixed* across rows (a random-subset "memoriser") is **indistinguishable
from a genuinely strong model by any fit-only statistic** — every resample sees the
same inflated mean. Trust therefore (correctly) cannot flag it; robustness to that
case comes from parsimony/flooring, not detection. Trust *does* separate the cases
whose competence is low in aggregate or whose errors collude (clustered
specialists, agreeing-garbage majorities).
"""

from __future__ import annotations

import numpy as np


def _argmax(preds: np.ndarray) -> np.ndarray:
    """Class predictions ``[N, K]`` from a ``[N, K, C]`` probability stack."""
    return np.asarray(preds).argmax(axis=2)


def bootstrap_fitness_ci(preds: np.ndarray, y: np.ndarray, *, n_boot: int = 200,
                         alpha: float = 0.1, rng: np.random.Generator | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-model accuracy with a bootstrap confidence interval.

    Returns ``(lo, mean, hi)`` each shape ``[K]`` — the ``alpha/2`` and
    ``1-alpha/2`` quantiles of the accuracy over ``n_boot`` row-resamples, and the
    point estimate. The CI *width* reflects how uncertain a model's competence is
    given the fit sample (wide for small/noisy fit), so a conservative trust uses
    ``lo``.
    """
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y).ravel()
    correct = (_argmax(preds) == y[:, None]).astype(float)  # [N, K]
    n, k = correct.shape
    boot = np.empty((n_boot, k))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = correct[idx].mean(axis=0)
    lo = np.quantile(boot, alpha / 2.0, axis=0)
    hi = np.quantile(boot, 1.0 - alpha / 2.0, axis=0)
    return lo, correct.mean(axis=0), hi


def error_correlation(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``[K, K]`` Pearson correlation of the per-model *error* indicators.

    Two models that fail on the same rows (a colluding / redundant pair) have
    correlation near 1; models with independent errors near 0. Models that never
    err (zero-variance error vector) get correlation 0 with everyone (undefined ->
    treated as uncorrelated), diagonal 1.
    """
    y = np.asarray(y).ravel()
    err = (_argmax(preds) != y[:, None]).astype(float)  # [N, K]
    e = err - err.mean(axis=0, keepdims=True)
    std = e.std(axis=0, keepdims=True)
    safe = std.copy()
    safe[safe == 0] = 1.0
    z = e / safe
    corr = (z.T @ z) / err.shape[0]
    # zero-variance models: no meaningful correlation -> 0 off-diagonal
    dead = (std.ravel() == 0)
    corr[dead, :] = 0.0
    corr[:, dead] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def redundancy(corr: np.ndarray) -> np.ndarray:
    """Per-model mean off-diagonal error-correlation ``[K]`` (collusion score)."""
    c = corr.copy()
    np.fill_diagonal(c, np.nan)
    return np.nanmean(c, axis=1)


def model_trust(preds: np.ndarray, y: np.ndarray, *, n_boot: int = 200,
                alpha: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Per-model trust in ``[0, 1]``: conservative competence above chance.

    ``trust_i = clip((acc_lo_i - chance) / (1 - chance), 0, 1)`` where ``acc_lo`` is
    the bootstrap CI lower bound and ``chance = 1/C``. A chance-level or
    confidently-wrong-about-``y`` model (e.g. agreeing garbage) -> ~0; a reliably
    competent model -> high. Conservative (uses the lower bound) so a model that is
    only *luckily* good on a small fit is not over-trusted.
    """
    lo, _, _ = bootstrap_fitness_ci(preds, y, n_boot=n_boot, alpha=alpha, rng=rng)
    chance = 1.0 / np.asarray(preds).shape[2]
    return np.clip((lo - chance) / (1.0 - chance), 0.0, 1.0)


def diversity_groups(preds: np.ndarray, y: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """Coarse grouping ``[K]`` of behaviourally-redundant models.

    Greedy single-linkage on the error-correlation: models whose error vectors
    correlate above ``threshold`` join the same group. Useful for diversity-aware
    seeding (pick representatives across groups rather than within one).
    """
    corr = error_correlation(preds, y)
    k = corr.shape[0]
    groups = -np.ones(k, dtype=int)
    g = 0
    for i in range(k):
        if groups[i] >= 0:
            continue
        groups[i] = g
        for j in range(i + 1, k):
            if groups[j] < 0 and corr[i, j] >= threshold:
                groups[j] = g
        g += 1
    return groups
