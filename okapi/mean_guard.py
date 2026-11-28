"""Select a frozen ensemble tree against the native equal-weight provider mean."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from numbers import Integral, Real

import numpy as np

from okapi.globals import BACKEND as B
from okapi.paired_fitness import _group_index, _loss_array, arithmetic_mean_tree
from okapi.tree import Tree


@dataclass(frozen=True)
class MeanGuardDecision:
    """index is None when the guard selects the mean. Gains use the loss's units."""

    index: int | None
    best_index: int | None
    gains: tuple[float, ...]
    threshold: float | None
    n_units: int
    reason: str


def select_against_mean(mean_losses, candidate_losses, *, groups=None, alpha: float = .05, draws: int = 4095,
                        seed: int = 0, min_units: int = 8) -> MeanGuardDecision:
    """Choose a frozen challenger only if its validation gain exceeds a common noise threshold.

    Supply lower-is-better paired losses for the baseline (N,) and ordered challengers
    (K, N). Fit and freeze every contender, including any hyperparameter selection, on
    separate data. Each row must be an independent sampling unit, or provide group IDs
    so losses are averaged within units and every independent group has equal weight.

    For each shared Rademacher draw, take the maximum over zero and the K centered,
    signed sums of paired gains divided by sqrt(U*(U-1)), for U independent units.
    The threshold is the upper empirical (1-alpha) quantile ('higher'). The largest
    mean gain must strictly exceed it. Ties among challengers favor input order.
    This multiplier approximation estimates sampling noise for a fixed family with
    finite-variance losses. Its nominal alpha is approximate, especially for few
    units or skewed contrasts. Fewer than min_units, or an empty family, selects
    the baseline. Evaluate the resulting choice on an untouched test set.
    """
    if not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    for value, name, minimum in ((draws, "draws", 1), (min_units, "min_units", 2), (seed, "seed", 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    baseline = _loss_array(mean_losses)
    losses = _loss_array(candidate_losses)
    if baseline.ndim != 1:
        raise ValueError("mean_losses must have shape (N,)")
    if losses.shape == (0,):
        losses = losses.reshape(0, baseline.size)
    if losses.ndim != 2 or losses.shape[1] != baseline.size:
        raise ValueError("candidate_losses must have shape (K, N), aligned to mean_losses")
    contrasts = baseline - losses
    if groups is not None:
        membership, counts = _group_index(groups, baseline.size)
        grouped = np.empty((len(contrasts), len(counts)))
        for i, contrast in enumerate(contrasts):
            grouped[i] = np.bincount(membership, weights=contrast) / counts
        contrasts = grouped
    n_units = contrasts.shape[1]
    if not len(contrasts):
        return MeanGuardDecision(None, None, (), None, n_units, "no_candidates")
    if not n_units:
        return MeanGuardDecision(None, None, (), None, 0, "too_few_units")
    gains = contrasts.mean(axis=1)
    best = int(gains.argmax())
    improvements = tuple(float(gain) for gain in gains)
    if n_units < min_units:
        return MeanGuardDecision(None, best, improvements, None, n_units, "too_few_units")

    centered = (contrasts - gains[:, None]) / np.sqrt(float(n_units) * (n_units - 1))
    rng = np.random.Generator(np.random.PCG64(seed))
    maxima = np.empty(draws)
    # Keep temporary sign and score matrices bounded on large validation sets.
    batch_size = max(1, min(256, 1_000_000 // max(n_units, len(losses))))
    for start in range(0, draws, batch_size):
        end = min(start + batch_size, draws)
        signs = np.where(rng.random((end - start, n_units)) < .5, -1., 1.)
        signed_gains = np.einsum("bu,ku->bk", signs, centered)
        maxima[start:end] = np.maximum(0., signed_gains.max(axis=1))
    # Order-statistic spelling also supports NumPy versions before method='higher'.
    rank = ceil((draws - 1) * (1 - alpha))
    threshold = float(np.partition(maxima, rank)[rank])
    accepted = gains[best] > threshold
    return MeanGuardDecision(best if accepted else None, best, improvements, threshold, n_units,
                             "accepted" if accepted else "below_threshold")


def _bind_copy(tree, tensors):
    copy = tree.copy()
    for node in copy.value_nodes:
        node.value = tensors[node.id]
    return copy


def select_tree_against_mean(candidates: Sequence[Tree], fit_tensors: Mapping, validation_tensors: Mapping,
                             labels, row_loss: Callable, **options) -> tuple[Tree, MeanGuardDecision]:
    """Compare source-fitted trees on fresh rows and return one tree bound to fit_tensors.

    Native mean and challengers use the active backend and postprocessing function.
    Postprocessing must preserve the arithmetic mean (as probability normalization
    does on already normalized vectors). row_loss(prediction, labels) returns a
    finite vector with one loss per validation row. Validation and fit mappings use
    the same provider IDs. Pass groups, alpha, draws, seed or min_units to configure
    select_against_mean. This step rebinds private copies and preserves search state.
    """
    if set(validation_tensors) != set(fit_tensors):
        raise ValueError("Validation must supply exactly the fit provider IDs")
    rows = len(labels)
    shapes = [tuple(B.shape(value)) for value in validation_tensors.values()]
    if not shapes or not shapes[0] or shapes[0][0] != rows or any(s != shapes[0] for s in shapes):
        raise ValueError("Validation providers must have matching shapes aligned to the labels")
    baseline = arithmetic_mean_tree(fit_tensors)
    candidates = tuple(candidates)

    def evaluate(tree):
        with B.no_grad():
            prediction = _bind_copy(tree, validation_tensors).predict()
            losses = _loss_array(row_loss(prediction, labels))
        if losses.shape != (rows,):
            raise ValueError("row_loss must return a vector with one scalar per validation row")
        return losses

    baseline_losses = evaluate(baseline)
    losses = [evaluate(candidate) for candidate in candidates]
    decision = select_against_mean(baseline_losses, losses, **options)
    chosen = baseline if decision.index is None else _bind_copy(candidates[decision.index], fit_tensors)
    return chosen, decision
