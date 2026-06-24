"""Unit tests for okapi.meta (pool meta-analysis: bootstrap competence CIs,
error correlation / redundancy, per-model trust, diversity groups)."""

import numpy as np
import pytest

from okapi.meta import (
    bootstrap_fitness_ci,
    diversity_groups,
    error_correlation,
    model_trust,
    redundancy,
)


def _pred_from_classes(pred_cls, c, conf=0.9):
    pred_cls = np.asarray(pred_cls)
    n = len(pred_cls)
    p = np.full((n, c), (1.0 - conf) / (c - 1))
    p[np.arange(n), pred_cls] = conf
    return p


def _stack(models, c):
    return np.stack([_pred_from_classes(m, c) for m in models], axis=1)


def _model(y, acc, rng, c=2):
    """A model of given accuracy: correct rows predict y, errors predict y+1."""
    correct = rng.random(len(y)) < acc
    return np.where(correct, y, (y + 1) % c)


# --------------------------------- bootstrap CI ---------------------------------


def test_bootstrap_ci_brackets_and_orders_accuracy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    preds = _stack([_model(y, 0.9, rng), _model(y, 0.6, rng)], 2)
    lo, mean, hi = bootstrap_fitness_ci(preds, y, n_boot=200, alpha=0.1, rng=rng)
    assert (lo <= mean).all() and (mean <= hi).all()
    assert mean[0] == pytest.approx(0.9, abs=0.03)
    assert mean[1] == pytest.approx(0.6, abs=0.03)
    assert mean[0] > mean[1]


def test_bootstrap_ci_perfect_model_is_degenerate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 800)
    lo, mean, hi = bootstrap_fitness_ci(_stack([y.copy()], 2), y, n_boot=50, rng=rng)
    assert mean[0] == 1.0 and lo[0] == 1.0 and hi[0] == 1.0


# ------------------------------- error correlation -------------------------------


def test_error_correlation_identical_vs_independent():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 3000)
    errs = rng.random(len(y)) < 0.3
    a = np.where(~errs, y, 1 - y)
    b = a.copy()  # identical error rows
    c = _model(y, 0.7, rng)  # independent errors
    corr = error_correlation(_stack([a, b, c], 2), y)
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr), 1.0)
    assert corr[0, 1] > 0.9
    assert abs(corr[0, 2]) < 0.15


def test_error_correlation_dead_model_uncorrelated():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    preds = _stack([y.copy(), _model(y, 0.7, rng)], 2)  # first never errs
    corr = error_correlation(preds, y)
    assert corr[0, 1] == 0.0 and corr[1, 0] == 0.0
    assert corr[0, 0] == 1.0


def test_redundancy_higher_for_colluders():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 3000)
    errs = rng.random(len(y)) < 0.3
    a = np.where(~errs, y, 1 - y)
    b = a.copy()
    c = _model(y, 0.7, rng)
    red = redundancy(error_correlation(_stack([a, b, c], 2), y))
    assert red[0] > red[2]  # a collides with b; c is independent


# ----------------------------------- trust -----------------------------------


def test_model_trust_range_and_separation_binary():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 3000)
    good = _model(y, 0.9, rng)
    chance = rng.integers(0, 2, 3000)  # independent of y
    trust = model_trust(_stack([good, chance], 2), y, rng=rng)
    assert trust.shape == (2,)
    assert (trust >= 0).all() and (trust <= 1).all()
    assert trust[0] > 0.6
    assert trust[1] < 0.15


def test_model_trust_multiclass():
    rng = np.random.default_rng(6)
    c = 4
    y = rng.integers(0, c, 2500)
    good = _model(y, 0.85, rng, c=c)
    chance = rng.integers(0, c, 2500)
    trust = model_trust(_stack([good, chance], c), y, rng=rng)
    assert trust[0] > 0.6 and trust[1] < 0.2


# ------------------------------- diversity groups -------------------------------


def test_diversity_groups_separate_colluders_from_independents():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 3000)
    errs = rng.random(len(y)) < 0.3
    a = np.where(~errs, y, 1 - y)
    b = a.copy()  # collude with a
    c = _model(y, 0.7, rng)
    d = _model(y, 0.7, rng)  # independent of a and c
    groups = diversity_groups(_stack([a, b, c, d], 2), y, threshold=0.5)
    assert groups[0] == groups[1]      # colluders share a group
    assert groups[2] != groups[0]      # independents are separate
    assert groups[3] != groups[0]
