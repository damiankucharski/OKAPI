import numpy as np
import pytest

from okapi.brier_center import fit_brier_center


def check_certificate(pool, weights, result):
    weights = weights / weights.sum()
    assert result.mixture.shape == (len(pool),)
    assert result.mixture.min() >= 0
    assert result.mixture.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(result.prediction, result.mixture @ pool, atol=1e-14, rtol=0)
    risk = (pool - result.prediction) ** 2 @ weights
    assert result.upper == pytest.approx(risk.max(), abs=1e-14)
    assert result.lower == pytest.approx(result.mixture @ risk, abs=1e-14)
    assert 0 <= result.upper - result.lower < 1e-7
    # Variance decomposition shows this lower bound applies to all choices q.
    for q in [np.zeros(len(weights)), np.ones(len(weights)), np.linspace(0, 1, len(weights))]:
        average_loss = result.mixture @ ((pool - q) ** 2 @ weights)
        assert average_loss == pytest.approx(result.lower + (result.prediction - q) ** 2 @ weights, abs=1e-13)


def test_triangle_and_singleton_control_rows():
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    fit = fit_brier_center(triangle)
    check_certificate(triangle, np.ones(2), fit)
    np.testing.assert_allclose(fit.prediction, [0.5, np.sqrt(3) / 6], atol=1e-8)
    assert fit.upper == pytest.approx(1 / 6)
    controls = np.eye(6)
    fit = fit_brier_center(controls)
    np.testing.assert_allclose(fit.prediction, np.ones(6) / 6, atol=1e-8)
    assert fit.upper == pytest.approx(5 / 36)
    check_certificate(controls, np.ones(6), fit)


def test_singleton_duplicates_and_zero_weight_convention():
    single = np.array([[0.1, 0.75, 1.0]])
    fit = fit_brier_center(single)
    np.testing.assert_array_equal(fit.prediction, single[0])
    assert fit.lower == fit.upper == 0.0
    pool = np.array([[0, 0], [0, 1], [0.2, 0.7], [1, 0.6]])
    weights = np.array([1.0, 0])
    fit = fit_brier_center(pool, weights)
    np.testing.assert_allclose(fit.mixture, [0.25, 0.25, 0, 0.5])
    np.testing.assert_allclose(fit.prediction, [0.5, 0.55])
    assert fit.upper == fit.lower == 0.25
    check_certificate(pool, weights, fit)


def test_weighted_targets_and_actual_numerical_gap():
    rng = np.random.default_rng(81)
    pool = rng.random((20, 17))
    weights = rng.uniform(0.05, 1, size=17)
    weights[:2] = 0
    fit = fit_brier_center(pool, weights)
    check_certificate(pool, weights, fit)
    # Repeating a candidate and rescaling coordinate weights leave nonzero-weight
    # predictions invariant and preserve the minimax objective.
    duplicated = fit_brier_center(np.vstack((pool, pool[0], pool[0])), weights * 1e301)
    np.testing.assert_allclose(duplicated.prediction[2:], fit.prediction[2:], atol=1e-8)
    assert duplicated.upper == pytest.approx(fit.upper, abs=1e-10)


@pytest.mark.parametrize(
    ("pool", "weights"),
    [
        ([], None),
        (np.ones((1, 0)), None),
        ([[0.1], [1.1]], None),
        ([[np.nan]], None),
        ([[0.1]], [0]),
        ([[0.1]], [np.inf]),
        ([[0, 1]], [1]),
        ([[0, 1]], [-1, 2]),
        ([[0.4]], ["x"]),
    ],
)
def test_invalid_probability_or_weight_input(pool, weights):
    with pytest.raises(ValueError):
        fit_brier_center(pool, weights)
