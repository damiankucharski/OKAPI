import numpy as np
import pytest
from scipy.special import expit
from okapi.sigmoid_merge import CURVATURE_FACTOR, merge_sigmoids


def test_all_widths_bound_errors_outside_training_support():
    rng = np.random.default_rng(10)
    w = rng.normal(size=(7, 3))
    w[3] = w[2]
    w[5:] = 0
    b = np.array([-2, 0.3, 1, 1, 3, -2, 7.0])
    a = np.array([0, 0.1, 0.05, 0.15, 0.2, 0.3, 0.2])
    x = rng.normal(size=(1024, 3)) * 4 + 1
    truth = expit(x @ w.T + b) @ a
    models = merge_sigmoids(w, b, a, np.zeros(3), np.eye(3), widths=[1, 2, 4, 7])
    for k, model in models.items():
        assert len(model.mixing) <= k
        assert np.all(np.abs(truth - model.predict(x)) <= model.error_bound(x) + 1e-12)
    np.testing.assert_allclose(models[7].predict(x), truth, atol=1e-12)
    np.testing.assert_array_equal(models[7].bound_matrix, np.zeros((4, 4)))
    assert len(models[7].mixing) == 4
    assert np.linalg.eigvalsh(models[1].bound_matrix - models[2].bound_matrix).min() > -1e-12


def test_two_unit_bound_is_weighted_logit_variance():
    w = np.array([[1.0], [-0.5]])
    b = np.array([-0.25, 0.75])
    a = np.array([0.3, 0.7])
    x = np.array([0.4])
    model = merge_sigmoids(w, b, a, [0.0], [[1.0]], widths=[1])[1]
    z = w @ x + b
    assert model.error_bound(x) == pytest.approx(CURVATURE_FACTOR * (a @ (z - a @ z) ** 2))
    assert model.predict(x) == pytest.approx(expit(a @ z))
