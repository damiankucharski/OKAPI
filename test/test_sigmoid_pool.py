import numpy as np
import pytest
from scipy.special import expit

from okapi.sigmoid_pool import collapse_sigmoids


def test_center_value_and_derivative_match_direct_pool():
    w = np.array([[2.0, -0.3], [-0.6, 2.0], [1.0, 3.0]])
    a = np.array([0.2, 0.3, 0.5])
    b = np.array([1.0, -3.0, 0.2])
    m = np.array([1.0, -0.7])
    weights, bias = collapse_sigmoids(w, b, a, m)
    p = expit(w @ m + b)
    f = a @ p
    np.testing.assert_allclose(expit(m @ weights + bias), f)
    np.testing.assert_allclose(f * (1 - f) * weights, (a * p * (1 - p)) @ w)


@pytest.mark.parametrize("bias", [0.3, 1000.0, -1000.0])
def test_identical_extreme_sigmoids_collapse_exactly(bias):
    for cov in [None, np.eye(2)]:
        w, b = collapse_sigmoids(
            [[0.8, -1.0], [0.8, -1.0]], [bias, bias], [0.3, 0.7], [0.1, 0.3], covariance=cov
        )
        np.testing.assert_allclose(w, [0.8, -1.0], atol=1e-12)
        np.testing.assert_allclose(b, bias, atol=1e-12)


def test_constant_pool_and_invalid_covariance():
    w, b = collapse_sigmoids(
        [[0.0, 0.0], [0.0, 0.0]], [1.0, -1.0], [3.0, 1.0], [0.0, 0.0], covariance=np.zeros((2, 2))
    )
    np.testing.assert_array_equal(w, [0, 0])
    np.testing.assert_allclose(expit(b), (0.75 * expit(1) + 0.25 * expit(-1)))
    with pytest.raises(ValueError):
        collapse_sigmoids([[1.0, 0.0]], [0.0], [1.0], [0, 0], covariance=-np.eye(2))
