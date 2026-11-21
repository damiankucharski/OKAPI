import numpy as np
import pytest
from scipy.special import expit

from okapi.binary_pairs import compile_binary_pairs, midpoint_certificate


def test_three_centers_with_one_oriented_boundary_and_arbitrary_differences():
    rng = np.random.default_rng(51)
    x = rng.normal(size=(400, 4))
    u = rng.normal(size=4)
    s = np.array([0.25, 1.0, 3.0])
    centers = s[:, None] * u
    bias = 0.2 * s
    d = rng.normal(size=centers.shape) * 4
    w = np.vstack([centers + d, centers - d])
    b = np.r_[bias + 2, bias - 2]
    router = compile_binary_pairs(w, b, [(0, 3), (1, 4), (2, 5)])
    pred, cert = router.predict(x, return_certificate=True)
    assert cert.all()
    np.testing.assert_array_equal(pred, x @ u + 0.2 >= 0)
    np.testing.assert_array_equal(pred, expit(x @ w.T + b).mean(1) >= 0.5)


def test_many_uncertain_inputs_use_full_pool():
    rng = np.random.default_rng(32)
    x = rng.normal(size=(600, 4))
    w = rng.normal(size=(6, 4))
    b = rng.normal(size=6)
    router = compile_binary_pairs(w, b, [(0, 1), (2, 3), (4, 5)])
    labels, cert = router.predict(x, return_certificate=True)
    assert cert.any() and (~cert).any()
    np.testing.assert_array_equal(labels, expit(x @ w.T + b).mean(1) >= 0.5)
    assert router.predict(x[7]) == labels[7]


def test_ties_and_saturation():
    m, c = midpoint_certificate([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    np.testing.assert_array_equal(m, [1, 0, 1, 0])
    np.testing.assert_array_equal(c, [1, 1, 1, 0])

    def decide(b):
        return compile_binary_pairs(np.zeros((len(b), 1)), b, np.arange(len(b)).reshape(-1, 2)).predict(
            [1.0]
        )

    assert expit([999.0, -1000.0]).mean() == 0.5 and not decide([999.0, -1000.0])
    assert decide([1000.0, -1000.0])
    assert decide([-14, -8, 8, 16, 14, -16])
    # Both pairs saturate in probability space, the smaller negative pair wins.
    assert not decide([2000.0, -1999.0, 999.0, -1000.0])
    assert decide([999.0, -1000.0, 1000.0, -999.0])
    with pytest.raises(ValueError):
        compile_binary_pairs([[1], [2], [3], [4]], [0] * 4, [[0, 1], [2, 2]])
