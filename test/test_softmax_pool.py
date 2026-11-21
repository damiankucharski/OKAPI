import numpy as np
import pytest
from scipy.special import logit, softmax

from okapi.softmax_pool import compile_softmax_pool


def test_tv_gauge_invariance_and_pointwise_certificate():
    rng = np.random.default_rng(93)
    w = rng.normal(size=(5, 4, 3))
    b = rng.normal(size=(5, 4))
    a = np.array([0.1, 0.3, 0, 0.22, 0.38])
    x = rng.normal(size=(250, 3))
    m = compile_softmax_pool(w, b, a)
    p = np.einsum("k,nkc->nc", a, softmax(np.einsum("nd,kcd->nkc", x, w) + b, axis=2))
    tv = np.abs(p - m.predict(x)).sum(1) / 2
    assert np.all(tv <= m.oscillation_bound(x) + 1e-12)
    assert np.all(m.oscillation_bound(x) <= m.error_bound(x) + 1e-12)
    shifted = compile_softmax_pool(w + rng.normal(size=(5, 1, 3)), b + rng.normal(size=(5, 1)), a)
    np.testing.assert_allclose(shifted.predict(x), m.predict(x), atol=1e-14)
    np.testing.assert_allclose(shifted.error_bound(x), m.error_bound(x), atol=1e-14)
    np.testing.assert_allclose(shifted.oscillation_bound(x), m.oscillation_bound(x), atol=1e-14)


def test_tv_constant_is_approached_for_any_two_equal_groups():
    for n in [1, 3, 17]:
        p = (3 - np.sqrt(3)) / 6
        base = np.r_[np.full(n, logit(p)), np.zeros(n)]
        direction = np.r_[np.ones(n), np.zeros(n)]
        eps = 0.001
        logits = np.stack([base + eps * direction, base - eps * direction])
        m = compile_softmax_pool(np.zeros((2, 2 * n, 1)), logits, [0.5, 0.5])
        prediction = m.predict([0])
        actual = softmax(logits, axis=1).mean(0)
        tv = np.abs(actual - prediction).sum() / 2
        assert 0.9999 < tv / m.oscillation_bound([0]) < 1.000001


def test_compact_export_and_identical_functions():
    w = np.array([[[1.0], [-1.0], [0.0]], [[1.0], [-1.0], [0.0]]])
    b = np.zeros((2, 3))
    m = compile_softmax_pool(w, b, [0.5, 0.5], keep_sources=False)
    assert m.centered_residuals is None and m.error_bound([2]) == 0
    with pytest.raises(ValueError):
        m.oscillation_bound([2])
