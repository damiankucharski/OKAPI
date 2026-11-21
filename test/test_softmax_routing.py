import numpy as np
import pytest
from scipy.special import softmax

from okapi.softmax_routing import compile_softmax_router


def test_rank_bounds_and_both_label_certificates():
    rng = np.random.default_rng(112)
    k, c, d = 5, 4, 18
    base = rng.normal(size=(c, d))
    w = np.stack([base + 0.12 * rng.normal(size=(c, d)) for _ in range(k)])
    b = rng.normal(size=(k, c)) * 0.2
    a = np.array([0.4, 0.3, 0.1, 0.15, 0.05])
    x = rng.normal(size=(600, d))
    mu = np.linspace(-1, 1, d)
    sd = np.linspace(0.4, 1.3, d)
    logits = np.einsum("nd,kcd->nkc", x, w) + b
    residual = logits - np.einsum("k,nkc->nc", a, logits)[:, None]
    residual -= residual.mean(axis=-1, keepdims=True)
    energy = np.einsum("k,nkc,nkc->n", a, residual, residual)
    pool = np.einsum("k,nkc->nc", a, softmax(logits, axis=-1))
    previous = np.full(len(x), np.inf)
    for r in [0, 1, 2, 4, 8, None]:
        router = compile_softmax_router(w, b, a, center=mu, scale=sd, rank=r)
        bound = router.residual_energy_upper(x)
        assert np.all((energy - 1e-10 <= bound) & (bound <= previous + 1e-10))
        pred, by_tv, unanimous = router.certified(x)
        valid = by_tv | unanimous
        assert np.array_equal(pred[valid], pool[valid].argmax(1))
        if unanimous.any():
            assert np.all(logits.argmax(-1)[unanimous] == pred[unanimous, None])
        previous = bound
    assert router.rank <= (k - 1) * (c - 1)
    np.testing.assert_allclose(previous, energy, atol=1e-11)
    assert by_tv.any() and unanimous.any()


def test_zero_weight_and_ties():
    w = np.array([[[1.0], [1.0]], [[100.0], [-100.0]]])
    model = compile_softmax_router(w, np.zeros((2, 2)), [1.0, 0.0], center=[0.0], scale=[1.0], rank=2)
    cls, tv, unanimous = model.certified([4.0])
    assert cls == 0 and not tv and not unanimous and model.rank == 0
    with pytest.raises(ValueError):
        compile_softmax_router(w, np.zeros((2, 2)), [1.0, 0.0], center=[0.0], scale=[1.0], rank=-1)
