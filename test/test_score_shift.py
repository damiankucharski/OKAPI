import numpy as np
import pytest

from okapi.score_shift import ReferenceScoreCDF


def test_exact_location_and_mixture():
    rng = np.random.default_rng(23)
    neg = rng.normal(-2, 0.5, 100)
    pos = rng.normal(2, 0.8, 100)
    model = ReferenceScoreCDF(neg, pos)
    delta = 0.4 * model.scale
    target = np.concatenate([neg, pos, pos, pos]) + delta
    fit = model.fit(target)
    assert fit.offset == pytest.approx(delta)
    assert fit.prior == pytest.approx(0.75)
    assert fit.cdf_rmse < 1e-12
    assert model.fit(target, allow_offset=False).offset == 0
    assert model.prior_from_mean(target) > 0.75


def test_indistinguishable_classes_have_neutral_prior():
    values = np.arange(30.0)
    ref = ReferenceScoreCDF(values, values)
    fit = ref.fit(values)
    assert (fit.prior, fit.offset) == (0.5, 0)
    assert ref.prior_from_mean(values) == 0.5
    with pytest.raises(ValueError):
        ref.fit([np.nan])
    with pytest.raises(ValueError):
        ReferenceScoreCDF([1], [1])
