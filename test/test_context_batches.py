import numpy as np
import pytest

from okapi.selection import estimate_context_prevalence, estimate_context_prevalences


def test_batched_target_means_match_individual_scalar_projections():
    rng = np.random.default_rng(44)
    a, b = rng.normal(size=(2,32,4))
    means = rng.normal(size=(7,3,4))
    expected = np.array([[estimate_context_prevalence(row,a,b,.2) for row in reps] for reps in means])
    np.testing.assert_allclose(estimate_context_prevalences(means,a,b,.2),expected)
    np.testing.assert_allclose(estimate_context_prevalences(means,a,a,.2),np.full((7,3),.2))
    with pytest.raises(ValueError):
        estimate_context_prevalence(means,a,b,.2)
