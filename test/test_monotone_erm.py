from itertools import product

import numpy as np
import pytest

from okapi.lattice_search import small_lattice_programs
from okapi.monotone_erm import fit_monotone_erm


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_extremal_rules_match_all_exhaustive_optima_including_noisy_repeats(dimension):
    cube = np.asarray(list(product((False, True), repeat=dimension)))
    rules = np.array([np.any([cube[:, term].all(axis=1) for term in p], axis=0) for p in small_lattice_programs(dimension)])
    rng = np.random.default_rng(53 + dimension)
    for n in [0, 1, 5, 23, 45]:
        indices = rng.integers(len(cube), size=n)
        labels = rng.integers(2, size=n)
        errors = (rules[:, indices] != labels).sum(axis=1)
        minimizers = rules[errors == errors.min()]
        fit = fit_monotone_erm(cube[indices], labels)
        lower, upper = fit.bounds(cube)
        assert fit.min_errors == errors.min()
        np.testing.assert_array_equal(lower, minimizers.all(axis=0))
        np.testing.assert_array_equal(upper, minimizers.any(axis=0))
        assert (minimizers == lower).all(axis=1).any()
        assert (minimizers == upper).all(axis=1).any()
        np.testing.assert_array_equal(fit.predict_positive(cube), 0.5 * (lower.astype(float) + upper))


def test_optimal_tie_requires_jointly_monotone_choices():
    x = [[0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1]]
    model = fit_monotone_erm(x, [1, 0, 1, 0])
    assert model.min_errors == 3
    np.testing.assert_array_equal(model.predict_positive(x), [0.5, 0.5, 0, 1])


def test_empty_dataset_and_maximum_dimension():
    fit = fit_monotone_erm(np.empty((0, 12)), [])
    assert fit.min_errors == 0
    query = np.array([[0] * 12, [0] * 11 + [1], [1] * 12])
    np.testing.assert_array_equal(fit.predict_positive(query), [0, 0.5, 1])
    lower, upper = fit.bounds(np.empty((0, 12)))
    assert lower.shape == upper.shape == (0,)
    with pytest.raises(ValueError, match="12 input"):
        fit.bounds([[0]])
    with pytest.raises(ValueError, match="12"):
        fit_monotone_erm(np.zeros((1, 13)), [1])


@pytest.mark.parametrize(("x", "y"), [([0, 1], [1]), ([[0.3, 1.0]], [1]), ([[0, 1]], []), ([[0, 1]], [[0]]), ([[1, 1]], [2])])
def test_invalid_inputs(x, y):
    with pytest.raises(ValueError, match="binary"):
        fit_monotone_erm(x, y)
