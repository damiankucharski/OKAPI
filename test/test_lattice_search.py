from itertools import product

import numpy as np
import pytest

from okapi.lattice_search import fit_small_lattice


@pytest.mark.parametrize(("n_inputs", "count"), [(1, 1), (2, 4), (3, 18), (4, 166)])
def test_all_nonconstant_monotone_functions_and_clone_tie_break(n_inputs, count):
    scores = np.full((4, n_inputs), 0.5)
    program = fit_small_lattice(scores, [0, 0, 1, 1])
    assert program.n_candidates == count
    assert program.terms == ((0,),)


def test_majority_lifts_to_continuous_median():
    scores = np.array(list(product([0.05, 0.95], repeat=3)))
    program = fit_small_lattice(scores, np.sum(scores > 0.5, axis=1) >= 2)
    assert program.terms == ((0, 1), (0, 2), (1, 2))
    rng = np.random.default_rng(37)
    query = rng.uniform(size=(51, 3))
    np.testing.assert_array_equal(program.predict(query), np.median(query, axis=1))


@pytest.mark.parametrize(
    ("scores", "labels"),
    [([0.3], [0]), ([[1.1]], [0]), ([[np.nan]], [1]), (np.zeros((1, 5)), [0]), ([[0.3], [0.7]], [0]), ([[0.4]], [2]), (np.empty((0, 2)), [])],
)
def test_invalid_fitting_data(scores, labels):
    with pytest.raises(ValueError):
        fit_small_lattice(scores, labels)


def test_prediction_validates_fitted_column_count():
    program = fit_small_lattice([[0.2], [0.8]], [0, 1])
    with pytest.raises(ValueError, match="expected 1 probability columns"):
        program.predict([[0, 1]])
