from itertools import product

import numpy as np
import pytest

from okapi.lattice_search import small_lattice_programs
from okapi.monotone_envelope import fit_monotone_envelope


@pytest.mark.parametrize("dimension", [3, 4])
def test_pointwise_bounds_equal_enumeration_of_all_consistent_programs(dimension):
    cube = np.array(list(product([0, 1], repeat=dimension)), dtype=bool)
    programs = small_lattice_programs(dimension)
    outputs = np.array([np.any([np.all(cube[:, term], axis=1) for term in program], axis=0) for program in programs])
    assert len(np.unique(outputs, axis=0)) == len(programs)
    seen = np.array([1, 3, 5, 6])
    for labels in np.unique(outputs[:, seen], axis=0):
        envelope = fit_monotone_envelope(cube[seen], labels)
        compatible = outputs[(outputs[:, seen] == labels).all(axis=1)]
        lower, upper = envelope.bounds(cube)
        np.testing.assert_array_equal(lower, compatible.all(axis=0))
        np.testing.assert_array_equal(upper, compatible.any(axis=0))
        np.testing.assert_array_equal(envelope.predict_positive(cube), (lower.astype(float) + upper) / 2)


def test_large_dimension_reduces_to_two_observed_antichains():
    x = np.zeros((4, 70), dtype=bool)
    x[0, 69] = True
    x[1, [3, 69]] = True  # Positive superset of row 0.
    x[2, [4, 68]] = True
    x[3, 68] = True  # Negative subset of row 2.
    fit = fit_monotone_envelope(x, [1, 1, 0, 0])
    assert fit.minimal_positive == (1 << 69,)
    assert fit.maximal_negative == ((1 << 4) | (1 << 68),)
    query = np.zeros((3, 70))
    query[0, 4] = 1
    query[1, 69] = 1
    query[2, 67] = 1
    np.testing.assert_array_equal(fit.predict_positive(query), [0, 1, 0.5])


def test_empty_observations_and_absorbing_endpoint_labels():
    model = fit_monotone_envelope(np.empty((0, 3)), [])
    np.testing.assert_array_equal(model.predict_positive([[0, 0, 0], [0, 1, 0], [1, 1, 1]]), [0, 0.5, 1])
    lower, upper = model.bounds(np.empty((0, 3)))
    assert lower.shape == upper.shape == (0,)
    with pytest.raises(ValueError, match="3 input"):
        model.bounds([[0, 0]])


@pytest.mark.parametrize(("x", "y"), [([[0, 0]], [1]), ([[1, 1]], [0]), ([[0, 1], [0, 1]], [0, 1]), ([[1, 0, 0], [1, 0, 1]], [1, 0])])
def test_contradictions_are_rejected(x, y):
    with pytest.raises(ValueError, match="contradictory"):
        fit_monotone_envelope(x, y)


@pytest.mark.parametrize(("x", "y"), [([0, 1], [1]), (np.empty((4, 0)), [1] * 4), ([[0.2]], [0]), ([[0]], [2]), ([[1]], []), ([[np.nan]], [1])])
def test_requires_binary_aligned_inputs(x, y):
    with pytest.raises(ValueError, match="binary"):
        fit_monotone_envelope(x, y)


def test_public_programs_are_immutable_and_ordered_by_simplicity():
    programs = small_lattice_programs(np.int64(3))
    assert isinstance(programs, tuple)
    assert programs == tuple(sorted(programs, key=lambda terms: (sum(map(len, terms)), terms)))
    assert [len(small_lattice_programs(k)) for k in [1, 2, 3, 4]] == [1, 4, 18, 166]
    for invalid in [0, 5, True, 2.5]:
        with pytest.raises(ValueError, match="integer"):
            small_lattice_programs(invalid)
