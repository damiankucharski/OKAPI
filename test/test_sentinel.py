import numpy as np
import pytest
from okapi.sentinel import upper_tail_pvalues


def test_randomized_ranks_include_held_out_observation_and_ties():
    p = upper_tail_pvalues([1, 1, 3], [1, 3, 4, 0], [0.5] * 4)
    np.testing.assert_allclose(p, [0.625, 0.25, 0.125, 0.875])
    u = np.array([0.03, 0.05, 0.11, 0.97])
    np.testing.assert_allclose(upper_tail_pvalues([1] * 7, [1] * 4, u), u)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        upper_tail_pvalues([], [0.5], [0.2])
    with pytest.raises(ValueError):
        upper_tail_pvalues([1, 2], [0.5], [1])
