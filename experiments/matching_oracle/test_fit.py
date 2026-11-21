import itertools
import math

import numpy as np
import pytest

from experiments.matching_oracle.fit import fit_matching, matching_tree
from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function


@pytest.fixture(autouse=True)
def numpy_backend(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)


def enumerate_matchings(vertices):
    if not vertices:
        yield ()
        return
    for j in vertices[1:]:
        tail = tuple(v for v in vertices[1:] if v != j)
        for remaining in enumerate_matchings(tail):
            yield ((vertices[0], j), *remaining)


@pytest.mark.parametrize("k,seed", [(4, 601), (6, 607)])
def test_dp_agrees_with_direct_exhaustive_brier_and_all_ties(k, seed):
    low, high = 1 / 64, 63 / 64
    rng = np.random.default_rng(seed)
    rows = np.array([rng.choice(2 * k, size=2, replace=False) for _ in range(29)])
    x = np.full((len(rows), 2 * k), low)
    x[np.arange(len(rows))[:, None], rows] = high
    y = rng.integers(2, size=len(x), dtype=np.uint8)
    fit = fit_matching(x, y)
    # Independent oracle computes every full program's probabilities on raw rows.
    risks = {}
    for matching in enumerate_matchings(tuple(range(2 * k))):
        left, right = np.array(matching).T
        scores = np.minimum(x[:, left], x[:, right]).max(axis=1)
        risks[matching] = float(np.mean((scores - y) ** 2))
    lowest = min(risks.values())
    winners = [matching for matching, loss in risks.items() if loss == lowest]
    assert fit.fit_brier == lowest and fit.pairs == min(winners)
    assert fit.n_optimal == len(winners) and len(risks) == math.prod(range(1, 2 * k, 2))
    assert 1 <= fit.n_states <= 2 ** (2 * k) and fit.transitions >= fit.n_states - 1
    tree = matching_tree(fit.pairs, x.T)
    assert len(tree.op_nodes) == k + 1
    assert np.mean((tree.predict() - y) ** 2) == lowest

    # On arbitrary new interior inputs the compiled program still performs max of pairwise mins.
    interior = rng.random((61, 2 * k))
    bound = tree.do_pred_on_another_tensors(current_tensors={i: interior[:, i] for i in range(2 * k)})
    left, right = np.array(fit.pairs).T
    np.testing.assert_array_equal(bound, np.minimum(interior[:, left], interior[:, right]).max(axis=1))


def test_all_scores_tied_and_endpoint_cases():
    edges = list(itertools.combinations(range(4), 2))
    x = np.zeros((len(edges), 4))
    x[np.arange(len(x))[:, None], edges] = 1
    fit = fit_matching(x, np.ones(len(x)), low=0, high=1)
    assert fit.n_optimal == 3 and fit.pairs == ((0, 1), (2, 3))
    assert fit.fit_brier == 4 / 6
    two = fit_matching([[1, 1], [1, 1]], [0, 1], low=0, high=1)
    assert two.pairs == ((0, 1),) and two.n_optimal == 1 and two.fit_brier == .5


@pytest.mark.parametrize("x,y", [
    ([], []), ([[0, 1, 0]], [0]), ([[1, 0, 0, 0]], [1]), ([[1, 1, 1, 0]], [0]),
    ([[1, 1, 0, .5]], [1]), ([[1, 1, 0, 0]], [2]), ([[1, 1, 0, 0]], [[1]]),
])
def test_scope_requires_binary_labels_and_exact_two_signal_rows(x, y):
    with pytest.raises(ValueError):
        fit_matching(x, y, low=0, high=1)
