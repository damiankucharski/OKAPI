from itertools import product

import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function
from okapi.guided import correlation_templates, seed_from_correlations
from okapi.okapi import Okapi
from okapi.operators import MAX, MIN
from okapi.pareto import minimize


def groups(tree):
    return 1 + sum(isinstance(op, (MIN, MAX)) for op in tree.op_nodes)


def first_outputs(trees, target):
    first = {}
    for tree in trees:
        scores = tree.do_pred_on_another_tensors(current_tensors=target)
        first.setdefault((groups(tree), scores.tobytes()), repr(tree))
    return list(first.values())


@pytest.mark.parametrize("shuffle_seed", [None, 23])
def test_first_representatives_match_boolean_cube_and_continuous_scores(shuffle_seed, monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)
    rng = np.random.default_rng(74)
    fit = {i: rng.random(100) for i in range(4)}
    full = correlation_templates(fit, shuffle_seed=shuffle_seed)
    compact = correlation_templates(fit, shuffle_seed=shuffle_seed, deduplicate=True)
    assert len(full) == 135 and len(compact) == 63
    assert [sum(groups(t) == size for t in compact) for size in range(1, 5)] == [1, 2, 8, 52]
    cube = np.array(list(product((0., 1.), repeat=4)))
    for x in (cube, rng.random((1024, 4))):
        target = {i: x[:, i] for i in range(4)}
        assert first_outputs(full, target) == [repr(t) for t in compact]
        assert first_outputs(compact, target) == first_outputs(full, target)


def test_capped_initializer_keeps_first_feasible_member(tmp_path, monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)
    rng = np.random.default_rng(61)
    paths = []
    for i in range(4):
        path = tmp_path / f"p{i}.npy"
        np.save(path, rng.random((512, 1)))
        paths.append(path)
    yfile = tmp_path / "y.npy"
    np.save(yfile, rng.integers(2, size=512))
    captured = []

    def objective(p, y):
        captured.append(p.tobytes())
        return np.mean((p-y)**2)

    def run(deduplicate):
        gp = Okapi(paths, yfile, 4, 2, 2, max_depth=5, backend="numpy", seed=3,
                   objective_functions=(objective,), objectives=(minimize,))
        captured.clear()
        calls = seed_from_correlations(gp, deduplicate=deduplicate)
        assert calls == len(captured)
        return captured.copy()

    full, compact = run(False), run(True)
    assert len(compact) < len(full)
    assert list(dict.fromkeys(full)) == list(dict.fromkeys(compact))
