"""Regression checks for RNG isolation and model enumeration in seeded evolution."""
from pathlib import Path

import numpy as np
import pytest

from okapi.okapi import Okapi


@pytest.fixture(autouse=True)
def _restore_caller_rng():
    before = np.random.get_state()
    yield
    np.random.set_state(before)


def _dataset(workdir):
    data = workdir / "predictions"
    data.mkdir()
    rng = np.random.RandomState(711)
    for i in range(6):
        np.save(data / f"m{i}.npy", rng.rand(60, 1))
    gt = workdir / "labels.npy"
    np.save(gt, rng.randint(2, size=(60, 1)))
    return data, gt


def _engine(data, gt, seed):
    return Okapi(data, gt, 5, 3, 3, seed=seed, backend="numpy",
                 objective_functions=(lambda pred, truth: 1 - np.mean((pred - truth) ** 2),))


def _snapshot(engine):
    return [(repr(tree), fitness.tolist(), tree.predict().tobytes()) for tree, fitness in zip(engine.population, engine.fitnesses)]


def test_interleaving_seeded_searches_reproduces_each_generation(tmp_path):
    data, gt = _dataset(tmp_path)
    reference = _engine(data, gt, 7)
    expected = []
    for _ in range(8):
        reference.run_iteration()
        expected.append(_snapshot(reference))

    engine = _engine(data, gt, 7)
    other = _engine(data, gt, 91)
    for step in expected:
        other.run_iteration()
        np.random.random(31)
        engine.run_iteration()
        assert _snapshot(engine) == step


def test_seeded_search_preserves_numpy_random_state(tmp_path):
    data, gt = _dataset(tmp_path)
    before = np.random.get_state()
    _engine(data, gt, 7).run_iteration()
    np.testing.assert_equal(np.random.get_state(), before)


def test_directory_inputs_are_ordered_and_explicit_order_is_preserved(tmp_path, monkeypatch):
    data, gt = _dataset(tmp_path)
    glob = Path.glob

    def reversed_glob(path, pattern):
        return reversed(sorted(glob(path, pattern)))

    monkeypatch.setattr(Path, "glob", reversed_glob)
    paths = sorted(data.iterdir())
    engine = _engine(data, gt, 7)
    assert engine.ids == [p.name for p in paths]
    explicit = _engine(reversed(paths), gt, 7)
    assert explicit.ids == [p.name for p in reversed(paths)]
