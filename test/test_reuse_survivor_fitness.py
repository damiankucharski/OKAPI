import numpy as np
import pytest

from okapi import Okapi
from okapi.callback import Callback
from okapi.pareto import maximize, minimize


@pytest.fixture
def inputs(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    labels = tmp_path / "labels.npy"
    rng = np.random.default_rng(72)
    for provider in range(5):
        np.save(predictions / f"{provider}.npy", rng.uniform(size=(60, 1)))
    np.save(labels, rng.integers(2, size=(60, 1)))
    return predictions, labels


def engine(inputs, **kwargs):
    calls = []

    def brier(predictions, labels):
        calls.append(1)
        return np.mean((predictions - labels) ** 2)

    def accuracy(predictions, labels):
        return np.mean((predictions >= 0.5) == labels)

    search = Okapi(*inputs, 5, 2, 2, seed=42, backend="numpy",
                   objective_functions=(brier, accuracy), objectives=(minimize, maximize), **kwargs)
    return search, calls


def test_reuse_matches_each_seeded_generation(inputs):
    normal, reference_calls = engine(inputs)
    cached, cached_calls = engine(inputs, reuse_survivor_fitness=True)
    saved_calls = 0
    for _ in range(8):
        saved_calls += len(cached.population)
        normal.run_iteration()
        cached.run_iteration()
        np.testing.assert_array_equal(cached.fitnesses, normal.fitnesses)
        assert [repr(t) for t in cached.population] == [repr(t) for t in normal.population]
        for actual, expected in zip(cached.population, normal.population, strict=True):
            np.testing.assert_array_equal(actual.predict(), expected.predict())
        assert len(reference_calls) - len(cached_calls) == saved_calls


def test_duplicate_parent_keeps_first_row_and_scores_no_duplicate_offspring(inputs):
    search, calls = engine(inputs, reuse_survivor_fitness=True, max_nodes=1)
    # Duplicate the display code at a nonadjacent position with different input.
    # The existing deduplication rule keeps the first tree for each code.
    search.population[3].root.id = search.population[0].root.id
    search.run_iteration()
    assert len(calls) == 5
    assert len(search.population) == 4
    for tree, stored in zip(search.population, search.fitnesses, strict=True):
        actual = [obj(tree.predict(), search.gt_tensor) for obj in search.objective_functions]
        np.testing.assert_array_equal(stored, np.round(actual, 3))


def test_callback_can_invalidate_scores_after_label_change(inputs):
    class ChangeLabels(Callback):
        def on_generation_start(self, okapi):
            if okapi.fitnesses is not None:
                okapi.gt_tensor = 1 - okapi.gt_tensor
                okapi.fitnesses = None

    normal, _ = engine(inputs, callbacks=[ChangeLabels()])
    cached, _ = engine(inputs, callbacks=[ChangeLabels()], reuse_survivor_fitness=True)
    normal.train(3)
    cached.train(3)
    np.testing.assert_array_equal(cached.fitnesses, normal.fitnesses)
    assert [repr(t) for t in cached.population] == [repr(t) for t in normal.population]
