from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from okapi import Okapi
from okapi.globals import postprocessing_function
from okapi.islands import migrate_ring
from okapi.node import ValueNode
from okapi.operators import MAX, MEAN
from okapi.pareto import maximize, minimize
from okapi.tree import Tree


@pytest.fixture
def make_island(tmp_path, monkeypatch):
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda scores: scores)

    def make(name, errors, residents, *, rows=6, tradeoff=False, **kwargs):
        path = tmp_path / name
        path.mkdir()
        labels = np.arange(rows)[:, None] % 2
        gt = path / "labels.npy"
        np.save(gt, labels)
        providers = path / "predictions"
        providers.mkdir()
        for provider, error in errors.items():
            np.save(providers / f"{provider}.npy", np.where(labels, 1 - error, error))
        calls = []

        def brier(scores, truth):
            assert scores.shape == truth.shape
            calls.append(scores.copy())
            return np.mean((scores - truth) ** 2)

        functions = (brier, lambda scores, truth: np.mean(np.abs(scores - truth))) if tradeoff else (brier,)
        directions = (minimize, maximize) if tradeoff else (minimize,)
        engine = Okapi(providers, gt, len(residents), 1, 1, backend="numpy",
                       objective_functions=functions, objectives=directions, **kwargs)
        population = []
        for ids in residents:
            values = [ValueNode(None, engine.train_tensors[f"{p}.npy"], f"{p}.npy") for p in ids]
            if len(values) > 1:
                values[0].add_child(MEAN(values[1:]))
            population.append(Tree(values[0], mutation_chance=0.63))
        engine.population = population
        engine.fitnesses = engine._selection_fitnesses(population)
        calls.clear()
        return engine, calls

    return make


def test_elites_move_one_hop_and_score_once_on_receiver_data(make_island):
    islands, counters, elites = [], [], []
    ids = ("a", "b", "c")
    for i, own in enumerate(ids):
        # The arriving elite will be the receiver's new best, exposing a
        # cascade if export selection happens after one receiver was updated.
        errors = dict.fromkeys(ids, 0.8)
        errors.update({own: 0.2, ids[i - 1]: 0.04, "common": 0.25, "bad": 0.75})
        engine, calls = make_island(own, errors, [("bad",), (own, "common")], rows=5 + i)
        islands.append(engine)
        counters.append(calls)
        elites.append(engine.population[1])

    results = migrate_ring(islands)
    assert isinstance(results, tuple) and len(results) == 3
    for i, result in enumerate(results):
        source = (i - 1) % len(islands)
        assert (result.source, result.receiver, result.evaluated, result.admitted, result.reason) == (source, i, True, True, "admitted")
        receiver = islands[i]
        assert len(counters[i]) == 1
        expected = np.mean([receiver.train_tensors[f"{p}.npy"] for p in (ids[source], "common")], axis=0)
        np.testing.assert_array_equal(counters[i][0], expected)
        j = next(j for j, tree in enumerate(receiver.population) if repr(tree) == repr(elites[source]))
        immigrant = receiver.population[j]
        assert immigrant is not elites[source]
        assert immigrant.mutation_chance == elites[source].mutation_chance
        for local, original in zip(immigrant.value_nodes, elites[source].value_nodes, strict=True):
            assert local is not original
            assert local.value is receiver.train_tensors[local.id]
            assert original.value is islands[source].train_tensors[original.id]
        np.testing.assert_array_equal(receiver.fitnesses[j], [np.round(np.mean((expected - receiver.gt_tensor) ** 2), 3)])


def test_mixed_objectives_export_closest_elite(make_island):
    source, _ = make_island("source", {"a": 0.1, "b": 0.3}, [("a",), ("b",)], tradeoff=True)
    receiver, _ = make_island("receiver", {"a": 0.9, "b": 0.1, "c": 0.7}, [("c",)])
    # A wins on loss. B's greater second score brings it closer to the ideal
    # (loss=0, second score=1), despite its higher loss.
    np.testing.assert_array_equal(source.fitnesses, [[0.01, 0.1], [0.09, 0.3]])
    result = migrate_ring([source, receiver])[1]
    assert result.admitted and receiver.population[0].root.id == "b.npy"
    with pytest.raises(FrozenInstanceError):
        result.admitted = False


@pytest.mark.parametrize('decimals', [3, 6, None])
def test_migrating_elite_uses_receiver_selection_precision(make_island, decimals):
    source, _ = make_island('source', {'a': .2, 'b': .3}, [('a',)])
    receiver, _ = make_island('receiver', {'a': .1294573, 'b': .2}, [('b',)], selection_fitness_decimals=decimals)
    assert migrate_ring([source, receiver])[1].admitted
    raw = receiver._calculate_fitnesses(receiver.population)
    np.testing.assert_array_equal(receiver.fitnesses, raw if decimals is None else raw.round(decimals))


@pytest.mark.parametrize("reason", ["duplicate", "provider", "operator", "caps", "survival"])
def test_rejections_and_receiver_budget(make_island, reason):
    pair = ("a", "b")
    source, _ = make_island("sender", {"a": 0.1, "b": 0.1, "c": 0.6}, [pair])
    receiver_errors = {"a": 0.8, "b": 0.8, "c": 0.05}
    if reason == "provider":
        del receiver_errors["b"]
    residents = [pair] if reason == "duplicate" else [("c",)]
    options = {"operator": {"allowed_ops": (MAX,)}, "caps": {"max_nodes": 1}}.get(reason, {})
    receiver, calls = make_island("receiver", receiver_errors, residents, **options)
    before = tuple(receiver.population)
    stored_scores = receiver.fitnesses.copy()
    result = migrate_ring([source, receiver])[1]
    assert (result.reason, result.admitted, result.evaluated) == (reason, False, reason == "survival")
    assert len(calls) == int(result.evaluated)
    assert tuple(receiver.population) == before
    np.testing.assert_array_equal(receiver.fitnesses, stored_scores)


def test_failed_evaluation_cannot_partially_update_ring(make_island):
    ids = ("a", "b", "c")
    islands = [make_island(p, {p: 0.6, ids[i - 1]: 0.1}, [(p,)])[0] for i, p in enumerate(ids)]
    before = [(tuple(e.population), e.fitnesses.copy()) for e in islands]

    def fail(scores, labels):
        raise RuntimeError("receiver objective failed")

    islands[-1].objective_functions = (fail,)
    with pytest.raises(RuntimeError, match="receiver objective"):
        migrate_ring(islands)
    for engine, (population, scores) in zip(islands, before, strict=True):
        assert tuple(engine.population) == population
        np.testing.assert_array_equal(engine.fitnesses, scores)


def test_invalid_ring_requires_distinct_evaluated_engines(make_island):
    first, _ = make_island("first", {"a": 0.4}, [("a",)])
    other, _ = make_island("other", {"a": 0.4}, [("a",)])
    for invalid in ([], [first], [first, first]):
        with pytest.raises(ValueError):
            migrate_ring(invalid)
    for invalid_scores in (None, np.zeros((0, 1)), np.zeros((1, 2)), np.array([[np.nan]])):
        other.fitnesses = invalid_scores
        with pytest.raises(ValueError):
            migrate_ring([first, other])
