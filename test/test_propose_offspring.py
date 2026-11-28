import numpy as np
import pytest

from okapi import Okapi
from okapi.globals import _passthrough, use_postprocessing_function
from okapi.random_state import use_rng


@pytest.fixture
def building_blocks(tmp_path):
    rng = np.random.default_rng(150)
    folder = tmp_path / "providers"
    folder.mkdir()
    label_path = tmp_path / "labels.npy"
    np.save(label_path, rng.integers(2, size=(20, 1)))
    for i in range(4):
        np.save(folder / f"{i}.npy", rng.random((20, 1)))
    return folder, label_path


def solver(paths):
    def brier(p, y):
        return 1 - ((p - y) ** 2).mean()

    gp = Okapi(*paths, 4, 3, 2, backend="numpy", objective_functions=(brier,), max_depth=5, max_nodes=7, seed=2026)
    gp.fitnesses = gp._selection_fitnesses(gp.population)
    return gp


def test_matched_native_variation_and_rng_without_search_side_effects(building_blocks, monkeypatch):
    with use_postprocessing_function(_passthrough):
        gp = solver(building_blocks)
        stock = solver(building_blocks)
        originals = tuple(gp.population)
        fit_scores = gp.fitnesses.copy()
        monkeypatch.setattr(gp, "_calculate_fitnesses", lambda *_: pytest.fail("Proposal should not score a tree"))
        reference = []
        mutate_one = stock._mutate_offspring

        def capture(tree):
            result = mutate_one(tree)
            reference.append(result[0].structural_key())
            return result

        monkeypatch.setattr(stock, "_mutate_offspring", capture)
        for _ in range(3):
            with use_rng(stock.rng):
                stock._perform_crossovers(stock.fitnesses)
                stock._mutate_additional_population()
                stock.additional_population = []
            proposed = gp.propose_offspring(12)
            assert [t.structural_key() for t, _parents in proposed] == reference
            reference.clear()
            assert all(0 <= i < 4 for entry in proposed for i in entry.parent_indices)
            assert all(gp._within_caps(entry.tree) for entry in proposed)
        assert tuple(gp.population) == originals and not gp.additional_population
        np.testing.assert_equal(gp.fitnesses, fit_scores)
        np.testing.assert_equal(gp.rng.get_state(), stock.rng.get_state())


def test_one_parent_odd_count_and_duplicate_slots(building_blocks):
    gp = solver(building_blocks)
    gp.population = gp.population[:1]
    gp.fitnesses = gp.fitnesses[:1]
    gp.max_nodes = 1
    proposed = gp.propose_offspring(5)
    assert len(proposed) == 5
    assert all(p.parent_indices == (0, 0) for p in proposed)
    assert all(p.tree.structural_key() == gp.population[0].structural_key() and p.tree is not gp.population[0] for p in proposed)
    assert len({id(p.tree) for p in proposed}) == 5
    assert gp.propose_offspring(0) == []
    with pytest.raises(ValueError, match="integer"):
        gp.propose_offspring(-1)
    gp.fitnesses = None
    with pytest.raises(ValueError, match="breeding"):
        gp.propose_offspring(2)
