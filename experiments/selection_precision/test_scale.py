"""Counterexamples for rounding native selection fitness at three decimals."""

import numpy as np
import pytest

from okapi import Okapi
from okapi.crossover import tournament_selection_indexes
from okapi.globals import use_postprocessing_function
from okapi.node import MeanNode, ValueNode
from okapi.pareto import maximize
from okapi.population import choose_pareto_then_proximity
from okapi.random_state import use_rng
from okapi.tree import Tree


class FixedOffspring(Okapi):
    """Supply one predetermined child; retain native assessment and survival."""

    def _perform_crossovers(self, fitnesses):
        self.additional_population = [self.candidate]
        return 0

    def _mutate_additional_population(self):
        return 0


def winner(candidates, scores):
    selected, _ = choose_pareto_then_proximity(candidates, scores, (maximize,), 1, minimize_node_count=True)
    return selected[0]


@pytest.mark.parametrize('reuse', [False, True])
@pytest.mark.parametrize(('b_confidence', 'uncompressed_native_winner'), [(.9, 'mean'), (.7012, 'singleton')])
def test_real_survival_quantizes_single_objective_and_can_prefer_the_higher_loss(tmp_path, reuse, b_confidence, uncompressed_native_winner):
    labels = np.array([[0], [1], [0], [1]])
    a, b = (np.where(labels == 1, confidence, 1.0 - confidence) for confidence in (.7, b_confidence))
    singleton = Tree(ValueNode([], a, 'a'))
    mean = Tree(ValueNode([MeanNode([ValueNode([], b, 'b')])], a, 'a'))
    candidates = [singleton, mean]
    np.save(tmp_path / 'a.npy', a)
    np.save(tmp_path / 'y.npy', labels)

    def run(compress):
        def score(p, y):
            fitness = 1.0 - np.mean((p - y) ** 2)
            return .75 + .0001 * fitness if compress else fitness

        engine = FixedOffspring(
            [tmp_path / 'a.npy'], tmp_path / 'y.npy', 1, 1, 1, backend='numpy', seed=1,
            objective_functions=(score,), objectives=(maximize,), minimize_node_count=True,
            reuse_survivor_fitness=reuse,
        )
        engine.population, engine.candidate = [singleton], mean
        with use_postprocessing_function(lambda p: p):
            raw = engine._calculate_fitnesses(candidates)
            engine.run_iteration()
        picked = engine.population[0]
        assert winner(candidates, raw) is mean
        assert winner(candidates, raw.round(3)) is picked
        return picked, raw

    original, raw = run(False)
    compressed, scaled = run(True)
    assert raw[0, 0] < raw[1, 0] and scaled[0, 0] < scaled[1, 0]
    assert original is {'mean': mean, 'singleton': singleton}[uncompressed_native_winner]
    assert compressed is singleton
    assert np.all(scaled.round(3) == .750)
    if b_confidence < .71:
        assert np.all(raw.round(3) == .910)

    # A two-draw native tournament on the same two candidates: both are always compared.
    def tournament(scores, seed):
        with use_rng(np.random.RandomState(seed)):
            return tuple(tournament_selection_indexes(scores, 2))

    assert all(tournament(raw, seed) == tournament(scaled, seed) == (1, 1) for seed in range(64))
    tied = [index for seed in range(64) for index in tournament(scaled.round(3), seed)]
    assert set(tied) == {0, 1}


def test_multiple_objectives_have_scale_sensitive_distance_even_without_rounding():
    # Two nondominated, equal-sized trees. Compress only the second coordinate toward its ideal.
    a, b = (Tree(ValueNode([], np.zeros((1, 1)), i)) for i in range(2))
    raw = np.array([[.6, 1.0], [1.0, .5]])
    scaled = raw.copy()
    scaled[:, 1] = .9 + .1 * scaled[:, 1]
    for scores, expected in ((raw, a), (scaled, b)):
        picked, _ = choose_pareto_then_proximity([a, b], scores, (maximize, maximize), 1)
        assert picked == [expected]
