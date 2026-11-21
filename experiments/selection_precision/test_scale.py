"""Counterexamples for rounding native selection fitness at three decimals."""

import numpy as np
import pytest

from okapi import Okapi
from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.crossover import tournament_selection_indexes
from okapi.globals import use_postprocessing_function
from okapi.node import MaxNode, MeanNode, ValueNode
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

    def run(compress, **options):
        def score(p, y):
            fitness = 1.0 - np.mean((p - y) ** 2)
            return .75 + .0001 * fitness if compress else fitness

        engine = FixedOffspring(
            [tmp_path / 'a.npy'], tmp_path / 'y.npy', 1, 1, 1, backend='numpy', seed=1,
            objective_functions=(score,), objectives=(maximize,), minimize_node_count=True,
            reuse_survivor_fitness=reuse, **options,
        )
        engine.population, engine.candidate = [singleton], mean
        with use_postprocessing_function(lambda p: p):
            raw = engine._calculate_fitnesses(candidates)
            engine.run_iteration()
        picked = engine.population[0]
        assert winner(candidates, raw) is mean
        decimals = engine.selection_fitness_decimals
        assert winner(candidates, raw if decimals is None else raw.round(decimals)) is picked
        return picked, raw

    original, raw = run(False)
    compressed, scaled = run(True)
    assert raw[0, 0] < raw[1, 0] and scaled[0, 0] < scaled[1, 0]
    assert original is {'mean': mean, 'singleton': singleton}[uncompressed_native_winner]
    assert compressed is singleton
    assert np.all(scaled.round(3) == .750)
    if b_confidence < .71:
        assert np.all(raw.round(3) == .910)
    assert run(False, selection_fitness_decimals=3)[0] is original
    assert run(True, selection_fitness_decimals=3)[0] is compressed
    assert run(False, selection_fitness_decimals=None)[0] is mean
    assert run(True, selection_fitness_decimals=None)[0] is mean
    assert run(False, selection_fitness_decimals=6)[0] is mean
    assert run(True, selection_fitness_decimals=6)[0] is (singleton if b_confidence < .71 else mean)

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


def test_distinct_float_scores_can_have_the_same_float_distance_to_the_ideal():
    raw = np.array([[0.], [1e-20]])
    assert raw[0, 0] < raw[1, 0]
    assert 1 - raw[0, 0] == 1 - raw[1, 0]
    with use_rng(np.random.RandomState(1)):
        assert tuple(tournament_selection_indexes(raw, 2)) == (0, 0)


def test_nominal_tournament_can_choose_a_parent_whose_same_extension_misses_the_target(monkeypatch):
    monkeypatch.setattr(Backend, '_current_backend', NumpyBackend)
    y = np.array([0., 1., 1.])
    a = np.array([0., .5 - 2**-12, 1.])
    b = np.array([0., 1., .5])
    e = np.array([0., 0., 1.])
    raw = np.array([[1 - np.mean((v - y)**2)] for v in (a, b)])
    assert raw[0, 0] < raw[1, 0] and np.all(raw.round(3) == .917)
    for scores, index in ((raw.round(3), 0), (raw.round(6), 1), (raw, 1)):
        with use_rng(np.random.RandomState(1)):
            parent = tournament_selection_indexes(scores, 2)[0]
        assert parent == index
        child = Tree(ValueNode([MaxNode([ValueNode([], e, 'e')])], (a, b)[parent], int(parent)))
        with use_postprocessing_function(lambda p: p):
            loss = np.mean((child.predict() - y)**2)
        assert child.nodes_count == 3
        assert min(1 - raw.max(), loss) == pytest.approx(1 / 12 if parent == 0 else 0)


def test_raw_fitness_can_resolve_accumulation_order_of_the_same_mean(monkeypatch):
    monkeypatch.setattr(Backend, '_current_backend', NumpyBackend)

    def score(order):
        nodes = [ValueNode([], np.array([v]), i) for i, v in enumerate(order)]
        nodes[0].add_child(MeanNode(nodes[1:]))
        with use_postprocessing_function(lambda p: p):
            p = Tree(nodes[0]).predict().item()
        return 1 - (1 - p)**2

    first, second = score([.1, .1, .4]), score([.1, .4, .1])
    assert 0 < abs(first - second) < 3e-16
    assert round(first, 6) == round(second, 6) == .36


def test_an_exact_fit_improvement_can_raise_conditional_brier():
    # q=.5 makes the two labels equally likely. Evaluate all possible first tie winners.
    epsilon = 2**-12
    predictions = np.array([.5 - epsilon, .5, .5 + epsilon])
    excess = (predictions - .5)**2
    for label in (0, 1):
        fitness = 1 - (predictions - label)**2
        assert np.all(fitness.round(3) == .750)
        assert excess[np.argmax(fitness)] == epsilon**2
        assert excess.mean() == 2 * epsilon**2 / 3


@pytest.mark.parametrize('bad', [-1, 16, 3.0, True, '3'])
def test_invalid_precision_is_rejected_before_reading_any_input(bad):
    with pytest.raises(ValueError, match='selection_fitness_decimals'):
        Okapi([], '', 1, 1, 1, selection_fitness_decimals=bad)
