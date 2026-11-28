import numpy as np
import pytest

from okapi.globals import _passthrough, set_backend, use_postprocessing_function
from okapi.mean_guard import select_against_mean, select_tree_against_mean
from okapi.node import ValueNode
from okapi.tree import Tree


def test_guard_matches_independent_scalar_shared_sign_calculation():
    rng = np.random.default_rng(21)
    b = rng.random(11)
    losses = rng.random((4, 11))
    r = select_against_mean(b, losses, draws=39, seed=4, alpha=.2)
    gen = np.random.Generator(np.random.PCG64(4))
    gains = b - losses
    maxima = []
    for _ in range(39):
        signs = [1. if gen.random() >= .5 else -1. for _ in b]
        means = [sum((g - g.mean()) * signs) / np.sqrt(11 * 10) for g in gains]
        maxima.append(max(0., *means))
    threshold = sorted(maxima)[int(np.ceil(.8 * (39 - 1)))]
    assert r.threshold == pytest.approx(threshold)
    assert r.best_index == int(gains.mean(1).argmax())
    np.testing.assert_allclose(r.gains, gains.mean(1))
    assert r.index == (r.best_index if max(r.gains) > threshold else None)


def test_multiple_contestants_can_stop_a_noisy_winner_and_duplicates_do_not_count_twice():
    n = np.arange(64)
    gains = np.array([.24 + (2 * ((n >> bit) % 2) - 1) for bit in range(6)])
    b = np.full(len(n), 2.)
    one = select_against_mean(b, b - gains[:1])
    several = select_against_mean(b, b - gains)
    copies = select_against_mean(b, b - gains[[0, 0, 0]])
    assert one.index == 0 and copies.index == 0
    assert copies.threshold == one.threshold
    assert one.threshold < .24 < several.threshold
    assert several.index is None and several.reason == "below_threshold"
    assert select_against_mean(b, [b]).index is None


def test_groups_define_effective_sample_size_and_group_average_gain():
    groups = np.repeat(["x", "a", "b", "f", "t", "e", "s", "r"], [1, 2, 3, 1, 7, 1, 12, 5])
    rng = np.random.default_rng(88)
    per_group = rng.random((2, 8))
    ix = np.repeat(np.arange(8), [1, 2, 3, 1, 7, 1, 12, 5])
    b = np.ones(len(ix))
    observed = select_against_mean(b, per_group[:, ix], groups=groups)
    same = select_against_mean(np.ones(8), per_group)
    np.testing.assert_allclose(observed.gains, same.gains)
    assert observed.threshold == pytest.approx(same.threshold)
    assert observed.index == same.index and observed.n_units == 8
    collapsed = select_against_mean(b, np.zeros((1, len(b))), groups=np.zeros(len(b)))
    assert collapsed.reason == "too_few_units" and collapsed.threshold is None and collapsed.index is None
    assert select_against_mean(b, []).reason == "no_candidates"
    assert select_against_mean([], np.empty((2, 0))).reason == "too_few_units"


def test_returned_trees_rebind_and_preserve_candidate_cache():
    set_backend("numpy")
    with use_postprocessing_function(_passthrough):
        fit = {"a": np.arange(3.) / 3, "b": np.zeros(3)}
        v = {"b": np.zeros(8), "a": np.ones(8)}
        candidate = Tree(ValueNode(None, np.ones(1), "a"))
        cached = candidate.evaluation

        def row_loss(p, y):
            return (p - y) ** 2

        chosen, r = select_tree_against_mean([candidate], fit, v, np.ones(8), row_loss)
        assert r.index == 0 and r.reason == "accepted" and r.threshold == 0.
        np.testing.assert_equal(chosen.predict(), fit["a"])
        baseline, r = select_tree_against_mean([candidate], fit, v, np.full(8, .5), row_loss)
        assert r.index is None
        np.testing.assert_equal(baseline.predict(), .5 * fit["a"])
        assert candidate.evaluation is cached and candidate.root.value.shape == (1,)
        with pytest.raises(ValueError, match="loss"):
            select_tree_against_mean([candidate], fit, v, np.zeros(8), lambda p, y: ((p - y) ** 2).mean())


@pytest.mark.parametrize("b,losses,options", [
    ([.1], [[np.inf]], {}), ([[0]], [[0]], {}), ([.1, .2], [[.1]], {}), ([.1], [[.1]], {"draws": 0}),
    ([.1], [[.1]], {"alpha": 1.0}), ([.1], [[.1]], {"min_units": 1}), ([.1], [[.1]], {"groups": []}),
])
def test_invalid_evidence(b, losses, options):
    with pytest.raises(ValueError):
        select_against_mean(b, losses, **options)
