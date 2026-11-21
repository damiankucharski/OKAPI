from collections import Counter

import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function
from okapi.guided import correlation_templates
from okapi.node import MaxNode, MeanNode, MinNode
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def numpy_backend(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)


def signature(value, rename=None):
    expression = ("input", value.id if rename is None else rename[value.id])
    for op in value.children:
        operands = [expression, *(signature(child, rename) for child in op.children)]
        if not isinstance(op, MeanNode):
            assert isinstance(op, (MinNode, MaxNode)) and len(operands) == 2
        expression = (type(op).__name__, tuple(sorted(operands)))
    return expression


def group_count(tree):
    return 1 + sum(isinstance(op, (MinNode, MaxNode)) for op in tree.op_nodes)


def paired_fit():
    rng = np.random.default_rng(71)
    first, second = rng.uniform(.2, .8, size=(2, 100))
    noise = rng.normal(0, .015, size=(2, 100))
    copies = first + noise[0], second + noise[1]
    # a/c and e/g are perfectly anticorrelated, but each positive pair
    # has independent jitter. Absolute correlation would join the wrong pair.
    columns = (first, copies[0], 1 - first, 1 - copies[0], second, copies[1], 1 - second, 1 - copies[1])
    return dict(zip("abcdefgh", columns, strict=True))


def test_grammar_enumerates_every_structure_and_recovers_pathways():
    providers = paired_fit()
    originals = {key: array.copy() for key, array in providers.items()}
    trees = correlation_templates(providers)
    assert Counter(map(group_count, trees)) == {1: 1, 2: 2, 3: 12, 4: 120}
    assert len({signature(tree.root) for tree in trees}) == 135
    nodes_seen = set()
    for tree in trees:
        assert isinstance(tree, Tree)
        assert sorted(v.id for v in tree.value_nodes) == sorted(providers)
        for node in tree.root.get_nodes():
            assert node not in nodes_seen
            nodes_seen.add(node)
            assert all(child.parent is node for child in node.children)
    # Evaluate on fresh uncorrelated provider inputs to distinguish structurally
    # wrong groupings that might appear equivalent on the correlated fit rows.
    rng = np.random.default_rng(72)
    target = {key: rng.uniform(size=(128, 1)) for key in providers}
    ab, cd, ef, gh = [(target[a] + target[b]) / 2 for a, b in ("ab", "cd", "ef", "gh")]
    expected = np.maximum(np.minimum(ab, cd), np.minimum(ef, gh))
    four_group = [tree for tree in trees if group_count(tree) == 4]
    assert any(np.allclose(tree.do_pred_on_another_tensors(current_tensors=target), expected) for tree in four_group)
    for key in providers:
        np.testing.assert_array_equal(providers[key], originals[key])


@pytest.mark.parametrize(("de_corr", "expected_groups"), [(.50, {"ab", "c", "de"}), (.30, {"abc", "d", "e"})])
def test_average_link_uses_all_cross_cluster_correlations(de_corr, expected_groups):
    correlation = np.eye(5)
    for i, j, corr in ((0, 1, .70), (0, 2, .65), (1, 2, .10), (3, 4, de_corr)):
        correlation[i, j] = correlation[j, i] = corr
    raw = np.random.default_rng(78).normal(size=(24, 5))
    basis, _ = np.linalg.qr(raw - raw.mean(axis=0))
    columns = .5 + .2 * basis @ np.linalg.cholesky(correlation).T
    providers = {key: columns[:, i:i + 1] for i, key in enumerate("abcde")}
    # After joining a,b, their average correlation with c is .375.
    tree = next(tree for tree in correlation_templates(providers) if group_count(tree) == 3)
    groups = [set([op.parent.id, *(v.id for v in op.children)]) for op in tree.op_nodes if isinstance(op, MeanNode)]
    singletons = set(providers).difference(*groups)
    assert {"".join(sorted(group)) for group in groups} | singletons == expected_groups


def test_placebo_applies_one_bijection_across_all_cuts():
    fit = paired_fit()
    guided = correlation_templates(fit)
    shuffled = correlation_templates(fit, shuffle_seed=17)
    source = next(t for t in guided if group_count(t) == 1)
    target = next(t for t in shuffled if group_count(t) == 1)
    remap = {a.id: b.id for a, b in zip(source.value_nodes, target.value_nodes, strict=True)}
    assert set(remap.values()) == set(fit) and any(key != value for key, value in remap.items())
    assert Counter(signature(t.root, remap) for t in guided) == Counter(signature(t.root) for t in shuffled)
    # Shuffling IDs must also bind each leaf to that provider's own predictions.
    for tree in shuffled:
        for v in tree.value_nodes:
            np.testing.assert_array_equal(np.ravel(v.value), fit[v.id])


def test_constant_scores_permit_repeatable_ties():
    fit = {key: np.full((8, 1), .5) for key in "abcde"}
    first = correlation_templates(fit)
    second = correlation_templates(fit)
    assert [signature(t.root) for t in first] == [signature(t.root) for t in second]
    assert len(first) == 135
    for tree in first:
        np.testing.assert_array_equal(np.ravel(tree.predict()), np.full(8, .5))


@pytest.mark.parametrize("bad", [np.ones((6, 2)), np.ones(5), np.full(6, np.nan), np.full(6, -.1), np.full(6, 1.1)])
def test_rejects_invalid_columns(bad):
    fit = {key: np.ones(6) for key in "abc"}
    with pytest.raises(ValueError):
        correlation_templates({**fit, "d": bad})


def test_requires_enough_providers_and_fit_rows():
    for n, k in ((5, 3), (1, 4)):
        with pytest.raises(ValueError):
            correlation_templates({i: np.ones(n) for i in range(k)})
