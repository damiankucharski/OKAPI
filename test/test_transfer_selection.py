import numpy as np
import pytest

from okapi.functions import scale_vector_to_sum_1
from okapi.globals import _passthrough, use_postprocessing_function
from okapi.node import BinaryAffineNode, MeanNode, ValueNode, WeightedLogitMeanNode
from okapi.transfer_selection import select_transfer_candidate, shrink_subtree_weights
from okapi.tree import Tree


def test_selects_by_clean_window_then_all_environment_minimum():
    scores = {
        "best clean": (0.875, 0.56, 0.7),
        "on boundary": (0.75, 0.80, 0.85),
        "clean is limiting": (0.70, 0.99, 0.99),
        "outside": (np.nextafter(0.75, 0), 0.95, 0.95),
    }
    assert select_transfer_candidate(scores, clean_tolerance=0.125) == "on boundary"
    assert select_transfer_candidate(scores, clean_tolerance=1.0) == "on boundary"
    assert select_transfer_candidate(scores, clean_tolerance=0.0) == "best clean"


def test_ties_prefer_clean_then_insertion_order_without_comparing_keys():
    first, second, third = object(), object(), object()
    scores = {first: (42.0, -7.0), second: (42.005, -7.0), third: (42.005, -7.0)}
    assert select_transfer_candidate(scores) is second
    assert select_transfer_candidate(dict(reversed(scores.items()))) is third


@pytest.mark.parametrize("scores,tolerance", [
    ({}, 0.0),
    ({"a": [0.2]}, 0.0),
    ({"a": [0.1, 0.2], "b": [0.0, 0.1, 0.2]}, 0.0),
    ({"a": [0.2, np.nan]}, 0.0),
    ({"a": [[0, 1], [2, 3]]}, 0.0),
    ({"a": [1, 2]}, -0.1),
    ({"a": [1, 2]}, [0.1]),
    ({"a": [1, 2]}, np.inf),
    ({"a": ["1", "2"]}, 0.0),
])
def test_rejects_invalid_scores_and_tolerances(scores, tolerance):
    with pytest.raises(ValueError):
        select_transfer_candidate(scores, clean_tolerance=tolerance)


@pytest.mark.parametrize("multiclass", [False, True])
def test_shrink_signed_final_head_only_with_fresh_private_state(multiclass):
    a = np.array([[0.1, 0.7, 0.2], [0.65, 0.1, 0.25]]) if multiclass else np.array([[0.1], [0.7]])
    b = a[::-1].copy()
    # Reuse the first tensor in two positions. A previous root update supplies
    # the first slot of the head, and the final slot contains a complete subtree.
    parent = ValueNode([MeanNode([ValueNode(None, b, 1)])], a, 0)
    raw = ValueNode(None, b, 1)
    extra = ValueNode([MeanNode([ValueNode(None, b, 1)])], a, 0)
    weights = [-1.5, 0.65, -3.0]
    node = WeightedLogitMeanNode([raw, extra], weights) if multiclass else BinaryAffineNode([raw, extra], weights, 0.37)
    parent.add_child(node)
    tree = Tree(parent, mutation_chance=0.37)
    source_key = tree.structural_key()
    with use_postprocessing_function(scale_vector_to_sum_1 if multiclass else _passthrough):
        original = tree.predict(clear_cache=False)
        cache = tree.root.evaluation
        scaled = shrink_subtree_weights(tree, 2)
        assert scaled.mutation_chance == tree.mutation_chance
        assert all(v.evaluation is None for v in scaled.value_nodes)
        assert tree.root.evaluation is cache
        assert tree.structural_key() == source_key
        head = scaled.root.children[-1]
        assert (head._weights if multiclass else head.coefficients) == [-1.5, 0.65, -1.5]
        if not multiclass:
            assert head.intercept == 0.37
        # Independent analytical calculation incorporates the earlier update
        # and subtree, whose means both equal (a + b) / 2.
        avg = (a + b) / 2
        if multiclass:
            expected = np.exp(-3 * np.log(avg) + 0.65 * np.log(b))
            expected /= expected.sum(axis=1, keepdims=True)
        else:
            def logit(q):
                return np.log(q / (1 - q))

            expected = 1 / (1 + np.exp(-(-3 * logit(avg) + 0.65 * logit(b) + 0.37)))
        np.testing.assert_allclose(scaled.predict(), expected)
        np.testing.assert_array_equal(shrink_subtree_weights(tree, 2, 1).predict(), original)
        assert scaled.root.value is scaled.root.children[-1].children[-1].value
        scaled.root.value[...] = 0.25
        for src in tree.value_nodes:
            assert not np.shares_memory(scaled.root.value, src.value)
        np.testing.assert_array_equal(tree.predict(), original)


def test_shrink_validation_and_no_added_features():
    x = np.array([[0.4]])
    root = ValueNode(None, x, 1)
    with pytest.raises(ValueError):
        shrink_subtree_weights(Tree(root), 1)
    root.add_child(MeanNode([]))
    with pytest.raises(ValueError):
        shrink_subtree_weights(Tree(root), 1)
    root.add_child(BinaryAffineNode([], [-0.7]))
    tree = Tree(root)
    assert shrink_subtree_weights(tree, 1, 0).structural_key() == tree.structural_key()
    for count, factor in [(0, 0.5), (2, 0.5), (True, 1), (1.0, 0), (1, -0.5), (1, 1.5), (1, np.nan)]:
        with pytest.raises(ValueError):
            shrink_subtree_weights(tree, count, factor)
