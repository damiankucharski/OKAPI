import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function
from okapi.interventions import operator_paths, tree_with_operand_provider
from okapi.node import MaxNode, MinNode, ValueNode, WeightedMeanNode
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def numpy_backend(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)


def _source():
    a, b, c = (np.asarray(v) for v in ([.2, .7], [.8, .3], [.5, .6]))
    left = ValueNode([MinNode([ValueNode(None, c, "c")])], b, "b")
    right = ValueNode(None, b, "b")
    update = WeightedMeanNode([left, right], [.25, .5, .25])
    original = Tree(ValueNode([update, MaxNode([ValueNode(None, a, "a")])], a, "a"), .62)
    original.nodes["op_nodes"].reverse()
    return original, a, b, c


def test_rebinding_a_nonterminal_operand_keeps_all_updates_and_indexes():
    source, a, b, c = _source()
    before = source.evaluation
    source_key = source.structural_key()
    new = np.array([.1, .95])
    changed = tree_with_operand_provider(source, (0,), np.int64(0), "new", new)
    assert changed.mutation_chance == .62
    assert [type(op) for op in changed.op_nodes] == [type(op) for op in source.op_nodes]
    nodes = dict(operator_paths(changed))
    assert nodes[(0,)]._weights == [.25, .5, .25]
    selected, sibling = nodes[(0,)].children
    assert (selected.id, sibling.id) == ("new", "b")
    assert selected.value is new and sibling.value is b and selected.children[0] is nodes[(0, 0, 0)]
    assert selected is not source.root.children[0].children[0]
    np.testing.assert_allclose(changed.predict(), np.maximum(a, .25 * a + .5 * np.minimum(new, c) + .25 * b))
    assert source.evaluation is before and source.structural_key() == source_key
    original = tree_with_operand_provider(source, (0,), 1, "b", b)
    assert original.structural_key() == source_key
    np.testing.assert_array_equal(original.predict(), before)


@pytest.mark.parametrize("path,slot,new_id,tensor,error", [
    ((1,), 2, "x", np.ones(2), ValueError),
    ((0, 0), 0, "x", np.ones(2), ValueError),
    ((False,), 0, "x", np.ones(2), ValueError),
    ((0,), -1, "x", np.ones(2), ValueError),
    ((0,), True, "x", np.ones(2), TypeError),
    ((0,), 0, [], np.ones(2), TypeError),
    ((0,), 0, "x", np.ones((2, 1)), ValueError),
])
def test_invalid_rebinding_leaves_parent_intact(path, slot, new_id, tensor, error):
    tree, *_ = _source()
    source_key = tree.structural_key()
    with pytest.raises(error):
        tree_with_operand_provider(tree, path, slot, new_id, tensor)
    assert tree.structural_key() == source_key


@pytest.mark.parametrize("new_type", [
    "MEAN", "MIN", "MAX", "WEIGHTED_MEAN", "FAR_THRESHOLD", "CLOSE_THRESHOLD",
])
def test_type_and_operand_edits_commute_with_the_same_parameter_draws(new_type):
    from okapi import operators
    from okapi.mutation import tree_with_operator_type
    from okapi.random_state import use_rng

    source, *_ = _source()
    new = np.array([.1, .95])
    node_type = getattr(operators, new_type)
    typed = np.random.RandomState(83)
    rebound = np.random.RandomState(83)
    with use_rng(typed):
        first = tree_with_operator_type(source, (0,), node_type)
        type_then_operand = tree_with_operand_provider(first, (0,), 0, "new", new)
    with use_rng(rebound):
        first = tree_with_operand_provider(source, (0,), 0, "new", new)
        operand_then_type = tree_with_operator_type(first, (0,), node_type)
    assert type_then_operand.structural_key() == operand_then_type.structural_key()
    assert [type(op) for op in type_then_operand.op_nodes] == [type(op) for op in operand_then_type.op_nodes]
    np.testing.assert_array_equal(type_then_operand.predict(), operand_then_type.predict())
    np.testing.assert_array_equal(typed.get_state()[1], rebound.get_state()[1])
    assert typed.get_state()[2:] == rebound.get_state()[2:]
