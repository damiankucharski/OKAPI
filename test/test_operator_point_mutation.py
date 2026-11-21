"""Shape-preserving native point mutation and its opt-in evolution path."""
import numpy as np
import pytest

from okapi import Okapi
from okapi.mutation import get_allowed_mutations, replace_operator_mutation, tree_with_operator_type
from okapi.node import FarThresholdNode, MaxNode, MeanNode, MinNode, ValueNode, WeightedMeanNode
from okapi.pareto import minimize
from okapi.random_state import use_rng
from okapi.tree import Tree


def pair():
    a = ValueNode(None, np.array([[0.], [0.], [1.], [1.]]), "A")
    b = ValueNode(None, np.array([[0.], [1.], [0.], [1.]]), "B")
    a.add_child(MeanNode([b]))
    return Tree(a, mutation_chance=.73)


def test_one_move_improves_and_keeps_parent_prediction():
    tree = pair()
    before = tree.predict().copy()
    child = replace_operator_mutation(tree, allowed_ops=(MeanNode, MinNode))
    labels = np.array([[0.], [0.], [0.], [1.]])
    assert np.mean((before - labels) ** 2) == 0.125
    np.testing.assert_array_equal(child.predict(), labels)
    np.testing.assert_array_equal(tree.predict(), before)
    assert child.nodes_count == tree.nodes_count == 3
    assert child.depth == tree.depth
    assert child.mutation_chance == tree.mutation_chance


def test_no_legal_substitution_preserves_random_stream_and_old_menu():
    stream = np.random.RandomState(31)
    tree = pair()
    with use_rng(stream):
        for t in (tree, Tree(ValueNode(None, np.zeros((3, 1)), "solo"))):
            assert replace_operator_mutation not in get_allowed_mutations(t)
            assert replace_operator_mutation not in get_allowed_mutations(t, point_mutation_ops=(MeanNode,))
            copy = replace_operator_mutation(t, allowed_ops=(MeanNode,))
            assert copy.structural_key() == t.structural_key()
            assert copy is not t
    assert stream.rand() == np.random.RandomState(31).rand()
    assert replace_operator_mutation in get_allowed_mutations(
        tree, max_nodes=tree.nodes_count, max_depth=tree.depth, point_mutation_ops=(MinNode,)
    )


def test_nested_operands_siblings_indices_and_lineage():
    tree = pair()
    extra_value = ValueNode(None, np.ones((4, 1)) * .4, "C")
    tree.root.children[0].add_child(extra_value)
    tree.root.add_child(MaxNode([ValueNode(None, np.ones((4, 1)) * .6, "D")]))
    extra_value.add_child(MinNode([ValueNode(None, np.ones((4, 1)) * .9, "E")]))
    tree.update_nodes()
    tree.op_nodes.reverse()
    tree.value_nodes.reverse()
    for i, node in enumerate(tree.op_nodes):
        node._okapi_gate_origin = str(i)
    orig_types = [type(n) for n in tree.op_nodes]
    before = tree.predict().copy()
    for seed in range(12):
        with use_rng(np.random.RandomState(seed)):
            child = replace_operator_mutation(tree, allowed_ops=(WeightedMeanNode,))
        assert child.nodes_count == tree.nodes_count and child.depth == tree.depth
        assert [n.id for n in child.value_nodes] == [n.id for n in tree.value_nodes]
        assert len({id(n) for n in child.root.get_nodes()} & {id(n) for n in tree.root.get_nodes()}) == 0
        changed, = [i for i, node in enumerate(child.op_nodes) if type(node) is not orig_types[i]]
        for i, (old, new) in enumerate(zip(tree.op_nodes, child.op_nodes, strict=True)):
            assert [v.id for v in new.children] == [v.id for v in old.children]
            assert new.parent.id == old.parent.id
            assert new.parent.children.index(new) == old.parent.children.index(old)
            assert all(v.parent is new for v in new.children)
            assert new._okapi_gate_origin == (None if i == changed else str(i))
        assert len(child.op_nodes[changed]._weights) == 1 + len(child.op_nodes[changed].children)
        assert np.isfinite(child.predict()).all()
    np.testing.assert_array_equal(tree.predict(), before)


def test_engine_offers_option_in_the_normal_mutation_path(tmp_path, monkeypatch):
    import okapi.mutation as mutation

    directory = tmp_path / "preds"
    directory.mkdir()
    tree = pair()
    for v in tree.value_nodes:
        np.save(directory / (v.id + ".npy"), v.value)
    labels_path = tmp_path / "y.npy"
    np.save(labels_path, np.zeros((4, 1)))
    options = dict(preds_source=directory, gt_path=labels_path,
                   population_size=2, population_multiplier=1, tournament_size=1,
                   objective_functions=(lambda p, y: np.mean((p - y) ** 2),), objectives=(minimize,),
                   backend="numpy", seed=126, allowed_ops=(MeanNode, MinNode, FarThresholdNode))
    calls = []
    original = mutation.replace_operator_mutation

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mutation, "replace_operator_mutation", record)
    for enable in (False, True):
        calls.clear()
        engine = Okapi(**options, operator_point_mutation=enable)
        tree.mutation_chance = 1.
        engine.additional_population = [tree] * 64
        with use_rng(engine.rng):
            engine._mutate_additional_population()
        assert bool(calls) == enable


def test_targeted_operator_preserves_other_updates_and_supports_matched_parameter_draws():
    tree = pair()
    tree.root.add_child(MinNode([ValueNode(None, np.full((4, 1), .25), "C")]))
    tree.update_nodes()
    with use_rng(np.random.RandomState(9)):
        edited = tree_with_operator_type(tree, (0,), FarThresholdNode)
    expected = np.random.RandomState(9).rand()
    assert edited.root.children[0].threshold == expected
    assert type(edited.root.children[1]) is MinNode
    assert edited.root.children[0].children[0].id == "B"
    assert tree.root.children[0].children[0].id == "B"
    assert type(tree.root.children[0]) is MeanNode
    stream = np.random.RandomState(10)
    with use_rng(stream):
        control = tree_with_operator_type(edited, (0,), FarThresholdNode)
    assert control.structural_key() == edited.structural_key()
    assert control is not edited and stream.rand() == np.random.RandomState(10).rand()


@pytest.mark.parametrize("path", [(), (-1,), (True,), (0, 0), (1,)])
def test_targeted_operator_rejects_invalid_path(path):
    with pytest.raises(ValueError, match="operator_path"):
        tree_with_operator_type(pair(), path, MinNode)
