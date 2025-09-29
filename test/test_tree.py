from pathlib import Path

import numpy as np
import pytest

from giraffe.globals import BACKEND as B
from giraffe.node import OperatorNode, ValueNode, WeightedMeanNode
from giraffe.tree import Tree


@pytest.fixture
def value_op_base_set():
    def x():
        return np.array([[2, 2], [3, 3]])

    nset = {
        "A": ValueNode(None, x(), "A"),
        "B": OperatorNode(None),
        "C": OperatorNode(None),
        "D": ValueNode(None, x(), "D"),
        "E": ValueNode(None, x(), "E"),
        "F": ValueNode(None, x(), "F"),
        "G": ValueNode(None, x(), "G"),
        "H": ValueNode(None, x(), "H"),
    }

    return nset


@pytest.fixture
def two_base_trees(value_op_base_set):
    r"""
    Creates two trees with the following structure:
    Tree 1:
         A
         |
         B
        /|\
       / | \
      D  E  G

    Tree 2:
         F
         |
         C
         |
         H
    """

    value_op_base_set["A"].add_child(value_op_base_set["B"])
    value_op_base_set["B"].add_child(value_op_base_set["D"])
    value_op_base_set["B"].add_child(value_op_base_set["E"])
    value_op_base_set["B"].add_child(value_op_base_set["G"])

    value_op_base_set["F"].add_child(value_op_base_set["C"])
    value_op_base_set["C"].add_child(value_op_base_set["H"])

    tree1 = Tree.create_tree_from_root(value_op_base_set["A"])
    tree2 = Tree.create_tree_from_root(value_op_base_set["F"])

    return tree1, tree2, value_op_base_set


def test_tree_nodes_lists(two_base_trees):
    tree1, tree2, value_op_base_set = two_base_trees

    assert len(tree1.nodes["value_nodes"]) == 4
    assert len(tree1.nodes["op_nodes"]) == 1

    assert len(tree2.nodes["value_nodes"]) == 2
    assert len(tree2.nodes["op_nodes"]) == 1


def test_prune_at(two_base_trees):
    tree1, _, value_op_base_set = two_base_trees

    tree1.prune_at(value_op_base_set["B"])

    assert len(tree1.nodes["value_nodes"]) == 1
    assert len(tree1.nodes["op_nodes"]) == 0

    assert tree1.root == value_op_base_set["A"]
    assert tree1.root.children == []
    assert value_op_base_set["B"].parent is None
    assert value_op_base_set["D"].parent == value_op_base_set["B"]
    assert value_op_base_set["E"].parent == value_op_base_set["B"]
    assert value_op_base_set["G"].parent == value_op_base_set["B"]


def test_prune_at_only_child_of_op(two_base_trees):
    _, tree2, value_op_base_set = two_base_trees

    tree2.prune_at(value_op_base_set["H"])

    assert len(tree2.nodes["value_nodes"]) == 1
    assert len(tree2.nodes["op_nodes"]) == 0

    assert tree2.root == value_op_base_set["F"]
    assert tree2.root.children == []
    assert value_op_base_set["F"].parent is None
    assert value_op_base_set["C"].parent is None
    assert value_op_base_set["H"].parent is value_op_base_set["C"]


@pytest.mark.parametrize("node_should_fail", ["F", "C", "H"])
def test_prune_at_fails(two_base_trees, node_should_fail):
    tree1, _, value_op_base_set = two_base_trees

    with pytest.raises(ValueError):
        tree1.prune_at(value_op_base_set[node_should_fail])


def test_append_after(two_base_trees):
    tree1, tree2, value_op_base_set = two_base_trees

    branch = tree1.prune_at(value_op_base_set["B"])

    tree2.append_after(value_op_base_set["F"], branch)

    assert len(tree2.nodes["value_nodes"]) == 5
    assert len(tree2.nodes["op_nodes"]) == 2

    assert tree2.root == value_op_base_set["F"]
    assert value_op_base_set["F"].children == [value_op_base_set["C"], value_op_base_set["B"]]

    assert value_op_base_set["B"].parent == value_op_base_set["F"]
    assert value_op_base_set["C"].parent == value_op_base_set["F"]

    assert value_op_base_set["D"].parent == value_op_base_set["B"]
    assert value_op_base_set["E"].parent == value_op_base_set["B"]
    assert value_op_base_set["G"].parent == value_op_base_set["B"]
    assert value_op_base_set["B"].children == [value_op_base_set["D"], value_op_base_set["E"], value_op_base_set["G"]]

    assert value_op_base_set["H"].parent == value_op_base_set["C"]
    assert value_op_base_set["C"].children == [value_op_base_set["H"]]


@pytest.mark.parametrize("node_should_fail", ["A", "B", "D", "E", "G", "C"])
def test_append_after_fails(two_base_trees, node_should_fail):
    tree1, tree2, value_op_base_set = two_base_trees

    branch = tree1.prune_at(value_op_base_set["B"])

    with pytest.raises(ValueError):
        tree2.append_after(value_op_base_set[node_should_fail], branch)


def test_replace_at(two_base_trees):
    tree1, tree2, value_op_base_set = two_base_trees

    branch = tree2.prune_at(value_op_base_set["C"])

    tree1.replace_at(value_op_base_set["B"], branch)

    assert len(tree1.nodes["value_nodes"]) == 4
    assert len(tree1.nodes["op_nodes"]) == 1

    assert tree1.root == value_op_base_set["A"]
    assert value_op_base_set["A"].children == [value_op_base_set["C"]]
    assert value_op_base_set["C"].parent == value_op_base_set["A"]


def test_get_random_node_root(two_base_trees):
    tree1, tree2, value_op_base_set = two_base_trees

    a_copy = value_op_base_set["A"].copy()

    t = Tree.create_tree_from_root(a_copy)

    o1 = t.get_random_node("value_nodes", True, True)

    assert o1 == a_copy

    with pytest.raises(ValueError):
        t.get_random_node("op_nodes", True, True)
    with pytest.raises(ValueError):
        t.get_random_node("value_nodes", False, True)


def test_get_random_node(two_base_trees):
    tree1, tree2, value_op_base_set = two_base_trees

    o1 = tree1.get_random_node("value_nodes", True, True)
    o2 = tree1.get_random_node("op_nodes", True, True)

    assert o1 in value_op_base_set.values()
    assert o2 in value_op_base_set.values()

    assert isinstance(o1, ValueNode)
    assert isinstance(o2, OperatorNode)


@pytest.fixture
def weighted_mean_tree():
    a = np.array([[2, 2], [3, 3]])
    c = np.array([[3, 3], [4, 4]])
    d = np.array([[4, 4], [5, 5]])

    nset = {
        "A": ValueNode(None, a, "1.npy"),
        "C": ValueNode(None, c, "2.npy"),
        "D": ValueNode(None, d, "3.npy"),
    }

    weights = [0.3, 0.2, 0.5]

    nset["B"] = WeightedMeanNode([nset["C"], nset["D"]], weights)  # type: ignore
    nset["A"].add_child(nset["B"])

    return nset


def test_weighted_tree_evaluation(weighted_mean_tree):
    tree = Tree.create_tree_from_root(weighted_mean_tree["A"])
    evaluation = tree.evaluation
    np.testing.assert_array_equal(B.to_numpy(evaluation), np.array([[3.2, 3.2], [4.2, 4.2]]))


# this should test all parametrized nodes
def test_save_and_load_architecture_weighted_mean(weighted_mean_tree):
    import os

    tree = Tree.create_tree_from_root(weighted_mean_tree["A"])

    if not os.path.exists(".test_dump"):
        os.makedirs(".test_dump")
    path = ".test_dump/test_tree_architecture.pkl"

    tree.save_tree_architecture(path)
    loaded_tree = Tree.load_tree_architecture(path)
    np.testing.assert_equal(loaded_tree.nodes["op_nodes"][0].weights, weighted_mean_tree["B"].weights)


def test_save_and_load_tree_weighted_tree(weighted_mean_tree):
    import os

    tree = Tree.create_tree_from_root(weighted_mean_tree["A"])

    if not os.path.exists(".test_dump"):
        os.makedirs(".test_dump")
    if not os.path.exists(".test_dump/test_tensors"):
        os.makedirs(".test_dump/test_tensors")

    for value_node in tree.nodes["value_nodes"]:
        np.save(f".test_dump/test_tensors/{value_node.id}", value_node.value)

    path = Path(".test_dump/test_tree.pkl")

    tree.save_tree_architecture(path)

    loaded_tree, _ = Tree.load_tree(path, Path(".test_dump/test_tensors"))

    np.testing.assert_equal(loaded_tree.nodes["op_nodes"][0].weights, weighted_mean_tree["B"].weights)

    for tree_node, loaded_tree_node in zip(tree.nodes["value_nodes"], loaded_tree.nodes["value_nodes"], strict=False):
        assert tree_node.id == loaded_tree_node.id
        np.testing.assert_equal(tree_node.value, loaded_tree_node.value)


@pytest.fixture
def duplicated_nodes_set():
    def x():
        return np.array([[2, 2], [3, 3]])

    nset = {
        "A": ValueNode(None, x(), "A"),
        "B": OperatorNode(None),
        "D": ValueNode(None, x(), "A"),
        "E": ValueNode(None, x(), "D"),
        "F": ValueNode(None, x(), "C"),
        "G": ValueNode(None, x(), "B"),
        "H": ValueNode(None, x(), "B"),
    }

    return nset


@pytest.fixture
def duplicated_values_tree(duplicated_nodes_set) -> Tree:
    duplicated_nodes_set["A"].add_child(duplicated_nodes_set["B"])
    duplicated_nodes_set["B"].add_child(duplicated_nodes_set["D"])
    duplicated_nodes_set["B"].add_child(duplicated_nodes_set["E"])
    duplicated_nodes_set["B"].add_child(duplicated_nodes_set["G"])

    duplicated_nodes_set["B"].add_child(duplicated_nodes_set["F"])
    duplicated_nodes_set["B"].add_child(duplicated_nodes_set["H"])

    tree1 = Tree.create_tree_from_root(duplicated_nodes_set["A"])

    return tree1


def test_unique_value_node_ids(duplicated_values_tree):
    unique_ids = duplicated_values_tree.unique_value_node_ids
    assert sorted(unique_ids) == ["A", "B", "C", "D"]
