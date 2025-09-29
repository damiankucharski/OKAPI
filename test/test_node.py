from unittest import mock

import numpy as np
import pytest

from giraffe.backend.numpy_backend import NumpyBackend as B
from giraffe.node import (
    CloseThresholdNode,
    FarThresholdNode,
    MaxNode,
    MeanNode,
    MinNode,
    Node,
    OperatorNode,
    ValueNode,
    WeightedMeanNode,
    check_if_both_types_operators,
    check_if_both_types_values,
)


@pytest.fixture(autouse=True)
def mock_random_uniform():
    with mock.patch("numpy.random.uniform", return_value=0.3) as mocked_uniform:
        yield mocked_uniform  # This allows the mock to be used in tests


@pytest.fixture
def node_set():
    nset = {
        "A": Node(),
        "B": Node(),
        "C": Node(),
        "D": Node(),
        "E": Node(),
    }
    return nset


@pytest.fixture
def example_tree(node_set):
    """Create a tree with the following structure:
    A
    ├── B
    │   ├── D
    │   └── E
    └── C
    """
    node_set["A"].add_child(node_set["B"])
    node_set["A"].add_child(node_set["C"])
    node_set["B"].add_child(node_set["D"])
    node_set["B"].add_child(node_set["E"])
    return node_set


def test_add_child(example_tree):
    assert example_tree["A"].children == [example_tree["B"], example_tree["C"]]
    assert example_tree["B"].children == [example_tree["D"], example_tree["E"]]
    assert example_tree["C"].children == []
    assert example_tree["D"].children == []
    assert example_tree["E"].children == []

    assert example_tree["A"].parent is None
    assert example_tree["B"].parent is example_tree["A"]
    assert example_tree["C"].parent is example_tree["A"]
    assert example_tree["D"].parent is example_tree["B"]
    assert example_tree["E"].parent is example_tree["B"]


def test_remove_child(example_tree):
    example_tree["A"].remove_child(example_tree["B"])
    example_tree["B"].remove_child(example_tree["D"])

    assert example_tree["A"].children == [example_tree["C"]]
    assert example_tree["B"].children == [example_tree["E"]]
    assert example_tree["C"].children == []
    assert example_tree["D"].children == []
    assert example_tree["E"].children == []

    assert example_tree["A"].parent is None
    assert example_tree["B"].parent is None
    assert example_tree["C"].parent is example_tree["A"]
    assert example_tree["D"].parent is None
    assert example_tree["E"].parent is example_tree["B"]


def test_replace_child_correct(example_tree):
    replacement = Node()

    example_tree["A"].replace_child(example_tree["B"], replacement)

    assert example_tree["A"].children == [replacement, example_tree["C"]]
    assert example_tree["B"].children == [example_tree["D"], example_tree["E"]]
    assert example_tree["C"].children == []
    assert example_tree["D"].children == []
    assert example_tree["E"].children == []
    assert replacement.children == []

    assert example_tree["A"].parent is None
    assert example_tree["B"].parent is None
    assert example_tree["C"].parent is example_tree["A"]
    assert example_tree["D"].parent is example_tree["B"]
    assert example_tree["E"].parent is example_tree["B"]
    assert replacement.parent is example_tree["A"]


def test_replace_child_incorrect(example_tree):
    with pytest.raises(ValueError):
        example_tree["A"].replace_child(example_tree["B"], example_tree["C"])


def test_get_nodes(example_tree):
    nodes = example_tree["A"].get_nodes()
    assert nodes == [example_tree["A"], example_tree["B"], example_tree["C"], example_tree["D"], example_tree["E"]]


def test_copy(example_tree):
    copy = example_tree["A"].copy()
    assert copy.children == []
    assert copy.parent is None


def test_copy_subtree(example_tree):
    copy = example_tree["A"].copy_subtree()
    nodes = copy.get_nodes()

    A, B, C, D, E = nodes

    assert A.children == [B, C]
    assert B.children == [D, E]
    assert C.children == []
    assert D.children == []
    assert E.children == []

    assert A.parent is None
    assert B.parent is A
    assert C.parent is A
    assert D.parent is B
    assert E.parent is B


@pytest.fixture
def value_op_base_set():
    def x():
        return np.array([[2, 2], [3, 3]])

    nset = {
        "A": ValueNode(None, x(), 1),
        "B": OperatorNode(None),
        "C": ValueNode(None, x(), 2),
        "D": ValueNode(None, x(), 3),
    }

    return nset


@pytest.fixture
def value_op_base_tree(value_op_base_set):
    """Create a tree with the following structure:
    A
    ├── B
    │   ├── D
    │   └── C
    """
    value_op_base_set["B"].add_child(value_op_base_set["D"])
    value_op_base_set["B"].add_child(value_op_base_set["C"])
    value_op_base_set["A"].add_child(value_op_base_set["B"])
    return value_op_base_set


def test_concat(value_op_base_tree):
    concat = value_op_base_tree["B"]._concat()
    assert np.array_equal(B.shape(concat), (3, 2, 2))


@pytest.fixture
def mean_tree_node_set():
    a = np.array([[2, 2], [3, 3]])
    c = np.array([[3, 3], [4, 4]])
    d = np.array([[4, 4], [5, 5]])

    nset = {
        "A": ValueNode(None, a, 1),
        "B": MeanNode(None),
        "C": ValueNode(None, c, 2),
        "D": ValueNode(None, d, 3),
    }

    return nset


@pytest.fixture
def mean_tree(mean_tree_node_set):
    """Create a tree with the following structure:
    A
    ├── B
    │   ├── D
    │   └── C
    """
    mean_tree_node_set["B"].add_child(mean_tree_node_set["D"])
    mean_tree_node_set["B"].add_child(mean_tree_node_set["C"])
    mean_tree_node_set["A"].add_child(mean_tree_node_set["B"])
    return mean_tree_node_set


def test_mean(mean_tree):
    mean_tree["A"].calculate()

    evaluation_A = mean_tree["A"].evaluation
    evaluation_C = mean_tree["C"].evaluation
    evaluation_D = mean_tree["D"].evaluation

    assert np.array_equal(evaluation_A, np.array([[3, 3], [4, 4]]))
    assert np.array_equal(evaluation_C, np.array([[3, 3], [4, 4]]))
    assert np.array_equal(evaluation_D, np.array([[4, 4], [5, 5]]))


@pytest.fixture
def weighted_mean_tree():
    a = np.array([[2, 2], [3, 3]])
    c = np.array([[3, 3], [4, 4]])
    d = np.array([[4, 4], [5, 5]])

    nset = {
        "A": ValueNode(None, a, 1),
        "C": ValueNode(None, c, 2),
        "D": ValueNode(None, d, 3),
    }

    weights = [0.3, 0.2, 0.5]

    nset["B"] = WeightedMeanNode([nset["C"], nset["D"]], weights)  # type: ignore
    nset["A"].add_child(nset["B"])

    return nset


def test_weighted_mean(weighted_mean_tree):
    weighted_mean_tree["A"].calculate()

    evaluation_A = weighted_mean_tree["A"].evaluation
    evaluation_C = weighted_mean_tree["C"].evaluation
    evaluation_D = weighted_mean_tree["D"].evaluation

    expected_weighted = np.array([[3.2, 3.2], [4.2, 4.2]])

    assert np.array_equal(evaluation_A, expected_weighted)
    assert np.array_equal(evaluation_C, np.array([[3, 3], [4, 4]]))
    assert np.array_equal(evaluation_D, np.array([[4, 4], [5, 5]]))


def test_weighted_mean_child_remove(weighted_mean_tree):
    weighted_mean_tree["B"].remove_child(weighted_mean_tree["C"])

    expected_weights = np.array([0.375, 0.625])

    np.testing.assert_array_almost_equal(weighted_mean_tree["B"].weights, expected_weights)

    weighted_mean_tree["A"].calculate()

    evaluation_A = weighted_mean_tree["A"].evaluation
    evaluation_C = weighted_mean_tree["C"].evaluation
    evaluation_D = weighted_mean_tree["D"].evaluation

    expected_weighted = np.array([[3.25, 3.25], [4.25, 4.25]])

    np.testing.assert_array_almost_equal(evaluation_A, expected_weighted)
    assert evaluation_C is None
    np.testing.assert_array_almost_equal(evaluation_D, np.array([[4, 4], [5, 5]]))


def test_weighted_mean_child_add(weighted_mean_tree):
    e = np.array([[5, 5], [6, 6]])

    weighted_mean_tree["E"] = ValueNode(None, e, 4)
    weighted_mean_tree["B"].add_child(weighted_mean_tree["E"])

    expected_weights = np.array([0.3 * 0.7, 0.2 * 0.7, 0.5 * 0.7, 0.3])

    np.testing.assert_array_almost_equal(weighted_mean_tree["B"].weights, expected_weights)

    weighted_mean_tree["A"].calculate()

    evaluation_A = weighted_mean_tree["A"].evaluation
    evaluation_C = weighted_mean_tree["C"].evaluation
    evaluation_D = weighted_mean_tree["D"].evaluation
    evaluation_E = weighted_mean_tree["E"].evaluation

    expected_weighted = np.array([[3.74, 3.74], [4.74, 4.74]])

    np.testing.assert_array_almost_equal(evaluation_A, expected_weighted)
    np.testing.assert_array_almost_equal(evaluation_C, np.array([[3, 3], [4, 4]]))
    np.testing.assert_array_almost_equal(evaluation_D, np.array([[4, 4], [5, 5]]))
    np.testing.assert_array_almost_equal(evaluation_E, np.array([[5, 5], [6, 6]]))


def test_weighted_mean_copy_subtree(weighted_mean_tree):
    copy = weighted_mean_tree["A"].copy_subtree()
    nodes = copy.get_nodes()

    A, B, C, D = nodes

    assert A.children == [B]
    assert B.children == [C, D]
    assert C.children == []
    assert D.children == []

    assert A.parent is None
    assert B.parent is A
    assert C.parent is B
    assert D.parent is B

    np.testing.assert_equal(B.weights, weighted_mean_tree["B"].weights)


@pytest.fixture
def min_tree():
    a = np.array([[2, 2], [3, 3]])
    c = np.array([[3, 3], [4, 4]])
    d = np.array([[4, 4], [5, 5]])

    nset = {
        "A": ValueNode(None, a, 1),
        "C": ValueNode(None, c, 2),
        "D": ValueNode(None, d, 3),
    }

    nset["B"] = MinNode([nset["C"], nset["D"]])  # type: ignore
    nset["A"].add_child(nset["B"])

    return nset


def test_min_tree(min_tree):
    min_tree["A"].calculate()

    evaluation_A = min_tree["A"].evaluation
    evaluation_C = min_tree["C"].evaluation
    evaluation_D = min_tree["D"].evaluation

    expected = np.array([[2.0, 2.0], [3.0, 3.0]])

    assert np.array_equal(evaluation_A, expected)
    assert np.array_equal(evaluation_C, np.array([[3, 3], [4, 4]]))
    assert np.array_equal(evaluation_D, np.array([[4, 4], [5, 5]]))


@pytest.fixture
def max_tree():
    a = np.array([[2, 2], [3, 3]])
    c = np.array([[3, 3], [4, 4]])
    d = np.array([[4, 4], [5, 5]])

    nset = {
        "A": ValueNode(None, a, 1),
        "C": ValueNode(None, c, 2),
        "D": ValueNode(None, d, 3),
    }

    nset["B"] = MaxNode([nset["C"], nset["D"]])  # type: ignore
    nset["A"].add_child(nset["B"])

    return nset


def test_max_tree(max_tree):
    max_tree["A"].calculate()

    evaluation_A = max_tree["A"].evaluation
    evaluation_C = max_tree["C"].evaluation
    evaluation_D = max_tree["D"].evaluation

    expected = np.array([[4.0, 4.0], [5.0, 5.0]])

    assert np.array_equal(evaluation_A, expected)
    assert np.array_equal(evaluation_C, np.array([[3, 3], [4, 4]]))
    assert np.array_equal(evaluation_D, np.array([[4, 4], [5, 5]]))


@pytest.fixture
def threshold_tree_close():
    a = np.array([[0.2, 0.2], [0.3, 0.3]])
    c = np.array([[0.3, 0.3], [0.4, 0.4]])
    d = np.array([[0.4, 0.4], [0.5, 0.5]])

    nset = {
        "A": ValueNode(None, a, 1),
        "C": ValueNode(None, c, 2),
        "D": ValueNode(None, d, 3),
    }

    nset["B"] = CloseThresholdNode([nset["C"], nset["D"]], 0.4)  # type: ignore
    nset["A"].add_child(nset["B"])

    return nset


def test_threshold_close_tree(threshold_tree_close):
    threshold_tree_close["A"].calculate()

    evaluation_A = threshold_tree_close["A"].evaluation
    evaluation_C = threshold_tree_close["C"].evaluation
    evaluation_D = threshold_tree_close["D"].evaluation

    expected = np.array([[0.4, 0.4], [0.4, 0.4]])

    assert np.array_equal(evaluation_A, expected)
    assert np.array_equal(evaluation_C, np.array([[0.3, 0.3], [0.4, 0.4]]))
    assert np.array_equal(evaluation_D, np.array([[0.4, 0.4], [0.5, 0.5]]))


@pytest.fixture
def threshold_tree_far():
    a = np.array([[0.2, 0.2], [0.3, 0.3]])
    c = np.array([[0.3, 0.3], [0.4, 0.4]])
    d = np.array([[0.4, 0.4], [0.5, 0.5]])

    nset = {
        "A": ValueNode(None, a, 1),
        "C": ValueNode(None, c, 2),
        "D": ValueNode(None, d, 3),
    }

    nset["B"] = FarThresholdNode([nset["C"], nset["D"]], 0.4)  # type: ignore
    nset["A"].add_child(nset["B"])

    return nset


def test_threshold_far_tree(threshold_tree_far):
    threshold_tree_far["A"].calculate()

    evaluation_A = threshold_tree_far["A"].evaluation
    evaluation_C = threshold_tree_far["C"].evaluation
    evaluation_D = threshold_tree_far["D"].evaluation

    expected = np.array([[0.2, 0.2], [0.3, 0.3]])

    assert np.array_equal(evaluation_A, expected)
    assert np.array_equal(evaluation_C, np.array([[0.3, 0.3], [0.4, 0.4]]))
    assert np.array_equal(evaluation_D, np.array([[0.4, 0.4], [0.5, 0.5]]))


@pytest.mark.parametrize(
    "type_1, type_2, expected",
    [
        (ValueNode, ValueNode, True),
        (OperatorNode, OperatorNode, False),
        (ValueNode, OperatorNode, False),
        (OperatorNode, ValueNode, False),
    ],
)
def test_check_both_value(type_1, type_2, expected):
    assert check_if_both_types_values(type_1, type_2) == expected


@pytest.mark.parametrize(
    "type_1, type_2, expected",
    [
        (ValueNode, ValueNode, False),
        (OperatorNode, OperatorNode, True),
        (ValueNode, OperatorNode, False),
        (OperatorNode, ValueNode, False),
        (MeanNode, WeightedMeanNode, True),
        (WeightedMeanNode, MeanNode, True),
        (MinNode, MaxNode, True),
        (MaxNode, MinNode, True),
        (ValueNode, MinNode, False),
        (MinNode, ValueNode, False),
        (ValueNode, MaxNode, False),
        (MaxNode, ValueNode, False),
        (MeanNode, MaxNode, True),
        (MaxNode, MeanNode, True),
        (MeanNode, MinNode, True),
        (MinNode, MeanNode, True),
        (FarThresholdNode, MeanNode, True),
        (FarThresholdNode, MaxNode, True),
        (FarThresholdNode, MinNode, True),
        (FarThresholdNode, WeightedMeanNode, True),
        (FarThresholdNode, ValueNode, False),  # No need to test CloseThresholdNode as it has the same implementation
    ],
)
def test_check_both_operators(type_1, type_2, expected):
    assert check_if_both_types_operators(type_1, type_2) == expected
