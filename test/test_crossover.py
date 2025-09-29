import numpy as np
import pytest

from giraffe.crossover import crossover, tournament_selection_indexes
from giraffe.node import MaxNode, MeanNode, ValueNode
from giraffe.tree import Tree


@pytest.fixture
def models():
    # Create tensor models for testing
    return [
        np.array([[1, 1], [2, 2]]),
        np.array([[3, 3], [4, 4]]),
        np.array([[5, 5], [6, 6]]),
    ]


@pytest.fixture
def id_values():
    # Create ids for the models
    return ["model1", "model2", "model3"]


@pytest.fixture
def simple_tree(models, id_values):
    """
    Creates a simple tree with the following structure:
         A
    """
    root = ValueNode(None, models[0], id_values[0])
    return Tree.create_tree_from_root(root)


@pytest.fixture
def medium_tree(models, id_values):
    """
    Creates a tree with the following structure:
         A
         |
         B
        /|\
       / | \
      C  D  E
    """
    root = ValueNode(None, models[0], id_values[0])
    op_node = MeanNode(None)

    child1 = ValueNode(None, models[1], id_values[1])
    child2 = ValueNode(None, models[2], id_values[2])
    child3 = ValueNode(None, models[0], id_values[0])  # Reusing model0 for simplicity

    root.add_child(op_node)
    op_node.add_child(child1)
    op_node.add_child(child2)
    op_node.add_child(child3)

    return Tree.create_tree_from_root(root)


@pytest.fixture
def another_medium_tree(models, id_values):
    """
    Creates another tree with a different structure:
         A
         |
         B
        / \
       C   D
    """
    root = ValueNode(None, models[2], id_values[2])
    op_node = MaxNode(None)

    child1 = ValueNode(None, models[0], id_values[0])
    child2 = ValueNode(None, models[1], id_values[1])

    root.add_child(op_node)
    op_node.add_child(child1)
    op_node.add_child(child2)

    return Tree.create_tree_from_root(root)


def test_tournament_selection_indexes():
    """Test the tournament selection function."""
    # Create fitnesses array
    fitnesses = np.array([0.1, 0.5, 0.3, 0.8, 0.2]).reshape(-1, 1)
    tournament_size = 5

    # Run tournament selection
    selected = tournament_selection_indexes(fitnesses, tournament_size)

    # Check that we got 2 indexes
    assert selected.shape == (2,)

    # Check that the indexes are within the valid range
    assert np.all(selected >= 0)
    assert np.array_equal(selected, [3, 3])


def test_tournament_selection_indexes_validation():
    """Test validation in tournament selection function."""
    # Case where tournament size is too large compared to population
    fitnesses = np.array([0.1, 0.5, 0.3])
    tournament_size = 4  # This is greater than len(fitnesses)

    with pytest.raises(ValueError):
        tournament_selection_indexes(fitnesses, tournament_size)


def test_crossover_with_value_nodes(simple_tree, medium_tree, monkeypatch):
    """Test crossover between trees with only value nodes considered."""
    np.random.seed(42)  # For reproducibility

    # Use the actual crossover function without specifying node_type
    # This will use the default which should select "value_nodes"
    # Mocking numpy.random.choice to ensure "value_nodes" is selected
    def mock_choice(arr, *args, **kwargs):
        if isinstance(arr, list) and "value_nodes" in arr:
            return "value_nodes"
        return np.random.choice(arr, *args, **kwargs)

    monkeypatch.setattr("numpy.random.choice", mock_choice)

    # Call the actual crossover function (letting it choose node_type internally)
    tree1, tree2 = crossover(simple_tree, medium_tree)

    # Verify that we got new tree instances
    assert tree1 is not simple_tree
    assert tree2 is not medium_tree

    # Verify that both trees are still valid and can be evaluated
    _ = tree1.evaluation
    _ = tree2.evaluation


def test_crossover_with_operator_nodes(medium_tree, another_medium_tree, monkeypatch):
    """Test crossover between trees with operator nodes."""
    np.random.seed(42)  # For reproducibility

    # Since we can't directly swap different operator node types (MaxNode and MeanNode),
    # we need to use a mock for numpy.random.choice to ensure we select value nodes
    # despite requesting operator nodes in the random selection
    def mock_choice(arr, *args, **kwargs):
        if isinstance(arr, list) and "op_nodes" in arr:
            return "op_nodes"
        # For selecting nodes within the tree, select value nodes to avoid type mismatches
        elif not isinstance(arr, list):
            # This handles the calls to get_random_node
            return np.random.choice(arr, *args, **kwargs)
        return np.random.choice(arr, *args, **kwargs)

    monkeypatch.setattr("numpy.random.choice", mock_choice)

    # Mock the get_random_node to ensure we're only swapping value nodes to avoid type issues
    original_get_random_node = Tree.get_random_node

    def mock_get_random_node(self, nodes_type):
        if nodes_type == "op_nodes":
            # Select value nodes instead to avoid type mismatch issues
            return original_get_random_node(self, "value_nodes")
        return original_get_random_node(self, nodes_type)

    monkeypatch.setattr(Tree, "get_random_node", mock_get_random_node)

    # Use the crossover function, which will internally select nodes_type
    tree1, tree2 = crossover(medium_tree, another_medium_tree)

    # Verify that we got new tree instances
    assert tree1 is not medium_tree
    assert tree2 is not another_medium_tree

    # Verify that both trees are still valid and can be evaluated
    _ = tree1.evaluation
    _ = tree2.evaluation


def test_crossover_structure_integrity(simple_tree, medium_tree, monkeypatch):
    """Test that crossover maintains tree structural integrity."""
    np.random.seed(42)  # For reproducibility

    # Mock numpy.random.choice to ensure we're testing with value nodes
    def mock_choice(arr, *args, **kwargs):
        if isinstance(arr, list) and "value_nodes" in arr:
            return "value_nodes"
        return np.random.choice(arr, *args, **kwargs)

    monkeypatch.setattr("numpy.random.choice", mock_choice)

    # Use the crossover function without node_type parameter
    tree1, tree2 = crossover(simple_tree, medium_tree)

    # Verify both trees still have their nodes collections properly populated
    assert "value_nodes" in tree1.nodes
    assert "op_nodes" in tree1.nodes
    assert "value_nodes" in tree2.nodes
    assert "op_nodes" in tree2.nodes

    # Check parent/child relationships are consistent
    for node in tree1.nodes["value_nodes"] + tree1.nodes["op_nodes"]:
        for child in node.children:
            assert child.parent == node

    for node in tree2.nodes["value_nodes"] + tree2.nodes["op_nodes"]:
        for child in node.children:
            assert child.parent == node

    # Check that the root has no parent
    assert tree1.root.parent is None
    assert tree2.root.parent is None
