import numpy as np
import pytest

from okapi.mutation import append_new_node_mutation, get_allowed_mutations, lose_branch_mutation, new_tree_from_branch_mutation
from okapi.node import MeanNode, OperatorNode, ValueNode
from okapi.tree import Tree


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
    # Create ids for the models - use integers for indexing
    return [0, 1, 2]


@pytest.fixture
def simple_tree(models, id_values):
    """
    Creates a simple tree with the following structure:
         A
    """
    root = ValueNode(None, models[0], "model1")
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
    root = ValueNode(None, models[0], "model1")
    op_node = MeanNode(None)

    child1 = ValueNode(None, models[1], "model2")
    child2 = ValueNode(None, models[2], "model3")
    child3 = ValueNode(None, models[0], "model1")  # Reusing model0 for simplicity

    root.add_child(op_node)
    op_node.add_child(child1)
    op_node.add_child(child2)
    op_node.add_child(child3)

    return Tree.create_tree_from_root(root)


def test_append_new_node_mutation_to_value_node(simple_tree, models, id_values):
    """Test appending a new node after a value node."""
    # Since there's only the root node, it will always be selected in get_random_node
    np.random.seed(42)  # For reproducibility

    # Apply mutation - use the indices as IDs
    new_tree = append_new_node_mutation(simple_tree, models, id_values, allowed_ops=(MeanNode,))

    # Verify structure
    assert new_tree is not simple_tree  # Should be a different tree (copy)
    assert new_tree.nodes_count == 3  # Root + op node + value node
    assert len(new_tree.nodes["value_nodes"]) == 2
    assert len(new_tree.nodes["op_nodes"]) == 1

    # The root should have one child (operator node)
    assert len(new_tree.root.children) == 1
    assert isinstance(new_tree.root.children[0], OperatorNode)

    # The operator node should have one child (value node)
    op_node = new_tree.root.children[0]
    assert len(op_node.children) == 1
    assert isinstance(op_node.children[0], ValueNode)
    assert new_tree.nodes_count == len(new_tree.root.get_nodes())


def test_append_new_node_mutation_to_operator_node(medium_tree, models, id_values, monkeypatch):
    """Test appending a new node after an operator node."""
    np.random.seed(42)  # For reproducibility

    # Mock random choice to select the operator node

    def mock_get_random_node():
        return medium_tree.nodes["op_nodes"][0]

    monkeypatch.setattr(medium_tree, "get_random_node", mock_get_random_node)

    # Apply mutation
    new_tree = append_new_node_mutation(medium_tree, models, id_values)

    # Verify structure
    assert new_tree is not medium_tree  # Should be a different tree (copy)
    assert new_tree.nodes_count > medium_tree.nodes_count

    # The operator node should have one more child
    op_node = new_tree.nodes["op_nodes"][0]
    assert len(op_node.children) == 4  # Original 3 + new one

    # Verify the new child is a value node
    new_child = op_node.children[-1]
    assert isinstance(new_child, ValueNode)
    assert new_tree.nodes_count == len(new_tree.root.get_nodes())


def test_lose_branch_mutation(medium_tree):
    """Test removing a branch from the tree."""
    np.random.seed(42)  # For reproducibility

    # Apply mutation
    new_tree = lose_branch_mutation(medium_tree)

    # Verify the tree has fewer nodes
    assert new_tree is not medium_tree  # Should be a different tree (copy)
    assert new_tree.nodes_count < medium_tree.nodes_count

    # The structure should be altered, but the roots should be equivalent (same id)
    assert new_tree.root.id == medium_tree.root.id
    assert new_tree.nodes_count == len(new_tree.root.get_nodes())


def test_lose_branch_mutation_too_small_tree(simple_tree):
    """Test that lose_branch_mutation raises an error with a tree that's too small."""
    with pytest.raises(AssertionError, match="Tree is too small"):
        lose_branch_mutation(simple_tree)


def test_new_tree_from_branch_mutation(medium_tree):
    """Test creating a new tree from a branch."""
    np.random.seed(42)  # For reproducibility

    # Apply mutation
    new_tree = new_tree_from_branch_mutation(medium_tree)

    # Verify the new tree is different
    assert new_tree is not medium_tree
    assert isinstance(new_tree, Tree)

    # The new tree should contain only a value node as root
    assert isinstance(new_tree.root, ValueNode)
    assert new_tree.nodes_count == 1
    assert len(new_tree.nodes["value_nodes"]) == 1
    assert len(new_tree.nodes["op_nodes"]) == 0
    assert new_tree.nodes_count == len(new_tree.root.get_nodes())


def test_new_tree_from_branch_mutation_insufficient_nodes(simple_tree):
    """Test that new_tree_from_branch_mutation raises an error when there aren't enough value nodes."""
    with pytest.raises(AssertionError):
        new_tree_from_branch_mutation(simple_tree)


def test_get_allowed_mutations_simple_tree(simple_tree):
    """Test get_allowed_mutations with a simple tree."""
    mutations = get_allowed_mutations(simple_tree)

    # Should only include append_new_node_mutation
    assert len(mutations) == 1
    assert mutations[0] == append_new_node_mutation


def test_get_allowed_mutations_medium_tree(medium_tree):
    """Test get_allowed_mutations with a medium tree."""
    mutations = get_allowed_mutations(medium_tree)

    # Should include all three mutations
    assert len(mutations) == 3
    assert append_new_node_mutation in mutations
    assert lose_branch_mutation in mutations
    assert new_tree_from_branch_mutation in mutations


def test_append_new_node_mutation_custom_ids(medium_tree, models):
    """Test append_new_node_mutation with custom IDs."""
    # Use integer IDs that can be used as indices
    custom_ids = [0, 1, 2]

    # Count value nodes before mutation
    value_nodes_count_before = len(medium_tree.nodes["value_nodes"])

    # Apply mutation
    new_tree = append_new_node_mutation(medium_tree, models, custom_ids)

    # Count value nodes after mutation
    value_nodes_count_after = len(new_tree.nodes["value_nodes"])

    # Verify one node was added
    assert value_nodes_count_after == value_nodes_count_before + 1

    # Get the new node (the last one added should be the new one)
    new_node = new_tree.nodes["value_nodes"][-1]

    # The ID should be one of our indices
    assert new_node.id in [0, 1, 2]
    assert new_tree.nodes_count == len(new_tree.root.get_nodes())
    # assert 1 == 2, "Fail for now, add tests for actual structure, not only numbers of nodes"


def test_append_new_node_mutation_with_custom_operator(simple_tree, models, id_values):
    """Test append_new_node_mutation with a custom operator class."""
    # Define our test operators
    test_ops = (MeanNode,)

    # Apply mutation
    new_tree = append_new_node_mutation(simple_tree, models, id_values, allowed_ops=test_ops)

    # Verify the new operator is of the correct type
    assert len(new_tree.nodes["op_nodes"]) == 1
    assert isinstance(new_tree.nodes["op_nodes"][0], MeanNode)
    assert new_tree.nodes_count == len(new_tree.root.get_nodes())
