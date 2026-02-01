import numpy as np
import pytest

from okapi.mutation import append_new_node_mutation, get_allowed_mutations, lose_branch_mutation, mutate_parameters, new_tree_from_branch_mutation
from okapi.node import CloseThresholdNode, FarThresholdNode, MeanNode, OperatorNode, ValueNode, WeightedMeanNode
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


# ==================== Parameter Mutation Tests ====================


@pytest.fixture
def weighted_mean_tree(models):
    """Creates a tree with a WeightedMeanNode."""
    root = ValueNode(None, models[0], "model1")
    child1 = ValueNode(None, models[1], "model2")
    child2 = ValueNode(None, models[2], "model3")

    # Create WeightedMeanNode with known weights
    wmn = WeightedMeanNode([child1, child2], weights=[0.5, 0.25, 0.25])
    root.add_child(wmn)

    return Tree.create_tree_from_root(root)


@pytest.fixture
def threshold_tree(models):
    """Creates a tree with a CloseThresholdNode."""
    root = ValueNode(None, models[0], "model1")
    child1 = ValueNode(None, models[1], "model2")

    # Create CloseThresholdNode with known threshold
    thn = CloseThresholdNode([child1], threshold=0.5)
    root.add_child(thn)

    return Tree.create_tree_from_root(root)


class TestWeightedMeanNodeMutateParams:
    """Tests for WeightedMeanNode.mutate_params()"""

    def test_weights_change_after_mutation(self, models):
        """Weights should change after mutation."""
        np.random.seed(42)
        child = ValueNode(None, models[1], "model2")
        wmn = WeightedMeanNode([child], weights=[0.6, 0.4])
        original_weights = wmn._weights.copy()

        wmn.mutate_params(mutation_strength=0.1)

        assert wmn._weights != original_weights

    def test_weights_sum_to_one_after_mutation(self, models):
        """Weights must still sum to 1 after mutation."""
        np.random.seed(42)
        child1 = ValueNode(None, models[1], "model2")
        child2 = ValueNode(None, models[2], "model3")
        wmn = WeightedMeanNode([child1, child2], weights=[0.5, 0.3, 0.2])

        for _ in range(10):  # Multiple mutations
            wmn.mutate_params(mutation_strength=0.2)
            assert np.isclose(sum(wmn._weights), 1.0)

    def test_weights_stay_positive_after_mutation(self, models):
        """All weights must remain positive after mutation."""
        np.random.seed(42)
        child = ValueNode(None, models[1], "model2")
        wmn = WeightedMeanNode([child], weights=[0.99, 0.01])  # One very small weight

        for _ in range(20):  # Many mutations to test edge cases
            wmn.mutate_params(mutation_strength=0.3)
            assert all(w > 0 for w in wmn._weights)

    def test_larger_strength_causes_larger_changes(self, models):
        """Larger mutation_strength should cause larger average changes."""
        np.random.seed(42)

        # Small strength
        child1 = ValueNode(None, models[1], "model2")
        wmn_small = WeightedMeanNode([child1], weights=[0.5, 0.5])
        original_small = wmn_small._weights.copy()
        wmn_small.mutate_params(mutation_strength=0.01)
        change_small = sum(abs(a - b) for a, b in zip(wmn_small._weights, original_small))

        # Large strength
        np.random.seed(42)  # Same seed for fair comparison
        child2 = ValueNode(None, models[1], "model2")
        wmn_large = WeightedMeanNode([child2], weights=[0.5, 0.5])
        original_large = wmn_large._weights.copy()
        wmn_large.mutate_params(mutation_strength=0.5)
        change_large = sum(abs(a - b) for a, b in zip(wmn_large._weights, original_large))

        assert change_large > change_small


class TestThresholdNodeMutateParams:
    """Tests for ThresholdNode.mutate_params()"""

    def test_threshold_changes_after_mutation(self, models):
        """Threshold should change after mutation."""
        np.random.seed(42)
        child = ValueNode(None, models[1], "model2")
        thn = CloseThresholdNode([child], threshold=0.5)
        original_threshold = thn.threshold

        thn.mutate_params(mutation_strength=0.1)

        assert thn.threshold != original_threshold

    def test_threshold_stays_in_valid_range(self, models):
        """Threshold must stay in [0, 1] after mutation."""
        np.random.seed(42)

        # Test near lower bound
        child1 = ValueNode(None, models[1], "model2")
        thn_low = CloseThresholdNode([child1], threshold=0.05)
        for _ in range(20):
            thn_low.mutate_params(mutation_strength=0.3)
            assert 0.0 <= thn_low.threshold <= 1.0

        # Test near upper bound
        child2 = ValueNode(None, models[1], "model2")
        thn_high = FarThresholdNode([child2], threshold=0.95)
        for _ in range(20):
            thn_high.mutate_params(mutation_strength=0.3)
            assert 0.0 <= thn_high.threshold <= 1.0


class TestMutateParametersFunction:
    """Tests for the mutate_parameters() function."""

    def test_returns_new_tree(self, weighted_mean_tree):
        """mutate_parameters should return a new tree, not modify in place."""
        original_tree = weighted_mean_tree
        mutated_tree = mutate_parameters(original_tree, mutation_strength=0.1)

        assert mutated_tree is not original_tree

    def test_mutates_all_parametrized_nodes(self, models):
        """All parametrized nodes should be mutated."""
        np.random.seed(42)

        # Create tree with multiple parametrized nodes
        root = ValueNode(None, models[0], "model1")
        child1 = ValueNode(None, models[1], "model2")
        child2 = ValueNode(None, models[2], "model3")

        wmn = WeightedMeanNode([child1], weights=[0.6, 0.4])
        thn = CloseThresholdNode([child2], threshold=0.5)

        root.add_child(wmn)
        root.add_child(thn)

        tree = Tree.create_tree_from_root(root)

        # Get original values
        original_wmn_weights = tree.nodes["op_nodes"][0]._weights.copy()
        original_thn_threshold = tree.nodes["op_nodes"][1].threshold

        mutated_tree = mutate_parameters(tree, mutation_strength=0.1)

        # Both should have changed
        assert mutated_tree.nodes["op_nodes"][0]._weights != original_wmn_weights
        assert mutated_tree.nodes["op_nodes"][1].threshold != original_thn_threshold

    def test_mean_node_unaffected(self, medium_tree):
        """MeanNode has no parameters, should be unaffected."""
        np.random.seed(42)
        original_code = medium_tree.nodes["op_nodes"][0].code

        mutated_tree = mutate_parameters(medium_tree, mutation_strength=0.1)

        # MeanNode code should be unchanged (it's just "MN")
        assert mutated_tree.nodes["op_nodes"][0].code == original_code == "MN"
