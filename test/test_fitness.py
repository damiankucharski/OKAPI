import numpy as np
import pytest
import torch

from okapi.fitness import (
    accuracy_fitness,
    average_precision_fitness,
    roc_auc_score_fitness,
)
from okapi.node import ValueNode
from okapi.tree import Tree


@pytest.fixture
def simple_tree():
    """Create a simple tree with prediction values."""
    # Predictions: probabilities for binary classification
    predictions = np.array([0.9, 0.8, 0.3, 0.2, 0.7])
    node = ValueNode(None, predictions, "test_model")
    return Tree.create_tree_from_root(node)


@pytest.fixture
def ground_truth():
    """Ground truth labels for binary classification."""
    return np.array([1, 1, 0, 0, 1])


def test_average_precision_accepts_tree(simple_tree, ground_truth):
    """Test that average_precision_fitness accepts Tree objects."""
    score = average_precision_fitness(simple_tree, ground_truth, task="binary")
    assert 0 <= score <= 1


def test_average_precision_accepts_tensor(ground_truth):
    """Test that average_precision_fitness accepts Tensor directly."""
    predictions = torch.tensor([0.9, 0.8, 0.3, 0.2, 0.7])
    score = average_precision_fitness(predictions, ground_truth, task="binary")
    assert 0 <= score <= 1


def test_average_precision_same_result_tree_vs_tensor(simple_tree, ground_truth):
    """Test that passing Tree or Tensor gives same result."""
    tree_score = average_precision_fitness(simple_tree, ground_truth, task="binary")

    # Get tensor from tree
    tensor = simple_tree.evaluation
    tensor_score = average_precision_fitness(tensor, ground_truth, task="binary")

    np.testing.assert_almost_equal(tree_score, tensor_score)


def test_roc_auc_accepts_tree(simple_tree, ground_truth):
    """Test that roc_auc_score_fitness accepts Tree objects."""
    score = roc_auc_score_fitness(simple_tree, ground_truth, task="binary")
    assert 0 <= score <= 1


def test_roc_auc_accepts_tensor(ground_truth):
    """Test that roc_auc_score_fitness accepts Tensor directly."""
    predictions = torch.tensor([0.9, 0.8, 0.3, 0.2, 0.7])
    score = roc_auc_score_fitness(predictions, ground_truth, task="binary")
    assert 0 <= score <= 1


def test_roc_auc_same_result_tree_vs_tensor(simple_tree, ground_truth):
    """Test that passing Tree or Tensor gives same result."""
    tree_score = roc_auc_score_fitness(simple_tree, ground_truth, task="binary")

    tensor = simple_tree.evaluation
    tensor_score = roc_auc_score_fitness(tensor, ground_truth, task="binary")

    np.testing.assert_almost_equal(tree_score, tensor_score)


def test_accuracy_accepts_tree(simple_tree, ground_truth):
    """Test that accuracy_fitness accepts Tree objects."""
    score = accuracy_fitness(simple_tree, ground_truth, task="binary")
    assert 0 <= score <= 1


def test_accuracy_accepts_tensor(ground_truth):
    """Test that accuracy_fitness accepts Tensor directly."""
    predictions = torch.tensor([0.9, 0.8, 0.3, 0.2, 0.7])
    score = accuracy_fitness(predictions, ground_truth, task="binary")
    assert 0 <= score <= 1


def test_accuracy_same_result_tree_vs_tensor(simple_tree, ground_truth):
    """Test that passing Tree or Tensor gives same result."""
    tree_score = accuracy_fitness(simple_tree, ground_truth, task="binary")

    tensor = simple_tree.evaluation
    tensor_score = accuracy_fitness(tensor, ground_truth, task="binary")

    np.testing.assert_almost_equal(tree_score, tensor_score)


def test_fitness_with_numpy_array(ground_truth):
    """Test that fitness functions work with numpy arrays."""
    predictions = np.array([0.9, 0.8, 0.3, 0.2, 0.7])

    ap_score = average_precision_fitness(predictions, ground_truth, task="binary")
    roc_score = roc_auc_score_fitness(predictions, ground_truth, task="binary")
    acc_score = accuracy_fitness(predictions, ground_truth, task="binary")

    assert 0 <= ap_score <= 1
    assert 0 <= roc_score <= 1
    assert 0 <= acc_score <= 1
