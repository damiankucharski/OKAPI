from typing import List, cast

import numpy as np
import pytest

from giraffe.node import ValueNode
from giraffe.pareto import maximize
from giraffe.population import choose_pareto, choose_pareto_then_proximity
from giraffe.tree import Tree


# Helper function to create trees for testing
def create_mock_tree(id_value):
    """Create a tree with specified ID for testing"""
    # Create a simple root node with a tensor value
    root = ValueNode(children=None, value=np.array([0.5]), id=id_value)

    # Create a regular Tree instance
    tree = Tree(root)

    return tree


class TestSelectionFunctions:
    @pytest.fixture(autouse=True)
    def setup_method(self, monkeypatch):
        """Set up test data before each test method execution"""
        # Define the mock node counts for each tree ID
        self.mock_node_counts = {
            "tree1": 10,  # Medium complexity
            "tree2": 5,  # Low complexity
            "tree3": 15,  # High complexity
            "tree4": 8,  # Medium complexity
            "tree5": 3,  # Low complexity
        }

        # Create a dictionary to store the original nodes_count method
        original_nodes_count = Tree.nodes_count

        # Create a mock nodes_count method that returns the mock value based on tree ID
        def mock_nodes_count(tree_instance):
            tree_id = str(tree_instance.root.id)
            if tree_id in self.mock_node_counts:
                return self.mock_node_counts[tree_id]
            # Fall back to the original implementation if the tree ID is not mocked
            return original_nodes_count.__get__(tree_instance)

        # Patch the nodes_count property at the class level
        monkeypatch.setattr(Tree, "nodes_count", property(mock_nodes_count))

        # Create sample trees with different IDs
        self.trees = [
            create_mock_tree("tree1"),  # Good fitness, medium complexity
            create_mock_tree("tree2"),  # Good fitness, low complexity
            create_mock_tree("tree3"),  # Medium fitness, high complexity
            create_mock_tree("tree4"),  # Low fitness, medium complexity
            create_mock_tree("tree5"),  # Medium fitness, low complexity
        ]

        # Define fitness values for each tree
        self.fitnesses = np.array([0.9, 0.85, 0.7, 0.5, 0.65]).reshape(-1, 1)
        self.objectives = [maximize]

    def test_choose_pareto(self):
        """Test that choose_pareto selects trees based on Pareto optimality"""
        # Test selecting up to 3 trees based on Pareto front
        selected_trees, selected_fitnesses = choose_pareto(cast(List[Tree], self.trees), self.fitnesses, 3, self.objectives)

        # In our test case, the Pareto-optimal trees should be:
        # - tree2: High fitness (0.85), low nodes (5)
        # - tree1: Highest fitness (0.9), medium nodes (10)
        # - tree5: Medium fitness (0.65), lowest nodes (3)

        # Check correct trees were selected
        selected_ids = [tree.root.id for tree in selected_trees]
        assert "tree1" in selected_ids
        assert "tree2" in selected_ids
        assert "tree5" in selected_ids

        # Verify fitness values match the selected trees
        for tree, fitness in zip(selected_trees, selected_fitnesses, strict=True):
            expected_fitness = self.fitnesses[self.trees.index(tree)]
            assert np.array_equal(fitness, expected_fitness), "fitnesses do not match"

        # Test with smaller selection limit than Pareto front size
        selected_trees, selected_fitnesses = choose_pareto(cast(List[Tree], self.trees), self.fitnesses, 2, self.objectives)

        # Should select the 2 best by fitness from the Pareto front
        assert len(selected_trees) == 2
        selected_ids = [tree.root.id for tree in selected_trees]
        assert "tree1" in selected_ids  # Highest fitness
        assert (
            "tree2" in selected_ids
        )  # Second highest fitness, because more pareto trees than to select. Therefore proximity based limit is triggered.

    def test_choose_pareto_then_proximity(self):
        """Test choose_pareto_then_proximity selects Pareto-optimal trees first, then fills with best remaining"""
        # Test selecting 4 trees (3 Pareto + 1 from remaining)
        selected_trees, selected_fitnesses = choose_pareto_then_proximity(cast(List[Tree], self.trees), self.fitnesses, self.objectives, 4, True)

        # First check we got the right number of trees
        assert len(selected_trees) == 4
        assert len(selected_fitnesses) == 4

        # Check all Pareto-optimal trees are included
        selected_ids = [tree.root.id for tree in selected_trees]
        assert "tree1" in selected_ids
        assert "tree2" in selected_ids
        assert "tree5" in selected_ids

        # The fourth tree should be the best remaining by fitness (tree3)
        assert "tree3" in selected_ids

        # Test with limit smaller than Pareto front
        selected_trees, selected_fitnesses = choose_pareto_then_proximity(cast(List[Tree], self.trees), self.fitnesses, self.objectives, 2, True)

        # Should select the 2 best by fitness from the Pareto front
        assert len(selected_trees) == 2
        selected_ids = [tree.root.id for tree in selected_trees]
        assert "tree1" in selected_ids  # Highest fitness
        assert "tree2" in selected_ids  # Second highest fitness

        # Test with limit larger than available trees
        selected_trees, selected_fitnesses = choose_pareto_then_proximity(cast(List[Tree], self.trees), self.fitnesses, self.objectives, 10, True)

        # Should include all trees
        assert len(selected_trees) == 5
        assert len(selected_fitnesses) == 5
