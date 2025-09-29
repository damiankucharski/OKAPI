from pathlib import Path
from typing import Self, Tuple, cast

import numpy as np
from loguru import logger

from giraffe.globals import BACKEND as B
from giraffe.globals import DEVICE
from giraffe.lib_types import Tensor
from giraffe.node import Node, OperatorNode, ValueNode, check_if_both_types_same_node_variant
from giraffe.utils import Pickle


class Tree:
    """
    Represents a computational tree structure for model ensemble composition.

    The Tree class is a central component in GIRAFFE, representing a hierarchical structure
    of nodes that define how different models are combined. Each tree has a ValueNode as its root,
    and may contain multiple ValueNodes and OperatorNodes arranged in a tree structure.

    ValueNodes contain tensor data (model predictions), while OperatorNodes define operations
    to combine these predictions (such as mean, min, max, weighted mean). The tree's evaluation
    produces a combined prediction by recursively applying these operations.

    Trees can be manipulated through various operations like pruning, appending, and replacing
    nodes, making them suitable for evolutionary algorithms where trees evolve over generations.

    Attributes:
        root: The root node of the tree (must be a ValueNode)
        nodes: Dictionary containing lists of all value nodes and operator nodes in the tree
        mutation_chance: Probability of mutation for this tree during evolution
    """

    def __init__(self, root: ValueNode, mutation_chance=0.1):
        self.root = root
        logger.debug(f"Creating new tree with root: {root}")

        if isinstance(self.root, OperatorNode):
            logger.error("Cannot initialize tree with OperatorNode as root")
            raise Exception("Cannot get evaluation of tree with OpNode as root")

        self.nodes: dict[str, list] = {"value_nodes": [], "op_nodes": []}
        self.mutation_chance = mutation_chance
        self.update_nodes()
        logger.trace(f"Tree initialized with {len(self.nodes['value_nodes'])} value nodes and {len(self.nodes['op_nodes'])} operator nodes")

    def update_nodes(self):
        """
        Update the internal collections of nodes in the tree.

        This method traverses the tree and categorizes all nodes into value nodes and operator nodes,
        updating the internal `nodes` dictionary.
        """
        logger.debug("Updating tree node collections")
        self.nodes = {"value_nodes": [], "op_nodes": []}
        root_nodes = self.root.get_nodes()
        for node in root_nodes:
            if isinstance(node, ValueNode):
                self.nodes["value_nodes"].append(node)
            else:
                self.nodes["op_nodes"].append(node)
        logger.trace(f"Updated nodes: {len(self.nodes['value_nodes'])} value nodes, {len(self.nodes['op_nodes'])} operator nodes")

    @staticmethod
    def create_tree_from_root(root: ValueNode, mutation_chance=0.1):
        """
        Create a new tree with the given root node.

        Args:
            root: The ValueNode to use as the root of the new tree
            mutation_chance: Probability of mutation for the new tree

        Returns:
            A new Tree instance
        """
        logger.debug(f"Creating tree from root node with mutation chance: {mutation_chance}")
        tree = Tree(root, mutation_chance)
        return tree

    @property
    def evaluation(self):
        """
        Calculate and return the evaluation of the tree.

        The evaluation is the result of recursively applying all operations
        in the tree, starting from the root node.

        Returns:
            The tensor resulting from evaluating the tree
        """
        # WARNING: This may not make sense for cases other than binary classification (Squeezing)
        # return B.squeeze(self.root.evaluation if self.root.evaluation is not None else self.root.calculate())
        return self.root.evaluation if self.root.evaluation is not None else self.root.calculate()

    @property
    def nodes_count(self):
        """
        Count the total number of nodes in the tree.

        Returns:
            The sum of value nodes and operator nodes
        """
        return len(self.nodes["value_nodes"]) + len(self.nodes["op_nodes"])

    def _clean_evals(self):
        """
        Reset the cached evaluation results for all value nodes in the tree.

        This forces recalculation of node evaluations when the tree structure changes.
        """
        logger.debug("Clearing cached evaluations for all value nodes")
        for node in self.nodes["value_nodes"]:
            node.evaluation = None

    def _clean_values_and_evals(self):
        for value_node in self.nodes["value_nodes"]:
            value_node.value = value_node.evaluation = None

    def recalculate(self):
        """
        Force recalculation of the tree evaluation.

        This method clears any cached evaluations and triggers a fresh calculation.
        It also updates the nodes dictionary

        Returns:
            The newly calculated evaluation of the tree
        """
        logger.debug("Recalculating tree evaluation")
        self._clean_evals()
        self.update_nodes()
        evaluation = self.evaluation
        logger.trace("Tree recalculation complete")
        return evaluation

    def copy(self):
        """
        Create a deep copy of the tree.

        Returns:
            A new Tree instance that is a deep copy of the current tree
        """
        logger.debug("Creating deep copy of tree")
        root_copy: ValueNode = cast(ValueNode, self.root.copy_subtree())
        return Tree.create_tree_from_root(root_copy)

    def prune_at(self, node: Node) -> Node:
        """
        Remove a node and its subtree from the tree.

        This method removes the specified node and all its descendants from the tree.
        If the node is the only child of an operator node, that operator node will
        also be pruned.

        Args:
            node: The node to prune from the tree

        Returns:
            The pruned node (which is no longer part of the tree). If parent was pruned, the parent will be returned.

        Raises:
            ValueError: If the node is not found in the tree or if attempting to prune the root node
        """
        logger.debug(f"Pruning node from tree: {node}")

        if node not in self.nodes["value_nodes"] and node not in self.nodes["op_nodes"]:
            logger.error(f"Attempted to prune node not in tree: {node}")
            raise ValueError("Node not found in tree")

        if node.parent is None:
            logger.error("Cannot prune root node")
            raise ValueError("Cannot prune root node")

        if isinstance(node.parent, OperatorNode) and (
            len(node.parent.children) < 2
        ):  # if only child of op node is to be pruned, remove the parent instead
            logger.debug(f"Node is the only child of operator node, pruning parent: {node.parent}")
            return self.prune_at(node.parent)

        subtree_nodes = node.get_nodes()
        node_count = len(subtree_nodes)

        logger.debug(f"Removing {node_count} nodes in subtree")
        for subtree_node in subtree_nodes:
            if isinstance(subtree_node, ValueNode):
                self.nodes["value_nodes"].remove(subtree_node)
            else:
                self.nodes["op_nodes"].remove(subtree_node)

        node.parent.remove_child(node)
        logger.debug("Pruning complete, clearing cached evaluations")
        self._clean_evals()
        return node

    def append_after(self, node: Node, new_node: Node):
        """
        Append a new node as a child of an existing node.

        The new node must be of a different type than the existing node
        (i.e., value nodes can only append operator nodes and vice versa).

        Args:
            node: The existing node to which the new node will be appended
            new_node: The new node to append

        Raises:
            ValueError: If the node is not found in the tree or if attempting to append
                       a node of the same type
        """
        logger.debug(f"Appending node {new_node} after {node}")

        if node not in self.nodes["value_nodes"] and node not in self.nodes["op_nodes"]:
            logger.error(f"Attempted to append to node not in tree: {node}")
            raise ValueError("Node not found in tree")

        if check_if_both_types_same_node_variant(type(node), type(new_node)):
            logger.error(f"Cannot append node of same type: {type(node).__name__} and {type(new_node).__name__}")
            raise ValueError("Cannot append node of the same type")

        subtree_nodes = new_node.get_nodes()
        logger.debug(f"Adding {len(subtree_nodes)} nodes from subtree")

        for subtree_node in subtree_nodes:
            if isinstance(subtree_node, ValueNode):
                self.nodes["value_nodes"].append(subtree_node)
            else:
                self.nodes["op_nodes"].append(subtree_node)

        node.add_child(new_node)
        logger.debug("Append complete, clearing cached evaluations")
        self._clean_evals()

    def replace_at(self, at: Node, replacement: Node) -> Self:
        """
        Replace a node in the tree with another node.

        The replacement node must be of the same type as the node being replaced.
        This operation preserves the parent-child relationships.

        Args:
            at: The node to be replaced
            replacement: The new node that will replace the existing node

        Returns:
            Self reference to allow method chaining

        Raises:
            AssertionError: If the replacement node is not of the same type as the node being replaced
        """
        assert (isinstance(replacement, ValueNode) and isinstance(at, ValueNode)) or (
            isinstance(replacement, OperatorNode) and isinstance(at, OperatorNode)
        ), "Replacement node must be of the same parent type (ValueNode or OperatorNode) as the node being replaced"
        at_parent = at.parent

        if at_parent is None:
            assert isinstance(self.root, ValueNode), "Root must be a value node"
            assert isinstance(replacement, ValueNode), "Replacement for root must be a value node"
            logger.warning("Node at replacement is root node")
            self.root = replacement
        else:
            at_parent.replace_child(at, replacement)

        if isinstance(at, ValueNode):
            self.nodes["value_nodes"].remove(at)
            self.nodes["value_nodes"].append(replacement)
        else:
            self.nodes["op_nodes"].remove(at)
            self.nodes["op_nodes"].append(replacement)

        self._clean_evals()
        return self

    def get_random_node(self, nodes_type: str | None = None, allow_root=True, allow_leaves=True):
        """
        Get a random node from the tree based on specified constraints.

        Args:
            nodes_type: Optional type of nodes to consider ('value_nodes' or 'op_nodes')
                       If None, a random type will be chosen
            allow_root: Whether to allow selecting the root node
            allow_leaves: Whether to allow selecting leaf nodes

        Returns:
            A randomly selected node that satisfies the constraints

        Raises:
            ValueError: If no node satisfying the constraints is found
        """
        if self.root.children == []:
            if allow_root:
                if nodes_type is None or nodes_type == "value_nodes":
                    return self.root
                else:
                    raise ValueError("Tree has only root node and nodes_type is not value_nodes")
            else:
                raise ValueError("Tree has only root node and allow_root is set to False")

        if nodes_type is not None:
            assert nodes_type in ("value_nodes", "op_nodes"), f'Unsupported node type "{nodes_type}" selected.'
            nodes_types = [
                nodes_type,
            ]
        else:
            nodes_types = list(np.random.permutation(["op_nodes", "value_nodes"]))

        for nodes_type in nodes_types:
            assert nodes_type is not None, "Nodes type cannot be None"
            order = np.arange(len(self.nodes[nodes_type]))
            for i in order:
                node = self.nodes[nodes_type][i]
                if (allow_leaves or node.children != []) and (allow_root or node != self.root):
                    return node
        raise ValueError("No node found that complies to the constraints")

    @property
    def unique_value_node_ids(self):
        """
        Get the unique IDs of all value nodes in the tree.

        Returns:
            A list of unique IDs from all value nodes
        """
        return list(set([node.id for node in self.nodes["value_nodes"]]))

    @property
    def value_nodes(self):
        return self.nodes["value_nodes"]

    @property
    def op_nodes(self):
        return self.nodes["op_nodes"]

    def save_tree_architecture(self, output_path):  # TODO: needs adjustment for weighted node
        """
        Save the tree's architecture to a file.

        This method creates a copy of the tree with tensor values removed
        and saves it to the specified path using pickle serialization.

        Args:
            output_path: Path where the tree architecture will be saved
        """
        logger.info(f"Saving tree architecture to {output_path}")
        copy_tree = self.copy()
        copy_tree._clean_values_and_evals()

        Pickle.save(output_path, copy_tree)
        logger.debug("Tree architecture saved successfully")

    @staticmethod
    def load_tree_architecture(architecture_path) -> "Tree":  # TODO: needs adjusted for weighted node
        """
        Load a tree architecture from a file.

        Args:
            architecture_path: Path to the saved tree architecture file

        Returns:
            The loaded Tree object without tensor values
        """
        logger.info(f"Loading tree architecture from {architecture_path}")
        tree = Pickle.load(architecture_path)
        logger.debug("Tree architecture loaded successfully")
        return tree

    def _load_tensors_from_path(self, preds_directory):
        current_tensors = {}
        preds_directory = Path(preds_directory)
        for value_node in self.nodes["value_nodes"]:
            node_id = value_node.id
            if node_id not in current_tensors:
                logger.debug(f"Loading tensor for node ID: {node_id}")
                current_tensors[node_id] = B.load(preds_directory / str(node_id), DEVICE)
            else:
                logger.trace(f"Using pre-loaded tensor for node ID: {node_id}")
        return current_tensors

    def _load_tensors_to_tree(self, preds_directory, current_tensors):
        if preds_directory is not None:
            preds_directory = Path(preds_directory)
            loaded_tensors = self._load_tensors_from_path(preds_directory)
            current_tensors.update(loaded_tensors)
        for value_node in self.nodes["value_nodes"]:
            node_id = value_node.id
            value_node.value = current_tensors[node_id]
        return current_tensors

    def do_pred_on_another_tensors(
        self, preds_directory: None | str | Path = None, current_tensors: None | dict = None, return_tree=False
    ) -> tuple[None | Tensor, "Tree"] | Tensor:
        assert not all(
            [current_tensors is not None, preds_directory is not None]
        ), "Either preds directory or current tensors needs to be set, not both"
        assert any(
            [current_tensors is not None, preds_directory is not None]
        ), "Either preds directory or current tensors needs to be set, none was set"

        current_tensors = {}
        copy_tree = self.copy()
        copy_tree._clean_values_and_evals()
        current_tensors = copy_tree._load_tensors_to_tree(preds_directory, current_tensors)
        if return_tree:
            return copy_tree.evaluation, copy_tree

        return copy_tree.evaluation

    @staticmethod
    def load_tree(architecture_path, preds_directory, tensors={}) -> Tuple["Tree", dict]:
        """
        Load a complete tree with tensor values from files.

        This method loads a tree architecture and then loads the associated tensor
        values for each value node from the specified directory.

        Args:
            architecture_path: Path to the saved tree architecture file
            preds_directory: Directory containing the tensor files
            tensors: Optional dictionary of pre-loaded tensors

        Returns:
            A tuple containing:
            - The loaded Tree object with tensor values
            - A dictionary of all tensors used in the tree
        """
        logger.info(f"Loading complete tree from {architecture_path} with tensors from {preds_directory}")
        logger.debug(f"Starting with {len(tensors)} pre-loaded tensors")

        current_tensors = {}
        current_tensors.update(tensors)  # tensors argument is mutable and we do not want to modify it

        loaded = Tree.load_tree_architecture(architecture_path)
        current_tensors = loaded._load_tensors_to_tree(preds_directory, current_tensors)

        logger.info(
            f"Tree loaded successfully with {len(loaded.nodes['value_nodes'])} value nodes and {len(loaded.nodes['op_nodes'])} operator nodes"
        )
        return loaded, current_tensors

    def __repr__(self):
        """
        Get a string representation of the tree.

        Returns:
            A string representation formed by concatenating the code of all nodes
        """
        return "_".join(node.code for node in self.root.get_nodes())
