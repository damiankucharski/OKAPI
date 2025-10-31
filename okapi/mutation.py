from typing import Callable, Sequence, Type

import numpy as np
from loguru import logger

from okapi.lib_types import Tensor
from okapi.node import MeanNode, OperatorNode, ValueNode
from okapi.tree import Tree


def append_new_node_mutation(
    tree: Tree, models: Sequence[Tensor], ids: None | Sequence[str | int] = None, allowed_ops: tuple[Type[OperatorNode], ...] = (MeanNode,), **kwargs
):
    """
    Mutation that adds a new node to the tree.

    This mutation randomly selects an existing node in the tree and appends a new node as its
    child. If the selected node is a ValueNode, a new OperatorNode is created as an intermediary,
    and the new ValueNode is added as its child. If the selected node is an OperatorNode,
    a new ValueNode is directly appended to it.

    Args:
        tree: The tree to mutate
        models: Sequence of tensor models that can be used as values for the new ValueNode
        ids: Optional sequence of identifiers for the models. If None, indices will be used
        allowed_ops: Tuple of OperatorNode types that can be used when creating a new operator node
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        A new Tree with the mutation applied
    """
    logger.debug("Applying append_new_node_mutation")
    tree = tree.copy()

    if ids is None:
        ids = list(range(len(models)))
        logger.trace("Using indices as IDs for models")
    else:
        assert len(models) == len(ids)
        logger.trace(f"Using provided IDs, confirmed length match: {len(ids)}")

    idx_model = np.random.randint(len(ids))
    logger.debug(f"Selected model ID: {ids[idx_model]}")
    node = tree.get_random_node()
    logger.debug(f"Selected random node for mutation: {node}")

    val_node: ValueNode = ValueNode([], models[idx_model], ids[idx_model])
    logger.trace(f"Created new value node with ID: {ids[idx_model]}")

    if isinstance(node, ValueNode):
        random_op: Type[OperatorNode] = np.random.choice(np.asarray(allowed_ops))
        logger.debug(f"Selected random operator type: {random_op.__name__}")
        op_node: OperatorNode = random_op.create_node([val_node])
        logger.debug("Appending operator node with value node child after selected node")
        tree.append_after(node, op_node)
    else:
        logger.debug("Appending value node directly to operator node")
        tree.append_after(node, val_node)

    logger.info(f"Append node mutation complete, new tree has {tree.nodes_count} nodes")
    return tree


def lose_branch_mutation(tree: Tree, **kwargs):
    """
    Mutation that removes a branch from the tree.

    This mutation randomly selects a non-root, non-leaf node in the tree and removes it along
    with all its descendants, effectively pruning that branch from the tree.

    Args:
        tree: The tree to mutate
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        A new Tree with the mutation applied

    Raises:
        AssertionError: If the tree has fewer than 3 nodes
    """
    logger.debug("Applying lose_branch_mutation")
    tree = tree.copy()

    if tree.nodes_count < 3:
        logger.error(f"Cannot apply lose_branch_mutation - tree is too small: {tree.nodes_count} nodes")
        assert tree.nodes_count >= 3, "Tree is too small"

    node = tree.get_random_node(allow_leaves=False, allow_root=False)
    logger.debug(f"Selected node for pruning: {node}")

    pruned = tree.prune_at(node)
    logger.info(f"Pruned branch with {len(pruned.get_nodes())} nodes, tree now has {tree.nodes_count} nodes")

    return tree


def new_tree_from_branch_mutation(tree: Tree, **kwargs):
    """
    Mutation that creates a new tree from a branch of the existing tree.

    This mutation randomly selects a non-root ValueNode, removes it from the tree along with
    its descendants, and creates a new tree with the removed node as its root.

    Args:
        tree: The tree to mutate
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        A new Tree created from the selected branch

    Raises:
        AssertionError: If the tree has only one ValueNode
    """
    assert len(tree.nodes["value_nodes"]) > 1, "Tree must have more than one value node"

    logger.debug("Applying new_tree_from_branch_mutation")
    tree = tree.copy()

    node = tree.get_random_node(nodes_type="value_nodes", allow_leaves=True, allow_root=False)
    logger.debug(f"Selected value node for creating new tree: {node}")

    _ = tree.prune_at(node)  # this may return parent op node, so we still want to use the original node.
    logger.debug("Pruned node and its subtree to create new tree")

    assert isinstance(node, ValueNode)
    new_tree = Tree.create_tree_from_root(node)

    logger.info(f"Created new tree from branch with {new_tree.nodes_count} nodes")
    return new_tree


def get_allowed_mutations(tree):
    """
    Determines which mutation operations are valid for a given tree.

    This function checks the tree's structure and size to determine which mutations
    can be safely applied without violating constraints.

    Args:
        tree: The tree to analyze

    Returns:
        A list of mutation functions that are valid for the given tree
    """
    logger.debug(f"Determining allowed mutations for tree with {tree.nodes_count} nodes")
    allowed_mutations: list[Callable] = [
        append_new_node_mutation,
    ]

    if tree.nodes_count >= 3:
        logger.trace("Tree is large enough for lose_branch_mutation")
        allowed_mutations.append(lose_branch_mutation)
    if len(tree.nodes["value_nodes"]) > 1:
        logger.trace("Tree has enough value nodes for new_tree_from_branch_mutation")
        allowed_mutations.append(new_tree_from_branch_mutation)

    logger.debug(f"Found {len(allowed_mutations)} allowed mutation types")
    return allowed_mutations
