from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from loguru import logger

from giraffe.lib_types import Tensor
from giraffe.node import ValueNode
from giraffe.pareto import minimize, paretoset, sort_by_optimal_point_proximity
from giraffe.tree import Tree


def initialize_individuals(tensors_dict: Dict[str, Tensor], n: int, exclude_ids=tuple()) -> List[Tree]:
    """
    Initialize a population of individuals (trees) from a dictionary of tensors.

    This function creates simple trees, each with a root node containing a different tensor
    from the provided dictionary. The tensors are selected randomly from the dictionary.

    Args:
        tensors_dict: Dictionary mapping model IDs to their tensor representations
        n: Number of individuals (trees) to create
        exclude_ids: Optional tuple of model IDs to exclude from selection

    Returns:
        List of initialized Tree objects

    Raises:
        Exception: If n is greater than the number of available tensors after exclusions
    """
    logger.info(f"Initializing {n} individuals")
    logger.debug(f"Available tensors: {len(tensors_dict)}, excluded IDs: {len(exclude_ids)}")

    order = np.arange(len(tensors_dict))
    np.random.shuffle(order)
    logger.trace("Shuffled tensor order")

    ids_list = list(tensors_dict.keys())
    tensors_list = list(tensors_dict.values())

    new_trees = []
    count = 0
    for idx in order:
        _id = ids_list[idx]
        tensor = tensors_list[idx]
        if count >= n:
            break
        if _id in exclude_ids:
            logger.trace(f"Skipping excluded ID: {_id}")
            continue

        logger.debug(f"Creating tree with tensor ID: {_id}")
        root: ValueNode = ValueNode(children=None, value=tensor, id=_id)
        tree = Tree.create_tree_from_root(root)
        new_trees.append(tree)
        count += 1

    if count < n:
        logger.error(f"Could not generate enough individuals. Requested: {n}, generated: {count}")
        raise Exception("Could not generate as many examples")

    logger.info(f"Successfully initialized {len(new_trees)} individuals")
    return new_trees


def choose_pareto(trees: List[Tree], fitnesses: np.ndarray, n: int, objectives: Sequence[Callable[[float, float], bool]], minimize_node_count=True):
    """
    Select up to n trees based on Pareto optimality.
    Optimizes for:
    - Maximizing fitness
    - Minimizing number of nodes in the tree

    Args:
        trees: List of Tree objects
        fitnesses: Array of fitness values for each tree
        n: Maximum number of trees to select

    Returns:
        List of selected trees and their corresponding fitness values
    """
    logger.debug(f"Selecting up to {n} Pareto-optimal trees from population of {len(trees)}")

    objectives = list(objectives)
    if minimize_node_count:
        objectives.append(minimize)
        sizes = np.array([tree.nodes_count for tree in trees]).reshape(-1, 1)
        objective_array = np.concatenate([fitnesses, sizes], axis=1)
    else:
        objective_array = fitnesses

    # Get Pareto-optimal mask using maximize for fitness and minimize for nodes count
    pareto_mask = paretoset(objective_array, objectives)
    pareto_count = np.sum(pareto_mask)
    logger.debug(f"Found {pareto_count} Pareto-optimal trees")

    # Get indices of Pareto-optimal trees
    pareto_indices = np.where(pareto_mask)[0]

    # If we have more Pareto-optimal trees than n, select the n with highest fitness
    if len(pareto_indices) > n:
        logger.debug(f"Too many Pareto-optimal trees ({len(pareto_indices)}), selecting top {n} by proximity")
        if minimize_node_count:
            objectives = objectives[:-1]
        _, sorted_indices = sort_by_optimal_point_proximity(fitnesses, objectives)
        selected_indices = sorted_indices[:n]
    else:
        selected_indices = pareto_indices
        logger.debug(f"Using all {len(selected_indices)} Pareto-optimal trees")

    # Return selected trees and their fitnesses, for now allow for duplicates with regards to fitrnesses

    selected_fitnesses = fitnesses[selected_indices]
    # uniques_mask = first_uniques_mask(selected_fitnesses)

    # selected_indices = selected_indices[uniques_mask]

    selected_trees = [trees[i] for i in selected_indices]
    # selected_fitnesses = fitnesses[selected_indices]

    if len(selected_trees) > 0:
        logger.debug(f"Selected {len(selected_trees)} trees with fitness range: {selected_fitnesses.min():.4f} - {selected_fitnesses.max():.4f}")
    else:
        logger.warning("No trees selected in Pareto optimization")

    return selected_trees, selected_fitnesses


def choose_n_by_proximity(
    trees: List[Tree], fitnesses: np.ndarray, objectives: Sequence[Callable[[float, float], bool]], n: int
) -> Tuple[List[Tree], np.ndarray]:
    _, sorted_indices = sort_by_optimal_point_proximity(fitnesses, objectives)

    proximity_remaining_trees: List[Tree] = [trees[i] for i in sorted_indices[:n]]
    proximity_remaining_fitnesses = fitnesses[sorted_indices[:n]]
    return proximity_remaining_trees, proximity_remaining_fitnesses


def choose_pareto_then_proximity(
    trees: List[Tree], fitnesses: np.ndarray, objectives: Sequence[Callable[[float, float], bool]], n: int, minimize_node_count=True
) -> Tuple[List[Tree], np.ndarray]:
    logger.info(f"Selecting {n} trees using Pareto-then-sorted strategy")

    # Get all Pareto-optimal trees without limiting the number
    # Internal implementation of choose_pareto uses a limit, so we use a large number
    # to effectively get all Pareto trees
    all_pareto_trees, all_pareto_fitnesses = choose_pareto(trees, fitnesses, len(trees), objectives, minimize_node_count)
    logger.debug(f"Found {len(all_pareto_trees)} Pareto-optimal trees")

    # If we have more Pareto-optimal trees than n, select the n with highest fitness
    if len(all_pareto_trees) > n:
        logger.debug(f"Too many Pareto trees ({len(all_pareto_trees)}), selecting top {n}")
        return choose_n_by_proximity(all_pareto_trees, all_pareto_fitnesses, objectives, n)

    # If we have exactly n Pareto trees, return them
    if len(all_pareto_trees) == n:
        logger.debug(f"Exactly {n} Pareto trees, returning all of them")
        return all_pareto_trees, all_pareto_fitnesses

    # We need to fill the remainder with sorted trees
    remaining_slots = n - len(all_pareto_trees)
    logger.debug(f"Need {remaining_slots} more trees to reach target of {n}")

    # Create a list of non-Pareto trees by excluding Pareto trees
    pareto_trees_set = set(all_pareto_trees)
    non_pareto_trees = []
    non_pareto_fitnesses = []

    for i, tree in enumerate(trees):
        if tree not in pareto_trees_set:
            non_pareto_trees.append(tree)
            non_pareto_fitnesses.append(fitnesses[i])

    logger.debug(f"Found {len(non_pareto_trees)} non-Pareto trees")
    non_pareto_fitnesses_np = np.array(non_pareto_fitnesses)
    _, sorted_indices = sort_by_optimal_point_proximity(non_pareto_fitnesses_np, objectives)
    proximity_remaining_trees = [non_pareto_trees[i] for i in sorted_indices[:remaining_slots]]
    proximity_remaining_fitnesses = non_pareto_fitnesses_np[sorted_indices[:remaining_slots]]

    selected_trees = all_pareto_trees + proximity_remaining_trees
    selected_fitnesses = np.concatenate([all_pareto_fitnesses, proximity_remaining_fitnesses])

    logger.info(
        f"Total selection: {len(selected_trees)} trees ({len(all_pareto_trees)} Pareto\
        + {len(proximity_remaining_trees)} by proximity to optimal optimization point)"
    )

    assert len(selected_fitnesses) == selected_fitnesses.shape[0]

    return selected_trees, selected_fitnesses
