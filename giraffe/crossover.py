import numpy as np
from loguru import logger

from giraffe.tree import Tree
from giraffe.utils import _euclidean_distances


def tournament_selection_indexes(fitnesses: np.ndarray, tournament_size: int = 5, optimal_point: np.ndarray | None = None) -> np.ndarray:
    """
    Selects parent indices for crossover using tournament selection.

    In tournament selection, a subset of individuals (of size tournament_size) is randomly
    selected from the population, and the one with the highest fitness is chosen as a parent.
    This process is repeated to select the second parent.

    Args:
        fitnesses: Array of fitness values for the entire population
        tournament_size: Number of individuals to include in each tournament

    Returns:
        Array with indices of the two selected parents

    Raises:
        ValueError: If tournament_size is too large relative to population size
    """
    logger.debug(f"Running tournament selection with tournament size {tournament_size}")

    if tournament_size > (len(fitnesses)):
        logger.error(f"Tournament size {tournament_size} is too large for population size {len(fitnesses)}")
        raise ValueError(f"Size of the tournament should be at most equal to number of participans but {len(fitnesses)=} and {tournament_size=}")

    if len(fitnesses) < (2 * tournament_size):
        logger.warning(
            f"Tournament size ({tournament_size}), is small related to the population size ({len(fitnesses)})."
            "The population should be at least twice as large as tournament for more stable parent selection"
        )

    if optimal_point is None:
        optimal_point = np.ones(shape=(fitnesses.shape[-1],))
    assert len(optimal_point.shape) == 1 and fitnesses.shape[-1] == optimal_point.shape[-1], "Shapes for fitnesses and optimal point do not match"

    selected: list | np.ndarray = []
    for _ in range(2):
        candidates_idx = np.random.choice(np.arange(len(fitnesses)), size=(tournament_size,), replace=False)
        candidates_fitnesses = fitnesses[candidates_idx]
        distances = _euclidean_distances(candidates_fitnesses, optimal_point)
        best_idx = np.argmin(distances)
        assert isinstance(selected, list)
        selected.append(candidates_idx[best_idx])
    selected = np.array(selected)

    assert selected.shape == (2,)

    logger.debug(f"Selected parent indices: {selected}")
    return selected


def crossover(tree1: Tree, tree2: Tree, node_type=None):
    """
    Performs crossover between two parent trees to produce two offspring trees.

    Crossover works by selecting a random node from each parent tree and swapping
    the subtrees rooted at those nodes. This creates two new offspring trees that
    contain genetic material from both parents.

    Args:
        tree1: First parent tree
        tree2: Second parent tree
        node_type: Type of nodes to consider for crossover points ('value_nodes' or 'op_nodes').
                   If None, a random suitable type will be chosen.

    Returns:
        Tuple of two new Tree objects created by crossover

    Raises:
        ValueError: If node_type is 'op_nodes' but one or both trees don't have operator nodes
    """
    logger.info("Performing crossover between two trees")

    if node_type is None:
        allowable_node_types = ["value_nodes"]  # TODO: this may be worth refactoring along with "get_random_node" to not use string but types instead

        if (len(tree1.nodes["op_nodes"]) > 0) & (len(tree2.nodes["op_nodes"]) > 0):
            allowable_node_types.append("op_nodes")
            logger.debug("Both trees have operator nodes, including them in potential crossover points")
        else:
            logger.debug("At least one tree has no operator nodes, using only value nodes for crossover")

        nodes_type = np.random.choice(allowable_node_types)
        logger.debug(f"Randomly selected node type for crossover: {nodes_type}")
    else:
        if node_type == "op_nodes" and not ((len(tree1.nodes["op_nodes"]) > 0) & (len(tree2.nodes["op_nodes"]) > 0)):
            logger.error("Node type was chosen to be operator nodes but there are no operator nodes in at least one of the parents")
            raise ValueError("Node type was chosen to be operator nodes but there are not operator nodes in at least one of the parents")
        nodes_type = node_type
        logger.debug(f"Using specified node type for crossover: {nodes_type}")

    logger.debug("Creating copies of parent trees")
    tree1, tree2 = tree1.copy(), tree2.copy()

    logger.debug("Selecting random nodes for crossover")
    node1, node2 = tree1.get_random_node(nodes_type), tree2.get_random_node(nodes_type)
    logger.debug(f"Selected nodes: {node1} from tree1, {node2} from tree2")

    logger.debug("Creating copies of subtrees")
    branch1, branch2 = node1.copy_subtree(), node2.copy_subtree()

    logger.debug("Swapping subtrees between trees")
    tree1.replace_at(node1, branch2).recalculate()
    tree2.replace_at(node2, branch1).recalculate()

    logger.info(f"Crossover complete, created two new trees with {tree1.nodes_count} and {tree2.nodes_count} nodes")
    return tree1, tree2
