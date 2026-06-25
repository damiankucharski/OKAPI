import numpy as np
from loguru import logger

from okapi.tree import Tree
from okapi.utils import _euclidean_distances


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
            f"Tournament size ({tournament_size}) is large relative to the population size ({len(fitnesses)})."
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


def _node_xover_metrics(tree: Tree):
    """Integer per-node geometry used to predict an offspring's depth/size *without*
    building it: depth-from-root, subtree height, subtree node-count, and the tree's
    max leaf-depth EXCLUDING each node's subtree. All cheap (no tensor ops); only
    computed on the capped path, where trees are small by construction."""
    nodes = tree.root.get_nodes()  # BFS: parents precede children
    depth_of: dict = {}
    for n in nodes:
        depth_of[n] = 1 if n.parent is None else depth_of[n.parent] + 1
    leaves = [n for n in nodes if not n.children]

    def _ancestors_inclusive(node):
        seen, cur = set(), node
        while cur is not None:
            seen.add(cur)
            cur = cur.parent
        return seen

    leaf_ancestors = {leaf: _ancestors_inclusive(leaf) for leaf in leaves}
    height_of: dict = {}
    count_of: dict = {}
    for n in reversed(nodes):  # children precede parents
        if not n.children:
            height_of[n], count_of[n] = 1, 1
        else:
            height_of[n] = 1 + max(height_of[c] for c in n.children)
            count_of[n] = 1 + sum(count_of[c] for c in n.children)
    depth_excluding: dict = {}
    for n in nodes:
        depth_excluding[n] = max((depth_of[leaf] for leaf in leaves if n not in leaf_ancestors[leaf]), default=0)
    return depth_of, height_of, count_of, depth_excluding


def _legal_crossover_pair(tree1: Tree, tree2: Tree, nodes_type: str, max_depth, max_nodes):
    """Uniformly pick a ``(node1, node2)`` pair of ``nodes_type`` whose *both* offspring
    respect the caps, or ``None`` if no such pair exists. Swapping the subtree at node1
    (in tree1) with the one at node2 (in tree2) gives offspring-1 depth
    ``max(excl1[n1], depth1[n1]-1 + height2[n2])`` and node-count ``N1 - count1[n1] +
    count2[n2]`` (symmetric for offspring-2). For ``value_nodes`` the root x root pair is
    always present (each offspring becomes a whole within-caps parent), so the engine
    path -- which always allows value_nodes -- never comes up empty."""
    depth1, height1, count1, excl1 = _node_xover_metrics(tree1)
    depth2, height2, count2, excl2 = _node_xover_metrics(tree2)
    n1_total, n2_total = tree1.nodes_count, tree2.nodes_count
    legal = []
    for n1 in tree1.nodes[nodes_type]:
        for n2 in tree2.nodes[nodes_type]:
            off1_depth = max(excl1[n1], depth1[n1] - 1 + height2[n2])
            off2_depth = max(excl2[n2], depth2[n2] - 1 + height1[n1])
            if max_depth is not None and (off1_depth > max_depth or off2_depth > max_depth):
                continue
            off1_nodes = n1_total - count1[n1] + count2[n2]
            off2_nodes = n2_total - count2[n2] + count1[n1]
            if max_nodes is not None and (off1_nodes > max_nodes or off2_nodes > max_nodes):
                continue
            legal.append((n1, n2))
    if not legal:
        return None
    return legal[np.random.randint(len(legal))]


def crossover(tree1: Tree, tree2: Tree, node_type=None, max_depth=None, max_nodes=None):
    """
    Performs crossover between two parent trees to produce two offspring trees.

    Crossover works by selecting a node from each parent tree and swapping the subtrees
    rooted at those nodes, producing two offspring that mix both parents.

    Cap enforcement is deterministic, not generate-and-reject: with ``max_depth`` /
    ``max_nodes`` set, the crossover points are drawn only from pairs whose *both*
    offspring stay within the caps (``_legal_crossover_pair``). Because the root x root
    value-node swap is always legal, a usable pair always exists, so the search never
    needs retries or a parent-copy fallback. With both caps ``None`` the original uniform
    ``get_random_node`` draw is used unchanged.

    Args:
        tree1: First parent tree
        tree2: Second parent tree
        node_type: Type of nodes to consider for crossover points ('value_nodes' or 'op_nodes').
                   If None, a random suitable type will be chosen.
        max_depth: Optional hard depth cap (see above).
        max_nodes: Optional hard node-count cap (see above).

    Returns:
        Tuple of two new Tree objects created by crossover

    Raises:
        ValueError: If node_type is 'op_nodes' but one or both trees don't have operator nodes
    """
    logger.info("Performing crossover between two trees")

    both_have_ops = (len(tree1.nodes["op_nodes"]) > 0) and (len(tree2.nodes["op_nodes"]) > 0)
    if node_type is None:
        allowable_node_types = ["value_nodes"]
        if both_have_ops:
            allowable_node_types.append("op_nodes")
            logger.debug("Both trees have operator nodes, including them in potential crossover points")
        else:
            logger.debug("At least one tree has no operator nodes, using only value nodes for crossover")
    else:
        if node_type == "op_nodes" and not both_have_ops:
            logger.error("Node type was chosen to be operator nodes but there are no operator nodes in at least one of the parents")
            raise ValueError("Node type was chosen to be operator nodes but there are not operator nodes in at least one of the parents")
        allowable_node_types = [node_type]

    logger.debug("Creating copies of parent trees")
    tree1, tree2 = tree1.copy(), tree2.copy()

    if max_depth is None and max_nodes is None:
        # Uncapped: original behaviour, byte-for-byte (uniform type then uniform node).
        nodes_type = np.random.choice(allowable_node_types)
        node1, node2 = tree1.get_random_node(nodes_type), tree2.get_random_node(nodes_type)
    else:
        # Capped: try the allowable type(s) in random order, take the first that has a
        # cap-legal pair. value_nodes always does (root x root), so this resolves.
        np.random.shuffle(allowable_node_types)
        node1 = node2 = None
        for nodes_type in allowable_node_types:
            pair = _legal_crossover_pair(tree1, tree2, nodes_type, max_depth, max_nodes)
            if pair is not None:
                node1, node2 = pair
                break
        if node1 is None:
            # Unreachable on the engine path (value root x root is always legal); only a
            # tight explicit op-only request can land here -> no-op (return parent copies).
            logger.trace("No cap-legal crossover pair for the requested type; returning parent copies")
            return tree1, tree2
    logger.debug(f"Selected nodes: {node1} from tree1, {node2} from tree2")

    logger.debug("Creating copies of subtrees")
    branch1, branch2 = node1.copy_subtree(), node2.copy_subtree()

    logger.debug("Swapping subtrees between trees")
    tree1.replace_at(node1, branch2)
    tree2.replace_at(node2, branch1)
    # Update node lists without computing evaluations (defer to fitness calculation)
    tree1.update_nodes()
    tree2.update_nodes()

    logger.info(f"Crossover complete, created two new trees with {tree1.nodes_count} and {tree2.nodes_count} nodes")
    return tree1, tree2
