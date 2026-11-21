"""Build bounded fusion-tree libraries from fit-time provider correlations."""

from collections.abc import Mapping
from functools import lru_cache
from itertools import combinations

import numpy as np

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.node import ValueNode
from okapi.operators import MAX, MEAN, MIN
from okapi.population import choose_pareto_then_proximity
from okapi.random_state import use_rng
from okapi.tree import Tree


def _correlation_distances(columns):
    # Subtracting a reference before centering keeps constant columns exactly
    # zero. Rescaling before taking norms also handles tiny score differences.
    columns = np.asarray(columns, dtype=np.float64)
    centered = columns - columns[:1]
    scale = np.max(np.abs(centered), axis=0)
    np.divide(centered, scale, out=centered, where=scale > 0)
    centered -= centered.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    np.divide(centered, norm, out=centered, where=norm > 0)
    return 1 - np.clip(centered.T @ centered, -1, 1)


def _average_link_cuts(distances):
    """Cut an ordered pairwise dissimilarity matrix at one through four groups."""
    distances = np.array(distances, dtype=np.float64, copy=True)
    groups = [(i,) for i in range(len(distances))]
    cuts = {}
    while len(groups) > 1:
        if len(groups) <= 4:
            cuts[len(groups)] = tuple(groups)
        left, right = np.triu_indices(len(groups), k=1)
        closest = np.argmin(distances[left, right])
        i, j = left[closest], right[closest]
        size_i, size_j = len(groups[i]), len(groups[j])
        # Average linkage weights the two previous means by their sizes.
        merged = (size_i * distances[i] + size_j * distances[j]) / (size_i + size_j)
        distances[i, :] = distances[:, i] = merged
        distances = np.delete(np.delete(distances, j, axis=0), j, axis=1)
        groups[i] = tuple(sorted((*groups[i], *groups[j])))
        del groups[j]
    cuts[1] = tuple(groups)
    return cuts


def _symbolic_compositions(operands):
    """Enumerate symbolic expressions and their Boolean truth-table bit masks."""
    if len(operands) == 1:
        yield operands[0]
        return
    # The first group always belongs to the left subtree. This enumerates
    # each unordered bipartition once for both commutative operators.
    rest = range(1, len(operands))
    for count in range(len(operands) - 1):
        for chosen in combinations(rest, count):
            left_inputs = [operands[i] for i in (0, *chosen)]
            right_inputs = [operands[i] for i in rest if i not in chosen]
            for left, left_table in _symbolic_compositions(left_inputs):
                for right, right_table in _symbolic_compositions(right_inputs):
                    yield (MIN, left, right), left_table & right_table
                    yield (MAX, left, right), left_table | right_table


@lru_cache(maxsize=4)
def _cut_expressions(count):
    # The bit at position j is the output at Boolean assignment j. The plan
    # is independent of all predictions, group membership and labels.
    variables = []
    for operand in range(count):
        truth_table = sum(1 << row for row in range(1 << count) if row & (1 << operand))
        variables.append((operand, truth_table))
    return tuple(_symbolic_compositions(variables))


def _expression_root(expression, groups):
    if isinstance(expression, int):
        return groups[expression].copy_subtree()
    op, left, right = expression
    root = _expression_root(left, groups)
    root.add_child(op([_expression_root(right, groups)]))
    return root


def _binary_compositions(roots):
    """Yield every unordered binary min/max tree over the supplied group roots in historical order."""
    if len(roots) == 1:
        yield roots[0]
        return
    # This historical helper accepts arbitrary arity and yields lazily. It
    # needs only expression order, so avoid computing unbounded truth tables.
    operands = [(i, 0) for i in range(len(roots))]
    for expression, _ in _symbolic_compositions(operands):
        yield _expression_root(expression, roots)


def _templates_from_cuts(ids, values, cuts, shuffle_seed=None, *, deduplicate=False, eligible=None):
    """Build the min/max library for prespecified ordered provider groups."""
    order = np.arange(len(ids)) if shuffle_seed is None else np.random.default_rng(shuffle_seed).permutation(len(ids))
    trees = []
    for count in range(1, 5):
        roots = []
        for group in cuts[count]:
            leaves = [ValueNode(None, values[order[i]], ids[order[i]]) for i in group]
            root = leaves[0]
            if len(leaves) > 1:
                root.add_child(MEAN(leaves[1:]))
            roots.append(root)
        seen = set()
        for expression, truth_table in _cut_expressions(count):
            if deduplicate and truth_table in seen:
                continue
            tree = Tree(_expression_root(expression, roots))
            if eligible is None or eligible(tree):
                trees.append(tree)
                if deduplicate:
                    seen.add(truth_table)
    return trees


def correlation_templates(
    providers: Mapping[int | str, np.ndarray], *, shuffle_seed: int | None = None, deduplicate: bool = False,
) -> list[Tree]:
    """Build a library of up to 135 OKAPI trees using only binary fit predictions.

    Supply four or more row-aligned positive-class probability arrays, each
    shaped (N,) or (N, 1), with at least two rows. Fit values follow the first
    provider's layout. Supply a consistent shape when rebinding new predictions.
    Insertion order breaks linkage ties. Average linkage uses signed Pearson
    distance 1 - r and treats constant providers as uncorrelated.

    Cut the hierarchy at one through four groups. Average each group's scores,
    then enumerate every unordered binary min/max composition over groups.
    The four blocks contain 1, 2, 12 and 120 trees including equivalent
    functions. Every tree owns its nodes. A shuffle seed replaces provider
    assignments with one fixed permutation across all cuts after clustering.

    With deduplicate=True, keep only the first tree per Boolean truth table
    within each cut. Thresholding min/max commutes with AND/OR, so every
    discarded composition agrees at all finite real group scores when node
    postprocessing is the identity. A nonlinear per-node postprocessor can
    break equivalence between parenthesizations. The default returns all 135 trees.
    """
    return _correlation_templates(providers, shuffle_seed, deduplicate)


def _correlation_templates(providers, shuffle_seed, deduplicate, eligible=None):
    if len(providers) < 4:
        raise ValueError("At least four providers are required")
    ids = list(providers)
    arrays = [np.asarray(providers[key]) for key in ids]
    shape = arrays[0].shape
    for array in arrays:
        if array.ndim not in (1, 2) or (array.ndim == 2 and array.shape[1] != 1):
            raise ValueError("Expected binary positive-class scores shaped (N,) or (N, 1)")
        if array.shape[0] < 2 or array.shape[0] != shape[0]:
            raise ValueError("All providers must have the same length with at least two rows")
        if array.dtype.kind not in "biuf" or not np.all((array >= 0) & (array <= 1)):
            raise ValueError("Provider scores must be finite probabilities in [0, 1]")
    dtype = np.result_type(np.float32, *(array.dtype for array in arrays))
    values = [(array if array.shape == shape else array.reshape(shape)).astype(dtype, copy=False) for array in arrays]
    cuts = _average_link_cuts(_correlation_distances(np.column_stack(values)))
    return _templates_from_cuts(ids, values, cuts, shuffle_seed, deduplicate=deduplicate, eligible=eligible)


def seed_from_correlations(engine, *, shuffle_seed: int | None = None, deduplicate: bool = False) -> int:
    """Offer fit-only correlation templates to a NumPy OKAPI population.

    Call after constructing the engine and before training. Existing operator
    and size/depth restrictions filter proposals. Score each resident and
    feasible template once and apply the engine's usual survivor selection.
    The return value counts full tree evaluations. With no feasible proposal
    the population is left in place and the count is zero.

    The templates assume binary probabilities and node postprocessing that
    leaves these probabilities unchanged. Ordinary GP training proceeds with
    its usual selection after this explicit initialization. With deduplicate=True,
    score only the first feasible representative of each min/max truth table
    per cut. This preserves the attainable template scores under identity
    postprocessing. Candidate multiplicities and subsequent GP search can change.
    """
    if Backend.get_backend() is not NumpyBackend:
        raise ValueError("Correlation initialization requires the NumPy backend")

    def eligible(tree):
        return engine._within_caps(tree) and all(type(op) in engine.allowed_ops for op in tree.op_nodes)

    proposals = _correlation_templates(engine.train_tensors, shuffle_seed, deduplicate, eligible)
    if not proposals:
        return 0
    pool = engine.population + proposals
    with use_rng(engine.rng):
        scores = engine._selection_fitnesses(pool)
        survivors, fitnesses = choose_pareto_then_proximity(
            pool, scores, engine.objectives, engine.population_size, engine.minimize_node_count,
        )
    engine.population, engine.fitnesses = survivors, fitnesses
    return len(pool)
