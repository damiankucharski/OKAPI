"""Propose a binary CLOSE/FAR graft at one terminal occurrence of a fit tree."""

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from operator import index

import numpy as np

from okapi.lib_types import Tensor
from okapi.node import CloseThresholdNode, FarThresholdNode, ValueNode
from okapi.threshold_refinement import _as_array, _probabilities, _row_costs, refine_threshold_coordinate
from okapi.tree import Tree


@dataclass(frozen=True)
class ThresholdGraftResult:
    tree: Tree
    initial_loss: float
    loss: float
    full_tree_evaluations: int = 1
    threshold_candidates: int = 0
    solver_events: int = 0
    solver_row_events: int = 0
    n_proposals: int = 0
    site_path: tuple[int, ...] | None = None
    anchor_id: int | str | None = None
    operand_order: tuple[int | str, ...] = ()
    selected_operand: int | str | None = None
    selected_direction: str | None = None
    selected_threshold: float | None = None


def _fit_inputs(tree, providers, labels):
    y = _as_array(labels)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or not y.size or y.dtype.kind not in "biuf" or not np.all((y == 0) | (y == 1)):
        raise ValueError("Expected a nonempty binary label vector shaped (N,) or (N, 1)")
    y = y.astype(np.float64, copy=False)

    layout = None
    checked = {}

    def fit_provider(provider_id):
        nonlocal layout
        if provider_id not in checked:
            if provider_id not in providers:
                raise ValueError(f"Missing fit provider {provider_id!r}")
            tensor = providers[provider_id]
            values = _as_array(tensor)
            if values.shape not in ((len(y),), (len(y), 1)) or (layout is not None and values.shape != layout):
                raise ValueError("Providers must have matching (N,) or (N, 1) shapes with one row per label")
            _probabilities(values, "Providers")
            layout = values.shape
            checked[provider_id] = tensor
        return checked[provider_id]

    incumbent = tree.copy()
    incumbent.mutation_chance = tree.mutation_chance
    for value_node in incumbent.value_nodes:
        value_node.value = fit_provider(value_node.id)
    return incumbent, y, fit_provider


def _terminal_sites(tree, max_nodes, max_depth):
    if max_nodes is not None and tree.nodes_count + 2 > max_nodes:
        return []
    if max_depth is not None and tree.depth > max_depth:
        return []
    sites = []
    queue = deque([(tree.root, ())])
    while queue:
        node, path = queue.popleft()
        # The root is level one. Adding an operator and a value adds two levels.
        if max_depth is not None and len(path) + 3 > max_depth:
            continue
        if isinstance(node, ValueNode) and not node.children:
            sites.append((path, node.id))
        for i, child in enumerate(node.children):
            queue.append((child, (*path, i)))
    return sites


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        count = index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if count < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return count


def threshold_graft_variant(
    tree: Tree, site_path: Sequence[int], *, reset_threshold: bool = False, duplicate_anchor: bool = False,
) -> Tree:
    """Copy a graft and optionally reset its cutoff or duplicate its anchor.

    Pass the accepted graft's site_path, which addresses the formerly terminal
    ValueNode (the root for an empty path). Its sole update must be a native
    CLOSE or FAR gate with one simple ValueNode operand. The two flags independently
    set the cutoff to 0.5 and replace that operand's provider ID and tensor with
    the anchor's. Remaining node parameters and provider bindings are retained.

    The duplicated operand produces the ungrafted terminal's exact activation
    with identity postprocessing. The gate still runs normal postprocessing,
    so a nonidentity function can change that activation. Construction neither
    evaluates the tree nor draws random numbers.
    """
    if not isinstance(reset_threshold, (bool, np.bool_)) or not isinstance(duplicate_anchor, (bool, np.bool_)):
        raise TypeError("reset_threshold and duplicate_anchor must be boolean")
    if not isinstance(site_path, Sequence) or isinstance(site_path, (str, bytes)):
        raise TypeError("site_path must be a sequence of child indices")
    path = tuple(_integer(i, "site_path index", 0) for i in site_path)
    anchor = tree.root
    for child_index in path:
        if child_index >= len(anchor.children):
            raise ValueError("site_path falls outside the tree")
        anchor = anchor.children[child_index]
    if not isinstance(anchor, ValueNode) or len(anchor.children) != 1:
        raise ValueError("site_path must identify a ValueNode with one grafted update")
    gate = anchor.children[0]
    if type(gate) not in (CloseThresholdNode, FarThresholdNode) or len(gate.children) != 1:
        raise ValueError("The graft must be a native binary CLOSE or FAR gate")
    operand = gate.children[0]
    if type(operand) is not ValueNode or operand.children:
        raise ValueError("The graft operand must be a terminal ValueNode")

    variant = tree.copy()
    variant.mutation_chance = tree.mutation_chance
    copied_anchor = variant.root
    for child_index in path:
        copied_anchor = copied_anchor.children[child_index]
    copied_gate = copied_anchor.children[0]
    if reset_threshold:
        copied_gate.threshold = 0.5
    if duplicate_anchor:
        copied_operand = copied_gate.children[0]
        copied_operand.id, copied_operand.value = copied_anchor.id, copied_anchor.value
    return variant


def propose_threshold_graft(
    tree: Tree, providers: Mapping[int | str, Tensor], labels, *, seed: int, search_operands: bool,
    fit_threshold: Callable | None = None, max_nodes: int | None = None, max_depth: int | None = None,
) -> ThresholdGraftResult:
    """Score one local graft at a seeded terminal using fit-set binary Brier loss.

    Providers share the active backend and an (N,) or (N, 1) layout. Pass a
    deterministic, rowwise tree and postprocessing as for coordinate refinement.
    A local NumPy generator draws a feasible terminal from breadth-first order,
    then permutes the other IDs from mapping order. ID values are opaque. Each
    chosen operand is tried with CLOSE then FAR at .5; search_operands chooses
    between the first operand and the entire permutation. The returned site
    path indexes children from the root, which itself has path ().

    With fit_threshold, each two-input gate uses refine_threshold_coordinate
    and costs four full-tree predictions. Its solver must also return counts
    n_thresholds, n_events and n_row_events for the separate search ledger.
    Without it, each proposal costs one prediction. Either mode scores the
    incumbent once. Equal losses retain the incumbent or first strict winner.
    selected_* describes an accepted graft only. The returned native tree is
    privately copied and bound to fit tensors, including when no edit wins.
    """
    if not isinstance(search_operands, (bool, np.bool_)):
        raise TypeError("search_operands must be a boolean")
    if max_nodes is not None:
        max_nodes = _integer(max_nodes, "max_nodes", 1)
    if max_depth is not None:
        max_depth = _integer(max_depth, "max_depth", 1)
    if fit_threshold is not None and not callable(fit_threshold):
        raise TypeError("fit_threshold must be callable")
    incumbent, y, fit_provider = _fit_inputs(tree, providers, labels)
    initial_loss = float(_row_costs(incumbent.predict(), y).mean())
    result = ThresholdGraftResult(incumbent, initial_loss, initial_loss)
    sites = _terminal_sites(incumbent, max_nodes, max_depth)
    if not sites:
        return result

    rng = np.random.default_rng(seed)
    path, anchor_id = sites[rng.integers(len(sites))]
    others = [provider_id for provider_id in providers if provider_id != anchor_id]
    order = tuple(others[i] for i in rng.permutation(len(others)))
    result = replace(result, site_path=path, anchor_id=anchor_id, operand_order=order)
    evaluations, n_proposals, threshold_candidates, events, row_events = 1, 0, 0, 0, 0
    for operand in (order if search_operands else order[:1]):
        tensor = fit_provider(operand)
        for direction, gate_type in (("close", CloseThresholdNode), ("far", FarThresholdNode)):
            candidate = incumbent.copy()
            candidate.mutation_chance = incumbent.mutation_chance
            terminal = candidate.root
            for child_index in path:
                terminal = terminal.children[child_index]
            gate = gate_type([ValueNode(None, tensor, operand)], 0.5)
            terminal.add_child(gate)
            candidate.update_nodes()
            threshold = gate.threshold
            n_proposals += 1
            if fit_threshold is None:
                loss = float(_row_costs(candidate.predict(), y).mean())
                evaluations += 1
            else:
                gate_index = next(i for i, node in enumerate(candidate.op_nodes) if node is gate)
                refined = refine_threshold_coordinate(candidate, providers, y, operator_index=gate_index,
                                                      fit_threshold=fit_threshold)
                candidate, loss = refined.tree, refined.loss
                threshold = candidate.op_nodes[gate_index].threshold
                evaluations += refined.full_tree_evaluations
                threshold_candidates += _integer(refined.solver_result.n_thresholds, "n_thresholds", 0)
                events += _integer(refined.solver_result.n_events, "n_events", 0)
                row_events += _integer(refined.solver_result.n_row_events, "n_row_events", 0)
            if loss < result.loss:
                result = replace(result, tree=candidate, loss=loss, selected_operand=operand,
                                 selected_direction=direction, selected_threshold=threshold)
    return replace(result, n_proposals=n_proposals, full_tree_evaluations=evaluations,
                   threshold_candidates=threshold_candidates, solver_events=events, solver_row_events=row_events)
