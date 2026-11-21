"""Fit one close/far threshold against the downstream binary Brier loss."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from operator import index
from typing import Generic, TypeVar

import numpy as np

from okapi.interventions import operator_paths
from okapi.lib_types import Tensor
from okapi.node import CloseThresholdNode, FarThresholdNode, OperatorNode, ThresholdNode
from okapi.tree import Tree

SolverResult = TypeVar("SolverResult")


@dataclass(frozen=True)
class ThresholdRefinement(Generic[SolverResult]):
    tree: Tree
    initial_loss: float
    loss: float
    realized_loss: float
    full_tree_evaluations: int
    solver_result: SolverResult


def _as_array(value):
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        # NumPy has no native bfloat16. Widening preserves each operand value;
        # the normal gate still runs at the original precision.
        if str(tensor.dtype) == "torch.bfloat16":
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def _probabilities(values, name):
    if values.dtype.kind not in "biuf" or not np.all((values >= 0) & (values <= 1)):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")


def _threshold(value, name):
    scalar = np.asarray(value)
    if scalar.ndim != 0 or scalar.dtype.kind not in "iuf" or not 0 <= scalar.item() <= 1:
        raise ValueError(f"{name} must be a finite scalar threshold in [0, 1]")
    return float(scalar)


def _row_costs(output, labels):
    values = _as_array(output)
    if values.shape not in ((len(labels),), (len(labels), 1)):
        raise ValueError("The final tree must return one probability per label, shaped (N,) or (N, 1)")
    _probabilities(values, "Tree output")
    return (values.ravel().astype(np.float64, copy=False) - labels) ** 2


class _SelectedSlot(OperatorNode):
    """Intervene after ordinary operand assembly and before ordinary postprocessing."""

    def __init__(self, children, rows):
        super().__init__(children)
        self.rows = rows
        self.slot = 0
        self.operands = None

    def op(self, x):
        if self.operands is None:
            values = _as_array(x)
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[:, :, 0]
            if values.shape != (len(self.children) + 1, self.rows):
                raise ValueError("Threshold operands must contain one positive-class probability per row")
            _probabilities(values, "Threshold operands")
            self.operands = values.T.copy()
        return x[self.slot]


def refine_threshold_coordinate(
    tree: Tree, providers: Mapping[int | str, Tensor], labels, *, operator_index: int,
    fit_threshold: Callable[..., SolverResult],
) -> ThresholdRefinement[SolverResult]:
    """Refit exactly tree.op_nodes[operator_index] on a private, fit-bound copy.

    Supply aligned, same-layout (N,) or (N, 1) provider tensors for the active
    backend and binary labels. The tree and its postprocessing must be pure,
    deterministic and rowwise: forcing one operand on a row must not affect
    other rows. Batch-normalizing, stateful, or random programs violate the
    loss-table contract. All inputs, captured operands and final predictions
    must be finite probabilities.

    The solver receives NumPy matrices operands[N, M] and downstream_losses[N, M]
    and keyword arguments close and original; it must return a scalar .threshold
    in [0, 1]. Slot zero is the actual preceding value of the gate's parent.
    Each loss column scores the final tree after forcing that gate's slot for
    every row. Gate postprocessing, following updates and ancestors run normally.

    The M+2 full evaluations score the incumbent, all M interventions, and the
    proposed threshold through the normal backend. Accept only a strictly
    smaller realized Brier; ties retain the input threshold. Native precision
    or selector ties can produce a different realized_loss than a solver's
    projected loss. loss describes the returned tree. The unmodified solver
    result exposes its own diagnostics, e.g. result.solver_result.n_thresholds.
    Solver-side candidate assessments are separate from the full-tree count.
    """
    if not callable(fit_threshold):
        raise TypeError("fit_threshold must be callable")
    if isinstance(operator_index, (bool, np.bool_)):
        raise TypeError("operator_index must be an integer")
    i = index(operator_index)
    if not 0 <= i < len(tree.op_nodes):
        raise IndexError("operator_index outside tree.op_nodes")
    selected = tree.op_nodes[i]
    if type(selected) not in (ThresholdNode, CloseThresholdNode, FarThresholdNode):
        raise ValueError("Selected operator must be a built-in close/far ThresholdNode")
    _threshold(selected.threshold, "Original threshold")
    if not isinstance(selected.close, (bool, np.bool_)):
        raise ValueError("Selected gate must have a boolean close/far mode")

    y = _as_array(labels)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or not y.size or y.dtype.kind not in "biuf" or not np.all((y == 0) | (y == 1)):
        raise ValueError("Expected a nonempty binary label vector shaped (N,) or (N, 1)")
    y = y.astype(np.float64, copy=False)

    # Tree.copy rebuilds breadth-first node lists; a caller's current op_nodes
    # order can differ after a replacement. A structural path denotes the
    # selected occurrence unambiguously in both copies.
    source_paths = {id(op): p for p, op in operator_paths(tree)}
    if id(selected) not in source_paths:
        raise ValueError("Selected operator is detached from the tree")
    path = source_paths[id(selected)]
    candidate = tree.copy()
    candidate.mutation_chance = tree.mutation_chance
    inputs = {}
    layout = None
    for node in candidate.value_nodes:
        if node.id not in inputs:
            if node.id not in providers:
                raise ValueError(f"Missing fit provider {node.id!r}")
            value = providers[node.id]
            array = _as_array(value)
            if array.shape not in ((len(y),), (len(y), 1)) or (layout is not None and array.shape != layout):
                raise ValueError("Providers must have matching (N,) or (N, 1) shapes with one row per label")
            _probabilities(array, "Providers")
            layout = array.shape
            inputs[node.id] = value
        node.value = inputs[node.id]
    copied_operators = dict(operator_paths(candidate))
    # Keep the public coordinate numbering for successive refinement calls,
    # even when a prior replace_at changed it relative to traversal order.
    if {id(op) for op in tree.op_nodes} == set(source_paths):
        candidate.nodes["op_nodes"] = [copied_operators[source_paths[id(op)]] for op in tree.op_nodes]
    target = copied_operators[path]
    original = target.threshold
    initial = float(_row_costs(candidate.predict(), y).mean())

    probe = candidate.copy()
    gate = dict(operator_paths(probe))[path]
    forced = _SelectedSlot(gate.children, len(y))
    # Replace exactly this child occurrence; custom siblings can override
    # equality, so an equality-based list.index search would be ambiguous.
    forced.parent = gate.parent
    forced.parent.children[path[-1]] = forced
    gate.parent = None
    probe.update_nodes()
    slots = len(forced.children) + 1
    losses = np.empty((len(y), slots), dtype=np.float64)
    for slot in range(slots):
        forced.slot = slot
        losses[:, slot] = _row_costs(probe.predict(), y)
    result = fit_threshold(forced.operands, losses, close=target.close, original=original)
    target.threshold = _threshold(getattr(result, "threshold", None), "Solver threshold")
    realized = float(_row_costs(candidate.predict(), y).mean())
    if realized >= initial:
        target.threshold = original
    return ThresholdRefinement(candidate, initial, min(initial, realized), realized, slots + 2, result)
