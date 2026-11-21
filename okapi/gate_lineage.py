"""Track the inheritance of individual native threshold gates during tree evolution."""

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

from okapi.interventions import Path, operator_paths
from okapi.node import CloseThresholdNode, FarThresholdNode, ThresholdNode
from okapi.threshold_refinement import _threshold
from okapi.tree import Tree

_GATE_TYPES = (ThresholdNode, CloseThresholdNode, FarThresholdNode)


@dataclass(frozen=True)
class GateOccurrence:
    path: Path
    direction: str
    threshold: float
    operand_ids: tuple[int | str, ...]


def _token_string(token):
    if not isinstance(token, str):
        raise TypeError("token must be a string")
    # Store only the plain string value, even for a str subclass with extra state.
    return str.__str__(token)


def _gate_at(gates_by_path, gate_path):
    path = tuple(gate_path)
    if any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 for i in path):
        raise ValueError("gate_path must contain nonnegative integer child indices")
    gate = gates_by_path.get(path)
    if type(gate) not in _GATE_TYPES:
        raise ValueError("gate_path must identify a built-in threshold gate")
    return gate


def mark_gate_origin(tree: Tree, gate_path: Path, *, token: str) -> Tree:
    """Return a private copy with one origin token set on the native gate at gate_path.

    The path contains child indices from the root. The caller supplies a distinct
    token for each origin to distinguish. Re-marking replaces only this occurrence's
    prior token on the returned copy. An independently constructed gate has no origin.
    Marking consumes no random draws and leaves predictions and structural keys intact.
    """
    token = _token_string(token)
    gate = _gate_at(dict(operator_paths(tree)), gate_path)

    marked = tree.copy()
    marked.mutation_chance = tree.mutation_chance
    copies = {id(a): b for a, b in zip(tree.root.get_nodes(), marked.root.get_nodes(), strict=True)}
    copies[id(gate)]._okapi_gate_origin = token
    # Tree.copy rebuilds node lists breadth-first. Valid source lists may have
    # a different order after edits; keep that order for index-based sampling.
    for name, nodes in marked.nodes.items():
        ordered = [copies.get(id(node)) for node in tree.nodes[name]]
        if len(ordered) == len(nodes) and {id(node) for node in ordered} == {id(node) for node in nodes}:
            marked.nodes[name] = ordered
    return marked


def gate_descendants(tree: Tree, *, token: str) -> tuple[GateOccurrence, ...]:
    """Report every gate that actually inherited the token, in structural path order.

    Native copies and crossover preserve a marked occurrence. Parameter edits
    and changing its operands preserve its origin; pruning or replacing the gate
    removes that occurrence. Copying material into multiple branches can produce
    several matches in one tree. Each record reports the current direction,
    full-precision threshold, parent provider ID and ordered direct child IDs.
    The incoming parent value may have earlier updates, and each child can have
    its own subtree, so identical operand IDs can describe different computations.
    """
    token = _token_string(token)
    occurrences = []
    for path, gate in operator_paths(tree):
        if type(gate) in _GATE_TYPES and gate._okapi_gate_origin == token:
            occurrences.append(GateOccurrence(
                path, "close" if gate.close else "far", gate.threshold,
                (gate.parent.id, *(child.id for child in gate.children)),
            ))
    return tuple(occurrences)


def tree_with_gate_cutoff(tree: Tree, paths: Iterable[Path], *, threshold: float = 0.5) -> Tree:
    """Return a private prediction tree with one common cutoff at the requested gates.

    Each path addresses an occurrence in the intact tree. Operands, operation order,
    postprocessing and lineage are preserved, including for nested and multi-input
    gates. This frozen-model intervention uses no labels, predictions or RNG draws.
    """
    cutoff = _threshold(threshold, "Replacement cutoff")
    candidate = tree.copy()
    candidate.mutation_chance = tree.mutation_chance
    gates = dict(operator_paths(candidate))
    for path in paths:
        _gate_at(gates, path).threshold = cutoff
    return candidate
