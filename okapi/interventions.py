"""Interventions on the ordered updates of an OKAPI tree."""
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from operator import index as integer_index

import numpy as np

from okapi.lib_types import Tensor
from okapi.node import Node, OperatorNode
from okapi.tree import Tree

Path = tuple[int, ...]


def operator_paths(tree: Tree) -> Iterator[tuple[Path, OperatorNode]]:
    """Yield each operator occurrence and its child-index path from the root."""
    def visit(node: Node, path: Path):
        for index, child in enumerate(node.children):
            child_path = (*path, index)
            if isinstance(child, OperatorNode):
                yield child_path, child
            yield from visit(child, child_path)

    yield from visit(tree.root, ())


def operator_at_path(tree: Tree, operator_path: Path) -> OperatorNode:
    """Resolve an operator's child-index path in the current tree."""
    node = tree.root
    for step in operator_path:
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("operator_path must contain nonnegative integer indices")
        if step >= len(node.children):
            raise ValueError("operator_path must identify an operator")
        node = node.children[step]
    if not isinstance(node, OperatorNode):
        raise ValueError("operator_path must identify an operator")
    return node


def copy_for_variation(tree: Tree) -> Tree:
    """Copy a tree while preserving the index order used to sample genetic edits."""
    candidate = tree.copy()
    candidate.mutation_chance = tree.mutation_chance
    copies = {id(a): b for a, b in zip(tree.root.get_nodes(), candidate.root.get_nodes(), strict=True)}
    for name, nodes in candidate.nodes.items():
        ordered = [copies.get(id(node)) for node in tree.nodes[name]]
        if len(ordered) == len(nodes) and {id(node) for node in ordered} == {id(node) for node in nodes}:
            candidate.nodes[name] = ordered
    return candidate


def tree_with_operand_provider(
    tree: Tree, operator_path: Path, child_index: int, provider_id: int | str, tensor: Tensor,
) -> Tree:
    """Bind one direct child of an existing operator to another provider in a private copy.

    Child indices address the operator's stored children. Its parent's incoming
    value is excluded. All operations attached to the child and every other
    occurrence of its former ID retain their original position and parameters.
    """
    if isinstance(child_index, (bool, np.bool_)):
        raise TypeError("child_index must be an integer")
    slot = integer_index(child_index)
    if isinstance(provider_id, (bool, np.bool_)) or not isinstance(provider_id, (int, str, np.integer)):
        raise TypeError("provider_id must be an integer or string")
    candidate = copy_for_variation(tree)
    operator = operator_at_path(candidate, operator_path)
    if not 0 <= slot < len(operator.children):
        raise ValueError("child_index must identify a direct operand")
    child = operator.children[slot]
    if getattr(tensor, "shape", None) != child.value.shape:
        raise ValueError("replacement tensor must match the original provider shape")
    child.id, child.value = provider_id, tensor
    return candidate


def tree_with_bypass(tree: Tree, paths: Iterable[Path]) -> Tree:
    """Copy a tree and skip updates, keeping their parent's incoming value.

    Paths refer to the intact tree. A bypassed update skips its entire operand
    subtree. Subsequent updates on the same value node still run in their
    original order. The input tree and its prediction caches stay untouched.
    """
    candidate = tree.copy()
    by_path = dict(operator_paths(candidate))
    selected = set(paths)
    unknown = selected.difference(by_path)
    if unknown:
        raise ValueError(f"Paths do not identify operators in this tree: {sorted(unknown)}")
    # The copied node references remain valid as sibling indices shift.
    # Descendants are removed first when the selection also includes an ancestor.
    for path in sorted(selected, key=len, reverse=True):
        candidate.prune_at(by_path[path])
    return candidate


def predict_with_bypass(tree: Tree, paths: Iterable[Path]):
    """Predict on the current input tensors of an intervened copy of the tree."""
    return tree_with_bypass(tree, paths).predict()


@dataclass(frozen=True)
class DecisionChanges:
    """Boolean masks aligned to target elements. Direction is bypassed -> intact."""
    repairs: np.ndarray
    harms: np.ndarray


def binary_decision_changes(intact, bypassed, labels, threshold: float = 0.5) -> DecisionChanges:
    """Compare binary probability tensors, including single-channel segmentation maps.

    Prediction shapes must match labels exactly, or include one extra trailing
    singleton channel. A score equal to the threshold predicts the positive class.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite probability")
    def as_array(values):
        # Labels and predictions may come from different backends/devices.
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values)

    y = as_array(labels)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be binary")

    def correct(values):
        scores = as_array(values)
        if scores.shape == (*y.shape, 1):
            scores = scores[..., 0]
        if scores.shape != y.shape:
            raise ValueError("prediction and label shapes are incompatible")
        if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
            raise ValueError("scores must be finite probabilities")
        return (scores >= threshold) == y

    full = correct(intact)
    without = correct(bypassed)
    return DecisionChanges(repairs=full & ~without, harms=~full & without)


def predict_with_substitution(tree: Tree, replacements: dict):
    """Substitute a provider tensor at every occurrence of its ID in a fresh copy.

    The same replacement updates the root and internal/leaf value nodes carrying
    that ID. Use a matched permutation or paired counterfactual that suits the
    data-generating process.
    """
    unknown = set(replacements).difference(tree.unique_value_node_ids)
    if unknown:
        raise ValueError(f"Unknown provider IDs: {unknown}")
    candidate = tree.copy()
    for value_node in candidate.value_nodes:
        if value_node.id in replacements:
            value_node.value = replacements[value_node.id]
    return candidate.predict()
