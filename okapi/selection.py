"""Select a compact tree from repeated searches using independent validation groups."""
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from okapi.lib_types import Tensor
from okapi.tree import Tree


@dataclass(frozen=True)
class SelectionResult:
    """Indices refer to the supplied candidate order, including the tie-break."""
    index: int
    best_index: int
    eligible: tuple[int, ...]


def select_compact(scores, sizes, *, multiplier: float = 1.0) -> SelectionResult:
    """Prefer a small tree within one paired standard error of the validation winner.

    scores[i, g] is loss (lower is better) for candidate i on group g. Groups
    count equally and should be independent units, e.g. flights. All candidates
    must be scored on exactly the same groups. The mean loss difference to the
    lowest-loss candidate is compared with its own between-group standard error.
    This is a model-selection heuristic, not an equivalence or significance test.
    Estimate performance on further unused groups.
    """
    scores = np.asarray(scores, dtype=float)
    sizes = np.asarray(sizes)
    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 2:
        raise ValueError("scores must be a nonempty candidate-by-group matrix with at least two groups")
    if not np.isfinite(scores).all():
        raise ValueError("all candidate losses must be finite")
    if sizes.shape != (len(scores),) or sizes.dtype.kind not in "iu" or (sizes < 1).any():
        raise ValueError("supply one positive integer node count per candidate")
    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("multiplier must be finite and nonnegative")
    mean_loss = scores.mean(axis=1)
    best = min(range(len(scores)), key=lambda i: (mean_loss[i], sizes[i], i))
    difference = scores - scores[best]
    se = difference.std(axis=1, ddof=1) / np.sqrt(scores.shape[1])
    eligible = tuple(int(i) for i in np.flatnonzero(difference.mean(axis=1) <= multiplier * se))
    index = min(eligible, key=lambda i: (sizes[i], mean_loss[i], i))
    return SelectionResult(index, best, eligible)


def select_tree(candidates: Sequence[Tree], groups: Iterable[tuple[dict, Tensor]],
                loss: Callable[[Tensor, Tensor], float], *, multiplier: float = 1.0) -> SelectionResult:
    """Score OKAPI trees on held-out groups of (provider tensors, labels).

    The returned indices refer to candidates. The supplied loss reduces one
    whole group to a finite scalar. Prediction uses a copy of each architecture
    with that group's inputs, preserving training tensors and evaluation caches.
    """
    if not candidates:
        raise ValueError("at least one candidate tree is required")
    # Reuse the private architectures across groups. predict() clears the
    # intermediate tensors after each evaluation, including for large maps.
    copies = [tree.copy() for tree in candidates]
    group_losses = []
    for tensors, labels in groups:
        losses = []
        for tree in copies:
            for node in tree.value_nodes:
                node.value = tensors[node.id]
            losses.append(float(loss(tree.predict(), labels)))
        group_losses.append(losses)
    return select_compact(np.asarray(group_losses).T, [tree.nodes_count for tree in candidates],
                          multiplier=multiplier)


def estimate_context_prevalence(target_mean, reference_features, changed_features, fallback):
    """Estimate a two-context mixture from unlabeled target feature means.

    Rows of each source matrix are observations in a known context. Columns
    must be the same features in all populations (e.g. expert disagreement
    indicators). Their conditional expectations must transfer to the target.
    Project the target mean onto the segment joining the two source means.
    Return the source prevalence if a context is absent or unidentifiable.
    """
    target = np.asarray(target_mean, dtype=float)
    a = np.asarray(reference_features, dtype=float)
    b = np.asarray(changed_features, dtype=float)
    if target.ndim != 1 or target.size == 0 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected a target mean vector and two source matrices")
    if a.shape[1:] != target.shape or b.shape[1:] != target.shape:
        raise ValueError("source and target features must have the same dimension")
    if not all(np.isfinite(x).all() for x in (target, a, b)) or not 0 <= fallback <= 1:
        raise ValueError("supply finite feature values and a fallback probability")
    if min(len(a), len(b)) == 0:
        return float(fallback)
    origin = a.mean(axis=0)
    direction = b.mean(axis=0) - origin
    norm_squared = direction @ direction
    if norm_squared < 1e-12:
        return float(fallback)
    return float(np.clip((target - origin) @ direction / norm_squared, 0.0, 1.0))
