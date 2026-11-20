"""Budgeted local search over provider assignments of an existing tree."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from operator import index

import numpy as np

from okapi.lib_types import Tensor
from okapi.tree import Tree


@dataclass(frozen=True)
class RefinementResult:
    tree: Tree
    initial_loss: float
    loss: float
    evaluations: int


def refine_provider_assignments(
    tree: Tree,
    providers: Mapping[int | str, Tensor],
    labels: Tensor,
    loss: Callable[[Tensor, Tensor], float],
    *,
    max_evaluations: int,
    seed: int | np.random.SeedSequence | None = 0,
) -> RefinementResult:
    """Minimize a deterministic fit loss over single-occurrence substitutions.

    Bind every value node of a private copy to the supplied fit tensors, even
    when the input tree still holds tensors from an earlier split. Score the
    incumbent once, then score unique (occurrence, alternative ID) pairs up to
    the remaining budget. Each move starts from the incumbent. A strict
    improvement changes one assignment and preserves topology and parameters.

    Supply providers in stable order for a reproducible sampled neighborhood.
    Ties favor the incumbent, then the first evaluated improvement. As during
    GP fitting, the loss must leave predictions and labels untouched. Use
    separate validation examples if choosing whether to deploy the result.
    """
    limit = index(max_evaluations)
    if limit < 1:
        raise ValueError("max_evaluations must include at least the incumbent")
    candidate = tree.copy()
    candidate.mutation_chance = tree.mutation_chance
    nodes = candidate.value_nodes
    ids = list(providers)
    for node in nodes:
        if node.id not in providers:
            raise ValueError(f"Missing fit predictions for provider {node.id!r}")
        node.value = providers[node.id]

    def score():
        value = float(loss(candidate.predict(), labels))
        if not np.isfinite(value):
            raise ValueError("loss must return a finite scalar")
        return value

    best_loss = initial_loss = score()
    # A value's current ID is excluded from its ordered block of moves.
    id_to_index = {provider: i for i, provider in enumerate(ids)}
    alternatives = len(ids) - 1
    total = alternatives * len(nodes)
    trials = min(limit - 1, total)
    moves = range(total) if trials == total else np.random.default_rng(seed).choice(total, trials, replace=False)
    best_move = None
    for move in moves:
        occurrence, alternative = divmod(int(move), alternatives)
        node = nodes[occurrence]
        old_id = node.id
        new_id = ids[alternative + (alternative >= id_to_index[old_id])]
        node.id, node.value = new_id, providers[new_id]
        value = score()
        node.id, node.value = old_id, providers[old_id]
        if value < best_loss:
            best_move = occurrence, new_id
            best_loss = value

    if best_move is not None:
        occurrence, new_id = best_move
        nodes[occurrence].id, nodes[occurrence].value = new_id, providers[new_id]
    return RefinementResult(candidate, initial_loss, best_loss, 1 + trials)
