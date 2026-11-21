"""Optional synchronous migration between separately trained OKAPI populations."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from okapi.okapi import Okapi
from okapi.pareto import sort_by_optimal_point_proximity
from okapi.population import choose_pareto_then_proximity
from okapi.random_state import use_rng
from okapi.tree import Tree


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of one incoming offer, with zero-based island indices."""

    source: int
    receiver: int
    evaluated: bool
    admitted: bool
    reason: str


def _elite(engine: Okapi) -> Tree:
    scores = engine.fitnesses
    shape = (len(engine.population), len(engine.objectives))
    if not engine.population or not isinstance(scores, np.ndarray) or scores.shape != shape:
        raise ValueError("Migration requires a nonempty population with aligned fitness scores on each island")
    if not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError("Migration requires finite fitness scores in [0, 1]")
    _, order = sort_by_optimal_point_proximity(scores, engine.objectives)
    return engine.population[order[0]]


def _rejection(elite: Tree, receiver: Okapi) -> str | None:
    if not receiver._within_caps(elite):
        return "caps"
    if any(type(node) not in receiver.allowed_ops for node in elite.op_nodes):
        return "operator"
    if any(node.id not in receiver.train_tensors for node in elite.value_nodes):
        return "provider"
    code = repr(elite)
    if any(repr(resident) == code for resident in receiver.population):
        return "duplicate"
    return None


def migrate_ring(engines: Sequence[Okapi]) -> tuple[MigrationResult, ...]:
    """Offer each island's fit elite to its next neighbor, once, between generations.

    Each island must have current scores for its population in OKAPI's usual
    [0, 1] range. The engines share the process backend and postprocessor. They
    may use different fit tensors, objective functions, caps and operator sets.
    The receiver scores the rebound copy once, then applies its own survival
    rule alongside already-scored residents. The return tuple follows receiver
    order. Count one fit assessment per evaluated offer, even when it loses.

    All offers use the populations at entry. Population and fitness updates
    take effect only after every receiver's scoring and selection succeeds.
    """
    islands = tuple(engines)
    if len(islands) < 2 or len({id(island) for island in islands}) != len(islands):
        raise ValueError("Migration requires at least two distinct islands and no repeated engine")
    elites = [_elite(island) for island in islands]
    results = []
    updates = []
    for i, receiver in enumerate(islands):
        source = (i - 1) % len(islands)
        elite = elites[source]
        reason = _rejection(elite, receiver)
        if reason is not None:
            results.append(MigrationResult(source, i, False, False, reason))
            continue

        with use_rng(receiver.rng):
            immigrant = elite.copy()
            immigrant.mutation_chance = elite.mutation_chance
            immigrant._load_tensors_to_tree(None, receiver.train_tensors)
            score = receiver._calculate_fitnesses([immigrant]).round(3)
        if not np.all((score >= 0) & (score <= 1)):
            raise ValueError("Receiver produced a nonfinite fitness or a score outside [0, 1]")
        survivors, scores = choose_pareto_then_proximity(
            [*receiver.population, immigrant],
            np.concatenate((receiver.fitnesses, score)),
            receiver.objectives, receiver.population_size, receiver.minimize_node_count,
        )
        admitted = any(tree is immigrant for tree in survivors)
        updates.append((receiver, survivors, scores))
        results.append(MigrationResult(source, i, True, admitted, "admitted" if admitted else "survival"))

    for receiver, survivors, scores in updates:
        receiver.population, receiver.fitnesses = survivors, scores
    return tuple(results)
