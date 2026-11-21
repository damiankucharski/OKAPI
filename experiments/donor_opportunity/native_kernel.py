"""Exact conditional laws of the native two-way tournament and two-child crossover."""

from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from numbers import Integral

import numpy as np

from okapi.crossover import crossover, tournament_selection_indexes
from okapi.random_state import use_rng
from okapi.tree import Tree


class _NextDraw(Exception):
    def __init__(self, choices):
        self.choices = choices


class FiniteDraws:
    """Supply a path of discrete choices to the ordinary crossover implementation.

    Missing choices reveal the next branch of the native control flow. A full
    permutation of the node list is represented by its first entry: crossover
    calls get_random_node with roots and leaves allowed, so no entry is skipped.
    """

    def __init__(self, path):
        self.path = path
        self.position = 0

    def _index(self, n):
        if n < 1:
            raise ValueError('Expected at least one available native option')
        if n == 1:
            return 0
        if self.position == len(self.path):
            raise _NextDraw(n)
        index = self.path[self.position]
        self.position += 1
        if not 0 <= index < n:
            raise ValueError('Choice index outside native support')
        return index

    def choice(self, values):
        return values[self._index(len(values))]

    def permutation(self, n):
        if not isinstance(n, Integral):
            raise ValueError('Expected the size of an unconstrained native node list')
        index = self._index(n)
        order = np.arange(n)
        order[0], order[index] = index, 0
        return order

    def shuffle(self, values):
        # Native capped crossover randomizes the order of the two node types,
        # then takes the first type with a legal pair for *both* children.
        if len(values) not in (1, 2):
            raise ValueError('Expected one or two native node types')
        if self._index(len(values)):
            values.reverse()

    def randint(self, n):
        # The actual native crossover has already enumerated all legal pairs.
        return self._index(n)


class _TournamentDraw:
    def __init__(self, ordered_pair):
        self.ordered_pair = ordered_pair

    def choice(self, candidates, size, replace):
        if size != (2,) or replace:
            raise ValueError('This checker requires draws of size two without replacement')
        return np.asarray(candidates)[list(self.ordered_pair)]


def tournament_probabilities(fitnesses, optimal_point=None):
    """Replay all ordered size-two samples, retaining the native first-on-tie rule."""
    scores = np.asarray(fitnesses, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2 or len(scores) < 2 or scores.shape[1] < 1 or not np.isfinite(scores).all():
        raise ValueError('Supply a finite fitness matrix with at least two parents')
    ideal = np.ones(scores.shape[1]) if optimal_point is None else np.asarray(optimal_point, dtype=np.float64)
    if ideal.shape != (scores.shape[1],) or not np.isfinite(ideal).all():
        raise ValueError('Supply one finite ideal coordinate per fitness objective')
    distance = np.sqrt(np.sum((scores - ideal)**2, axis=1))
    wins = Counter()
    for a, b in permutations(range(len(scores)), 2):
        with use_rng(_TournamentDraw((a, b))):
            left, right = tournament_selection_indexes(scores, tournament_size=2, optimal_point=ideal)
        expected = b if distance[b] < distance[a] else a
        if not left == right == expected:
            raise AssertionError('Native parent choice differs from its ordered distance comparison')
        wins[int(left)] += 1
    total = len(scores) * (len(scores) - 1)
    return tuple(Fraction(wins[i], total) for i in range(len(scores)))


@dataclass(frozen=True)
class Pair:
    """Conditional child-signature probabilities given this ordered parent pair.

    n_draw_paths counts branches of the discrete choices represented by FiniteDraws.
    Two different capped type orders may reach the same legal exchange.
    """

    child_probs: dict[tuple[Hashable, Hashable], Fraction]
    productive: Fraction
    tournament_weight: Fraction
    uniform_weight: Fraction
    n_draw_paths: int


@dataclass(frozen=True)
class Kernel:
    parent_probs: tuple[Fraction, ...]
    parent_signatures: tuple[Hashable, ...]
    pairs: dict[tuple[int, int], Pair]
    productive: Fraction
    uniform_productive: Fraction


def child_distribution(parent1, parent2, *, signature, max_depth=None, max_nodes=None):
    """Count both children's ordered signatures over the actual native draw support."""
    pending = [((), Fraction(1))]
    probabilities = Counter()
    leaves = 0
    while pending:
        path, mass = pending.pop()
        draws = FiniteDraws(path)
        with use_rng(draws):
            try:
                left, right = crossover(parent1, parent2, max_depth=max_depth, max_nodes=max_nodes)
            except _NextDraw as draw:
                branch_mass = mass / draw.choices
                pending.extend(((*path, i), branch_mass) for i in range(draw.choices))
                continue
        if draws.position != len(path):
            raise AssertionError('Native crossover consumed an unexpected number of choices')
        probabilities[signature(left), signature(right)] += mass
        leaves += 1
    if sum(probabilities.values()) != 1:
        raise AssertionError('Child probability mass must sum to one')
    return dict(probabilities), leaves


def kernel(
    parents: Sequence[Tree], fitnesses, *, signature: Callable[[Tree], Hashable], target: Hashable,
    max_depth=None, max_nodes=None, optimal_point=None,
):
    """Compute the probability of a target child from two non-target parents.

    Supply the actual stored native selection scores, including any chosen fitness
    rounding. A productive event has two target-free input parents and at least one
    child equal to target. Both overall probabilities include all ordered parent
    pairs (with replacement); the uniform comparison gives each pair mass 1/N².
    The signature callback must be deterministic and must not draw random numbers.
    """
    parents = tuple(parents)
    if len(parents) != len(fitnesses):
        raise ValueError('Supply one fitness row per input parent')
    parent_probs = tournament_probabilities(fitnesses, optimal_point)
    parent_signatures = tuple(signature(parent) for parent in parents)
    uniform_weight = Fraction(1, len(parents)**2)
    pairs = {}
    productive, uniform_productive = Fraction(), Fraction()
    for i, first in enumerate(parents):
        for j, second in enumerate(parents):
            outcomes, leaves = child_distribution(first, second, signature=signature, max_depth=max_depth, max_nodes=max_nodes)
            gain = Fraction()
            if parent_signatures[i] != target and parent_signatures[j] != target:
                gain = sum((mass for children, mass in outcomes.items() if target in children), Fraction())
            weight = parent_probs[i] * parent_probs[j]
            pairs[i, j] = Pair(outcomes, gain, weight, uniform_weight, leaves)
            productive += gain * weight
            uniform_productive += gain * uniform_weight
    return Kernel(parent_probs, parent_signatures, pairs, productive, uniform_productive)
