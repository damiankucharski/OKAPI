"""Exact probabilities for one ordered OKAPI tournament, including ties."""

import json
from fractions import Fraction
from itertools import permutations
from math import comb, perm
from numbers import Integral

import numpy as np


def tournament_weights(fitnesses, tournament_size, optimal_point):
    """Return each indexed parent's exact winning probability as a Fraction.

    Supply the actual native selection matrix (including any configured rounding)
    and the engine's ideal point: one component per objective, 1 for maximize and
    0 for minimize. The contender order is uniformly sampled without replacement.
    The first contender at the minimum native Euclidean distance wins a tie.
    Two separate parent tournaments independently use these same probabilities.
    """
    scores = np.asarray(fitnesses)
    point = np.asarray(optimal_point)
    if scores.ndim != 2 or min(scores.shape) == 0 or scores.dtype.kind not in 'iuf':
        raise ValueError('fitnesses must be a nonempty numeric [parents, objectives] matrix')
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('normalized selection fitnesses must be finite values in [0, 1]')
    if point.shape != (scores.shape[1],) or point.dtype.kind not in 'iuf' or not np.isin(point, [0, 1]).all():
        raise ValueError('pass the engine ideal point as a vector with one 0 or 1 for each objective')
    n = len(scores)
    if isinstance(tournament_size, bool) or not isinstance(tournament_size, Integral) or not 1 <= tournament_size <= n:
        raise ValueError('tournament_size must be an integer between 1 and the population size')

    # Match the native square, sum, sqrt operations. Squared distances can differ
    # while their floating-point square roots tie, so grouping must occur afterwards.
    distances = np.sqrt(np.sum((scores - point.reshape(1, -1))**2, axis=1))
    _, group, counts = np.unique(distances, return_inverse=True, return_counts=True)
    groups = []
    worse = n
    total_subsets = comb(n, tournament_size)
    for count in counts:
        tied = int(count)
        worse -= tied
        # Winning subsets exclude every strictly better parent and include at
        # least one of this tie group. Each tied parent wins equally often by symmetry.
        winning_subsets = comb(worse + tied, tournament_size) - comb(worse, tournament_size)
        groups.append(Fraction(winning_subsets, total_subsets * tied))
    weights = tuple(groups[int(index)] for index in group)
    assert sum(weights) == 1
    return weights


def verify():
    from okapi.crossover import tournament_selection_indexes
    from okapi.random_state import use_rng

    # The first pair has equal distance with different objective vectors. Other
    # parents include exactly repeated scores and a deliberately poor singleton.
    scores = np.array([[.75, 0.], [1., .25], [.75, .25], [.75, .25], [.25, .25], [0., .75]])
    exact_orders = drawn_parents = 0
    for ideal in (np.array([1., 0.]), np.array([0., 1.])):
        # Independent Manhattan simplification would fail in general; hypot is
        # exact enough for this dyadic fixture while keeping the enumerator independent.
        distance = np.hypot(scores[:, 0] - ideal[0], scores[:, 1] - ideal[1])
        for k in (2, 3, 5):
            expected = tournament_weights(scores, k, ideal)
            counts = [0] * len(scores)
            for order in permutations(range(len(scores)), k):
                first_at_minimum = min(order, key=lambda i: distance[i])
                counts[first_at_minimum] += 1
            exact_orders += sum(counts)
            assert tuple(Fraction(count, perm(len(scores), k)) for count in counts) == expected

            # Draw native tournaments with a real seeded stream. Independently
            # reconstruct both ordered sets from its twin and score each in Python.
            seed = 1429 + 17 * k + int(ideal[0])
            native_rng, witness_rng = np.random.RandomState(seed), np.random.RandomState(seed)
            histogram = np.zeros(len(scores), dtype=int)
            with use_rng(native_rng):
                for _ in range(1024):
                    actual = tournament_selection_indexes(scores, k, ideal)
                    for index in actual:
                        order = witness_rng.choice(np.arange(len(scores)), size=k, replace=False)
                        assert index == min(order, key=lambda i: distance[i])
                        histogram[index] += 1
            drawn_parents += int(histogram.sum())
            assert np.max(np.abs(histogram / histogram.sum() - np.asarray(expected, dtype=float))) < .05

    # Equal native distances can arise only after the final floating point sqrt.
    tiny_difference = np.array([[.5, 0.], [.5, 2.**-27]])
    squares = (tiny_difference**2).sum(axis=1)
    assert squares[0] < squares[1] and np.sqrt(squares[0]) == np.sqrt(squares[1])
    assert tournament_weights(tiny_difference, 2, np.zeros(2)) == (Fraction(1, 2), Fraction(1, 2))
    for invalid in (None, 1, np.ones(3), np.array([np.nan, 1.]), np.array([.5, 1.])):
        try:
            tournament_weights(scores, 2, invalid)
        except ValueError:
            continue
        raise AssertionError(f'Accepted an incompatible native ideal point: {invalid}')

    # The full 38-parent geometry admits free sanity checks, without reading any study data.
    full = np.linspace(0., 1., 38).reshape(-1, 1)
    full_weights = tournament_weights(full, 5, np.ones(1))
    assert full_weights[-1] == Fraction(5, 38) and all(w == 0 for w in full_weights[:4])
    tied = np.ones((38, 1))
    assert tournament_weights(tied, 5, np.ones(1)) == (Fraction(1, 38),) * 38
    assert tournament_weights(full, 1, np.ones(1)) == (Fraction(1, 38),) * 38
    assert tournament_weights(full, 38, np.zeros(1)) == (Fraction(1),) + (Fraction(0),) * 37
    return dict(exhaustive_orders=exact_orders, seeded_native_parents=drawn_parents, mixed_ideal_points=2, actual_study_cohorts=0)


if __name__ == '__main__':
    from loguru import logger

    import okapi  # initialize native logging before suppressing high-volume tournament messages

    logger.disable(okapi.__name__)
    print(json.dumps(verify(), sort_keys=True))
