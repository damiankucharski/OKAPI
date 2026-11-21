"""Exhaustively verify the finite binary audit on a rational 3-row forecast grid.

Run with python experiments/paired_label_audit/verify.py. Only the standard library
is used. Every forecast, loss, probability, expectation and variance is a Fraction.
"""

import json
from fractions import Fraction
from itertools import combinations, product
from math import comb
from time import perf_counter

N = 3
HALF = Fraction(1, 2)
GRID = (Fraction(0), HALF, Fraction(1))
FORECASTS = tuple(product(GRID, repeat=N))
LABELS = tuple(product((0, 1), repeat=N))
SUBSETS = tuple(subset for size in range(N + 1) for subset in combinations(range(N), size))


def mean(values):
    return sum(values, Fraction(0)) / len(values)


def risks(f, g, y):
    return tuple((a - truth)**2 - (b - truth)**2 for a, b, truth in zip(f, g, y, strict=True))


def two_draws(offset, h, probabilities, truth):
    """Enumerate ordered samples and the estimator's first two centered moments."""
    n = len(h)
    support = [(q, h[i] / (n * q)) for i, q in enumerate(probabilities) if q > 0]
    expected = variance = Fraction(0)
    counts = [Fraction(0)] * 3
    for (qi, value_i), (qj, value_j) in product(support, repeat=2):
        chance = qi * qj
        estimate = offset + (value_i + value_j) / 2
        expected += chance * estimate
        variance += chance * (estimate - truth)**2
        counts[int(value_i > 0) + int(value_j > 0)] += chance
    assert sum(counts) == 1
    return expected, variance, counts


def check_sampling(d, offset, y, truth):
    h = [delta * (1 - 2 * target) for delta, target in zip(d, y, strict=True)]
    H = mean(h)
    assert offset + H == truth
    uniform_e, uniform_var, _ = two_draws(offset, h, [Fraction(1, N)] * N, truth)
    second = mean([v**2 for v in d])
    assert uniform_e == truth and uniform_var == (second - H**2) / 2
    magnitudes = [abs(v) for v in d]
    A = mean(magnitudes)
    if A == 0:
        assert all(v == 0 for v in h) and uniform_var == truth == offset == 0
        return False

    probabilities = [v / sum(magnitudes) for v in magnitudes]
    expected, variance, counts = two_draws(offset, h, probabilities, truth)
    assert expected == truth and variance == (A**2 - H**2) / 2
    assert uniform_var - variance == mean([(v - A)**2 for v in magnitudes]) / 2
    theta = (1 + H / A) / 2
    assert 0 <= theta <= 1
    assert counts == [comb(2, k) * theta**k * (1 - theta)**(2 - k) for k in range(3)]
    return True


def check_bounds(d, offset, y, actual, compatible):
    """Enumerate every consistent label completion for each acquired-row subset."""
    widths = {}
    for seen in SUBSETS:
        attainable = [actual[labels] for labels in compatible[y, seen]]
        minimum, maximum = min(attainable), max(attainable)
        observed = sum((d[i] * (1 - 2 * y[i]) for i in seen), Fraction(0))
        remaining = sum((abs(d[i]) for i in range(N) if i not in seen), Fraction(0))
        assert minimum == offset + (observed - remaining) / N
        assert maximum == offset + (observed + remaining) / N
        widths[seen] = maximum - minimum
    priority = sorted(range(N), key=lambda i: -abs(d[i]))
    for size in range(N + 1):
        ordered = tuple(sorted(priority[:size]))
        assert widths[ordered] == min(width for subset, width in widths.items() if len(subset) == size)


def different_offset_counterexample():
    # The alternative offset mean(f²−g²), leaving residual −2*d*y, is unbiased for all Y.
    f, g = (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))
    d = [a - b for a, b in zip(f, g, strict=True)]
    alternative = mean([a*a - b*b for a, b in zip(f, g, strict=True)])
    centered = mean([delta * (a + b - 1) for delta, a, b in zip(d, f, g, strict=True)])
    result = {}
    for y in product((0, 1), repeat=2):
        truth = mean(risks(f, g, y))
        h = [v * (1 - 2 * t) for v, t in zip(d, y, strict=True)]
        residual = [-2 * v * t for v, t in zip(d, y, strict=True)]
        expected, centered_var, _ = two_draws(centered, h, [HALF, HALF], truth)
        alternative_e, alternative_var, _ = two_draws(alternative, residual, [HALF, HALF], truth)
        assert expected == alternative_e == truth
        if y == (0, 0):
            assert centered_var == HALF and alternative_var == 0
            result = {'fixed_labels': y, 'centered_variance_two_draws': str(centered_var), 'alternative_variance_two_draws': str(alternative_var)}
    return result


def verify():
    start = perf_counter()
    compatible = {
        (y, subset): [candidate for candidate in LABELS if all(candidate[i] == y[i] for i in subset)]
        for y in LABELS for subset in SUBSETS
    }
    cases = positive_mass = 0
    for f, g in product(FORECASTS, repeat=2):
        d = [a - b for a, b in zip(f, g, strict=True)]
        offset = mean([delta * (a + b - 1) for delta, a, b in zip(d, f, g, strict=True)])
        actual = {y: mean(risks(f, g, y)) for y in LABELS}
        for y in LABELS:
            positive_mass += check_sampling(d, offset, y, actual[y])
            check_bounds(d, offset, y, actual, compatible)
            cases += 1
    assert cases == 5832 and positive_mass == 5616
    return {
        'ordered_forecast_pairs': len(FORECASTS)**2, 'forecast_pair_label_vectors': cases,
        'uniform_two_draw_checks': cases, 'pps_two_draw_checks_including_binomial': positive_mass,
        'identical_forecasts_no_labels_required': cases - positive_mass,
        'exact_intervals_against_every_compatible_completion': cases * len(SUBSETS),
        'greedy_width_against_all_subsets_at_fixed_size': cases * (N + 1),
        'different_offset_counterexample': different_offset_counterexample(), 'runtime_s': round(perf_counter() - start, 3),
    }


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
