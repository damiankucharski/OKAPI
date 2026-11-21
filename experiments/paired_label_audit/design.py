"""Finite-pool positive-channel Brier contrast for two frozen binary forecasts."""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from itertools import accumulate
from math import lcm
from numbers import Rational
from operator import index

from scipy.stats import binomtest


def _probability(value):
    if isinstance(value, Rational):
        numerator, denominator = value.numerator, value.denominator
    else:
        numerator, denominator = value.as_integer_ratio()
    result = Fraction(int(numerator), int(denominator))
    if not 0 <= result <= 1:
        raise ValueError("Forecasts must be finite probabilities in [0, 1]")
    return result


def _label_sign(read_label, i):
    value = read_label(i)
    if value not in (0, 1):
        raise ValueError("The finite-pool audit requires binary labels")
    return 1 - 2 * int(value)


@dataclass(frozen=True)
class SampleEstimate:
    estimate: Fraction
    lower: float
    upper: float
    draws: int
    labels_read: int
    positives: int


@dataclass(frozen=True)
class AttainableBounds:
    lower: Fraction
    upper: Fraction
    labels_read: int


class PairedBrierAudit:
    """Audit the Brier loss of f minus g. Construction reads only forecasts."""

    def __init__(self, f, g):
        pairs = [(_probability(a), _probability(b)) for a, b in zip(f, g, strict=True)]
        if not pairs:
            raise ValueError("Expected at least one row")
        self.gaps = tuple(a - b for a, b in pairs)
        self.n_rows = len(pairs)
        self.offset = sum(d * (a + b - 1) for d, (a, b) in zip(self.gaps, pairs)) / self.n_rows
        self.total_gap = sum(map(abs, self.gaps))

    def sample(self, read_label, draws, rng, confidence=.95):
        """Fixed draw count with replacement, exact row weights and cached reads.

        rng supplies randrange(n), for example random.Random(seed). The exact
        binomial confidence procedure is for one fixed pair and fixed draw count.
        """
        draws = index(draws)
        if draws < 1 or not 0 < confidence < 1:
            raise ValueError("Expected a positive draw count and confidence in (0, 1)")
        if not self.total_gap:
            return SampleEstimate(Fraction(0), 0., 0., 0, 0, 0)

        # Integer masses avoid rounding the row sampling probabilities.
        denominator = lcm(*(d.denominator for d in self.gaps))
        cumulative = list(accumulate(int(abs(d) * denominator) for d in self.gaps))
        signs = {}
        positives = 0
        for _ in range(draws):
            i = bisect_right(cumulative, rng.randrange(cumulative[-1]))
            if i not in signs:
                signs[i] = _label_sign(read_label, i)
            positives += int(self.gaps[i] * signs[i] > 0)
        scale = self.total_gap / self.n_rows
        mean = self.offset + scale * Fraction(2 * positives - draws, draws)
        ci = binomtest(positives, draws).proportion_ci(confidence_level=confidence, method="exact")
        low = float(self.offset) + float(scale) * (2 * ci.low - 1)
        high = float(self.offset) + float(scale) * (2 * ci.high - 1)
        return SampleEstimate(mean, low, high, draws, len(signs), positives)

    def attainable(self, read_label, budget, *, stop_when_decided=False):
        """Sharp risk-difference endpoints after inspecting distinct labels."""
        budget = index(budget)
        if budget < 0:
            raise ValueError("The label budget must be nonnegative")
        ordered = sorted(range(self.n_rows), key=lambda i: abs(self.gaps[i]), reverse=True)
        known = self.offset * self.n_rows
        remainder = self.total_gap
        reads = 0
        for i in ordered[:budget]:
            if not remainder or (stop_when_decided and abs(known) > remainder):
                break
            d = self.gaps[i]
            known += d * _label_sign(read_label, i)
            remainder -= abs(d)
            reads += 1
        return AttainableBounds((known - remainder) / self.n_rows, (known + remainder) / self.n_rows, reads)
