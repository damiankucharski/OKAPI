from fractions import Fraction
from itertools import product
from random import Random

import pytest

from experiments.paired_label_audit.design import PairedBrierAudit


class LoggedLabels:
    def __init__(self, labels, allowed):
        self.labels = labels
        self.allowed = set(allowed)
        self.read = []

    def __call__(self, i):
        assert i in self.allowed and i not in self.read
        self.read.append(i)
        return self.labels[i]


def test_sparse_gap_cache_and_ties():
    audit = PairedBrierAudit([.4, .25, .75, .5], [.4, .75, .75, .5])
    labels = LoggedLabels([0, 0, 0, 1], {1})
    result = audit.sample(labels, 24, Random(13))
    assert labels.read == [1]
    assert result.draws == 24 and result.labels_read == 1 and result.positives == 0
    assert result.estimate == Fraction(-1, 8)
    assert result.lower <= result.estimate <= result.upper


def test_all_identical_uses_no_labels():
    labels = LoggedLabels([1, 0], {})
    audit = PairedBrierAudit([.3, .8], [.3, .8])
    assert audit.sample(labels, 99, Random(23)).draws == 0
    result = audit.attainable(labels, 10)
    assert result.lower == result.upper == 0 and result.labels_read == 0


def test_deterministic_endpoints_and_stopping():
    f, g = [1, .25, .5], [0, .25, .25]
    audit = PairedBrierAudit(f, g)
    for y in product((0, 1), repeat=3):
        labels = LoggedLabels(y, {0, 2})
        full = audit.attainable(labels, 10)
        truth = sum(Fraction((a - c) ** 2 - (b - c) ** 2) for a, b, c in zip(f, g, y)) / 3
        assert full.lower == full.upper == truth
        assert labels.read == [0, 2]
        early_labels = LoggedLabels(y, {0})
        early = audit.attainable(early_labels, 10, stop_when_decided=True)
        assert early.lower <= truth <= early.upper and early.labels_read == 1
        assert early.lower > 0 or early.upper < 0
        if y[0] == 0:
            assert early.lower > 0


def test_sampling_matches_exact_draw_grid_and_reads_only_disagreements():
    audit = PairedBrierAudit([0, 1, 0, 1], [0, .5, .25, 1])

    class ThreeTickets:
        def __init__(self):
            self.tickets = iter((0, 1, 2))
        def randrange(self, total):
            assert total == 3
            return next(self.tickets)

    labels = LoggedLabels([0, 0, 0, 1], {1, 2})
    result = audit.sample(labels, 3, ThreeTickets())
    assert labels.read == [1, 2]
    assert (result.positives, result.draws, result.labels_read) == (2, 3, 2)
    assert result.estimate == Fraction(11, 64)


@pytest.mark.parametrize("f,g", [([], []), ([0], []), ([0, 2], [0, 1])])
def test_invalid_forecast(f, g):
    with pytest.raises(ValueError):
        PairedBrierAudit(f, g)
