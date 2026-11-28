import math

import numpy as np
import pytest

from okapi.replacement import select_brier_replacement


@pytest.mark.parametrize("n,thresholds", [(16, (12, 16, None)), (64, (45, 56, 58))])
def test_fixed_binary_menu_strict_promotion_thresholds(n, thresholds):
    inc = np.full(n, .5)
    challengers = np.full((8, n), .9)
    for rule, threshold in zip(("raw", "kl", "hoeffding"), thresholds, strict=True):
        for k in (n,) if threshold is None else (threshold - 1, threshold):
            d = select_brier_replacement(inc, challengers, np.arange(n) < k, rule=rule)
            accept = threshold is not None and k >= threshold
            assert d.index == (0 if accept else None)
            assert d.eligible == (tuple(range(8)) if accept else ())
            np.testing.assert_allclose(d.gains, 1.6 * k/n - 1.12, atol=1e-15)
            np.testing.assert_allclose(np.asarray(d.gains) - d.margins, d.lower_bounds, atol=1e-15)


def test_duplicate_offer_slots_pay_and_exact_prediction_ties_favor_incumbent():
    y = np.ones(12, dtype=int)
    anchor, child = np.full(12, .5), np.full(12, .9)
    assert select_brier_replacement(anchor, [child], y).index == 0
    assert select_brier_replacement(anchor, [child] * 8, y).index is None
    for rule in ("raw", "kl", "hoeffding"):
        result = select_brier_replacement(anchor, [anchor.copy()], y, rule=rule)
        assert result.index is None and result.lower_bounds == (0.,) and result.margins == (0.,)
    assert select_brier_replacement(anchor, [], y).eligible == ()


def test_general_multiclass_kl_recovers_reference_from_label_enumeration():
    rng = np.random.default_rng(42)
    n, classes = 36, 4
    y = np.arange(n) % classes
    inc = rng.dirichlet(np.full(classes, 2.), n)
    children = rng.dirichlet(np.full(classes, 2.), (2, n))
    children[:, :7] = inc[:7]
    children[0, 7:] = (inc[7:] + children[0, 7:]) / 2
    delta = .05 / 6
    d = select_brier_replacement(inc, children, y, alpha=delta)
    identity = np.eye(classes)
    for i, child in enumerate(children):
        # Evaluate both full Brier losses at every possible outcome, including
        # the unchanged rows, independently of the production algebra.
        values = ((inc[:, None] - identity) ** 2 - (child[:, None] - identity) ** 2).sum(2)
        lo = values.min(1)
        width = np.ptp(values, axis=1)
        observed = values[np.arange(n), y]
        active = width > 0
        m = active.sum()
        b = width.max()
        v = ((observed[active] - lo[active]) / b).mean()
        p_lower = (d.lower_bounds[i] * n - lo.sum()) / (b * m)
        assert 0 < p_lower < v
        assert m * (v*math.log(v/p_lower) + (1-v)*math.log((1-v)/(1-p_lower))) == pytest.approx(math.log(2/delta))
        assert d.gains[i] == pytest.approx(observed.mean())
    perm = rng.permutation(classes)
    permuted = select_brier_replacement(inc[:, perm], children[:, :, perm], np.argsort(perm)[y], alpha=delta)
    np.testing.assert_allclose(permuted.lower_bounds, d.lower_bounds)


def test_cross_slot_best_bound_can_disagree_with_raw_gain():
    # Concentrating a larger gain in one high-variance slot loses to the
    # modest, repeatable candidate with a narrower label-dependent range.
    y = np.ones(64, dtype=int)
    y[:10] = 0
    inc = np.full(64, .5)
    big = np.full(64, 1.)
    small = np.where(y == 0, .46, .54)
    raw = select_brier_replacement(inc, [big, small], y, rule="raw")
    certified = select_brier_replacement(inc, [big, small], y)
    assert raw.index == 0 and certified.index == 1


def test_independent_groups_and_torch_inputs():
    torch = pytest.importorskip("torch")
    ids = np.repeat(np.arange(3), [20, 10, 2])
    inc = torch.full((32, 1), .5)
    child = torch.full((32, 1), .9)
    y = torch.tensor(ids != 2).to(torch.int64)
    assert select_brier_replacement(inc, [child], y).index == 0
    d = select_brier_replacement(inc, [child], y, groups=ids)
    assert d.n_units == 3 and d.index is None
    assert d.gains[0] == pytest.approx((.48 + .48 - 1.12) / 3)


@pytest.mark.parametrize("inc,challengers,y,kw", [
    ([np.nan], [[.5]], [0], {}), ([[.3,.2]], [[.5]], [0], {}), ([.6], [[.2,.3]], [0], {}),
    ([.5], [[.6]], [2], {}), ([.5], [[.6]], [.5], {}), ([.5], [[.6]], [0], {"alpha":0}),
    ([.5], [[.6]], [0], {"groups":[]}), ([.5], [[.6]], [0], {"rule":"unrecognized"}),
])
def test_invalid_arguments(inc, challengers, y, kw):
    with pytest.raises(ValueError):
        select_brier_replacement(inc, challengers, y, **kw)
