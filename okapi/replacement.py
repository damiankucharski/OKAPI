"""A fixed-design full-Brier comparison for one incumbent and an ordered family of challengers."""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from okapi.paired_fitness import _as_array, _group_index


@dataclass(frozen=True)
class ReplacementDecision:
    """index is None when the incumbent stays. All numeric tuples follow challenger order."""

    index: int | None
    eligible: tuple[int, ...]
    gains: tuple[float, ...]
    lower_bounds: tuple[float, ...]
    margins: tuple[float, ...]
    n_units: int
    rule: str


def _probabilities(prediction):
    p = _as_array(prediction)
    if p.dtype.kind not in "iuf" or p.ndim not in (1, 2) or not p.size:
        raise ValueError("Supply nonempty (N,C) probabilities or binary positive-class scores")
    p = p.astype(np.float64, copy=False)
    if not np.all((0 <= p) & (p <= 1)):
        raise ValueError("Probabilities must be finite and in [0,1]")
    if p.ndim == 1 or p.shape[1] == 1:
        positive = p.reshape(-1)
        return np.column_stack((1 - positive, positive))
    if not np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-6):
        raise ValueError("Class probabilities must sum to one")
    return p


def _lower_agreement(observed, rate):
    """Conservative lower bracket of kl(observed || p) = rate for p <= observed."""
    if observed == 0:
        return 0.
    if observed == 1:
        return math.nextafter(math.exp(-rate), 0.)
    lo, hi = 0., observed
    for _ in range(64):
        p = (lo + hi) / 2
        if p == lo or p == hi:
            break
        divergence = (observed * (math.log(observed) - math.log(p))
                      + (1 - observed) * (math.log1p(-observed) - math.log1p(-p)))
        if divergence >= rate:
            lo = p
        else:
            hi = p
    return lo


def select_brier_replacement(incumbent_prediction, challenger_predictions, labels, *, alpha=.05, rule="kl", groups=None) -> ReplacementDecision:
    """Choose the greatest strictly positive lower bound on gain over a frozen incumbent.

    Supply the incumbent (N,C), ordered challengers (K,N,C), and integer class
    indices (N,). Binary positive-class predictions (N,) or (N,1) also work.
    Gains use full Brier loss summed over classes. The incumbent and all K
    slots must be determined before revealing these labels. Duplicates pay
    the same alpha/K allowance as distinct offers. Strict ties with the
    incumbent keep it, and positive challenger ties favor the earlier slot.

    For each unit, minimize and maximize the paired full-Brier gain over
    possible labels to obtain its lower endpoint and width. The 'kl' rule
    scales all positive widths by their maximum, inverts the one-sided
    Bernoulli KL-Chernoff bound for the resulting bounded independent scores,
    and adds back the deterministic endpoints. 'hoeffding' uses the sum of
    squared widths. 'raw' returns empirical gains as scores for comparison.

    With independent labels, the two confidence rules simultaneously lower
    bound the candidates' *fixed-design* expected gains with probability at
    least 1-alpha. For independent groups with dependent rows, pass group
    IDs to average row gains and endpoints within groups and give each group
    equal weight. Repeated adaptive generations require fresh, independent
    confirmation units and a prespecified allocation of the total error.
    """
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if rule not in ("kl", "hoeffding", "raw"):
        raise ValueError("rule must be 'kl', 'hoeffding' or 'raw'")
    inc = _probabilities(incumbent_prediction)
    n, c = inc.shape
    y = _as_array(labels)
    if y.shape == (n, 1):
        y = y[:, 0]
    if y.shape != (n,) or y.dtype.kind not in "iub" or np.any((y < 0) | (y >= c)):
        raise ValueError("Supply one in-range integer class index per row")
    y = y.astype(np.intp, copy=False)
    group_index = None if groups is None else _group_index(groups, n)
    units = n if group_index is None else len(group_index[1])
    challengers = tuple(challenger_predictions)
    k = len(challengers)
    if not k:
        return ReplacementDecision(None, (), (), (), (), units, rule)
    log_budget = math.log(k) - math.log(alpha)
    gains, bounds = [], []
    rows = np.arange(n)
    for candidate in challengers:
        p = _probabilities(candidate)
        if p.shape != inc.shape:
            raise ValueError("All challengers must match the incumbent's row and class counts")
        diff = p - inc
        minimum = diff.min(axis=1)
        # For label c the gain is ||inc||² - ||p||² + 2*(p_c - inc_c).
        # Compute the part above the lower endpoint directly to avoid cancellation.
        lower_end = 2 * minimum - (diff * (p + inc)).sum(axis=1)
        widths = 2 * (diff.max(axis=1) - minimum)
        offsets = 2 * (diff[rows, y] - minimum)
        if group_index is not None:
            membership, counts = group_index
            lower_end, widths, offsets = (np.bincount(membership, weights=v) / counts for v in (lower_end, widths, offsets))
        gain = float((lower_end + offsets).mean())
        if rule == "raw":
            bound = gain
        elif rule == "hoeffding":
            bound = gain - math.sqrt(float(widths @ widths) * log_budget / 2) / units
        else:
            maximum = float(widths.max())
            active = int(np.count_nonzero(widths))
            bound = float(lower_end.mean())
            if active:
                # Inactive offsets are zero. Cap harmless rounding at the endpoints.
                observed = min(1., max(0., (float(offsets.sum()) / active) / maximum))
                bound += maximum * (active / units) * _lower_agreement(observed, log_budget / active)
            bound = min(gain, bound)
        gains.append(gain)
        bounds.append(bound)

    eligible = tuple(i for i, bound in enumerate(bounds) if bound > 0)
    index = max(eligible, key=bounds.__getitem__, default=None)
    return ReplacementDecision(index, eligible, tuple(gains), tuple(bounds),
                               tuple(g - b for g, b in zip(gains, bounds, strict=True)), units, rule)
