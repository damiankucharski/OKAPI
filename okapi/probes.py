"""Sequential monitoring of paired, randomized input interventions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbeDecision:
    """Index is -1 on no declaration. Records counts the pairs consumed."""
    index: np.ndarray
    records: np.ndarray


def probe_log_evalues(active, sham, stakes=(.25, .50, .75, .95)):
    """Return log e-values per record and witness for increased response.

    Supply bounded changes in [0,1] from two calls per witness on the same
    record. Inputs end in [record, witness] and may have leading batch axes.
    Under the sharp null the two calls must be exchangeable conditionally on
    the past, e.g. under independent fair randomization of their order and no
    carryover. For each stake l, 1 + l*(active - sham) then has conditional
    expectation at most one. Mix complete wealth processes to retain validity.
    """
    active, sham = np.asarray(active, dtype=float), np.asarray(sham, dtype=float)
    if active.shape != sham.shape or active.ndim < 2 or min(active.shape[-2:]) < 1:
        raise ValueError("active and sham must share nonempty record and witness axes")
    if not all(np.isfinite(x).all() and np.all((x >= 0) & (x <= 1)) for x in (active, sham)):
        raise ValueError("changes must be finite and within [0,1]")
    stakes = np.asarray(stakes, dtype=float)
    if stakes.ndim != 1 or not len(stakes) or not np.all((stakes > 0) & (stakes < 1)):
        raise ValueError("use a nonempty vector of stakes strictly between 0 and 1")
    difference = active - sham
    log_mix = np.full_like(difference, -np.inf)
    for stake in stakes:
        log_wealth = np.cumsum(np.log1p(stake * difference), axis=-2)
        log_mix = np.logaddexp(log_mix, log_wealth)
    return log_mix - np.log(len(stakes))


def first_probe_declaration(log_evalues, alpha=.05):
    """Stop at the first crossing using Bonferroni over witnesses and Ville over time.

    Simultaneous crossings lead to abstention. A declaration concerns the
    intervention response under the stated null, not the witness's accuracy.
    """
    evidence = np.asarray(log_evalues, dtype=float)
    if evidence.ndim < 2 or min(evidence.shape[-2:]) < 1 or not 0 < alpha < 1:
        raise ValueError("expected nonempty record/witness axes and 0 < alpha < 1")
    if np.isnan(evidence).any():
        raise ValueError("log_evalues cannot contain NaN")
    n, witnesses = evidence.shape[-2:]
    crossing = evidence >= np.log(witnesses / alpha)
    any_cross = crossing.any(axis=-1)
    stopped = any_cross.any(axis=-1)
    first = any_cross.argmax(axis=-1)
    at_stop = np.take_along_axis(crossing, first[..., None, None], axis=-2).squeeze(axis=-2)
    unique = stopped & (at_stop.sum(axis=-1) == 1)
    return ProbeDecision(np.where(unique, at_stop.argmax(axis=-1), -1),
                         np.where(stopped, first + 1, n))
