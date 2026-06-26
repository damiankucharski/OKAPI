"""Memetic post-hoc parameter tuning (L1.2).

OKAPI's evolutionary search is strong on *structure* but weak on the *continuous
parameters* of parameterised nodes: in-evolution parameter search is only small
fixed-sigma Gaussian mutation plus the initial draw, so a chosen tree is frequently
structurally right yet numerically under-tuned (the under-realisation seen on
parameter-sensitive scenarios such as ``two_pathways`` / threshold gates).

This module polishes a tree's continuous parameters by derivative-free local search
against a caller-supplied score (``higher = better``), holding the structure frozen.
It is **accept-or-revert**: the polished parameters are kept only if they strictly
improve the score, so tuning can never worsen the tree on the objective it is given.
The caller chooses that objective -- in fusionbench it is OKAPI's own CV-robust
selection score (``_calculate_fitnesses([tree])[0, 0]``), so tuning optimises exactly
the criterion that selected the tree, on the same fit data (no held-out fold; the
nested split was descoped in favour of k-fold CV fitness).

Scope (deliberate): this polishes the *final* chosen tree only. It does **not**
strengthen the weak *in-evolution* parameter search that decides which structures
survive -- that is the separate, harder "stronger parameter evolution" item. Post-hoc
tuning is necessary but not sufficient for it.

Tunable nodes and their constraints (mirrored from ``okapi.node``):

============================  =====================  ==============================
Node                          Parameters             Constraint
============================  =====================  ==============================
``LogitMeanNode``             temperature, shift     t in [0.05, 50], s free
``SoftMedianNode``            temperature            [0.01, 5]
``ThresholdNode`` (+sub)      threshold              [0, 1] (piecewise-constant)
``WeightedMeanNode``          weights                simplex (sum = 1, >= 0.01)
``WeightedLogitMeanNode``     weights                [0, W_MAX]
``MeanNode`` / others         --                     no-op
============================  =====================  ==============================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from okapi.node import (
    LogitMeanNode,
    SoftMedianNode,
    ThresholdNode,
    WeightedLogitMeanNode,
    WeightedMeanNode,
)


@dataclass
class TuneResult:
    """Outcome of :func:`tune_tree_params`."""

    improved: bool
    score_before: float
    score_after: float
    n_params: int
    n_eval: int


@dataclass
class _ParamSpec:
    """How to read / write / restore one node's tunable parameters in optimiser space."""

    x0: list[float]
    bounds: list[tuple[float | None, float | None]]
    set_fn: Callable[[Sequence[float]], None]
    restore_fn: Callable[[], None]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _spec_for(node) -> Optional[_ParamSpec]:
    """Build a parameter spec for ``node``, or ``None`` if it has no tunable params.

    Each spec exposes the node's parameters as a flat real vector for a derivative-free
    optimiser, with a constraint-respecting setter and an exact restore that snapshots
    the node's original raw parameters.
    """
    # WeightedLogitMean BEFORE WeightedMean (distinct types; be explicit).
    if isinstance(node, WeightedLogitMeanNode):
        orig = list(node._weights)
        hi = float(getattr(node, "_W_MAX", 50.0))

        def _set(v: Sequence[float], node=node, hi=hi) -> None:
            node._weights = [float(np.clip(w, 0.0, hi)) for w in v]

        def _restore(node=node, orig=orig) -> None:
            node._weights = list(orig)

        return _ParamSpec(list(map(float, orig)), [(0.0, hi)] * len(orig), _set, _restore)

    if isinstance(node, WeightedMeanNode):
        orig = list(node._weights)
        # Optimise in log space: weights = softmax(theta) covers the whole simplex with
        # no equality constraint, and theta0 = log(weights) reproduces the current point.
        theta0 = np.log(np.clip(np.asarray(orig, dtype=float), 1e-6, None)).tolist()

        def _set(v: Sequence[float], node=node) -> None:
            w = _softmax(np.asarray(v, dtype=float))
            w = np.clip(w, 0.01, None)  # mirror WeightedMeanNode.mutate_params floor
            w = w / np.sum(w)  # keep the sum-to-1 invariant
            node._weights = w.tolist()

        def _restore(node=node, orig=orig) -> None:
            node._weights = list(orig)

        return _ParamSpec(theta0, [(-20.0, 20.0)] * len(theta0), _set, _restore)

    if isinstance(node, LogitMeanNode):
        t0, s0 = float(node.temperature), float(node.shift)

        def _set(v: Sequence[float], node=node) -> None:
            node.temperature = float(np.clip(v[0], 0.05, 50.0))
            node.shift = float(v[1])

        def _restore(node=node, t0=t0, s0=s0) -> None:
            node.temperature, node.shift = t0, s0

        return _ParamSpec([t0, s0], [(0.05, 50.0), (-10.0, 10.0)], _set, _restore)

    if isinstance(node, SoftMedianNode):
        t0 = float(node.temperature)

        def _set(v: Sequence[float], node=node) -> None:
            node.temperature = float(np.clip(v[0], 0.01, 5.0))

        def _restore(node=node, t0=t0) -> None:
            node.temperature = t0

        return _ParamSpec([t0], [(0.01, 5.0)], _set, _restore)

    if isinstance(node, ThresholdNode):  # covers Close/Far subclasses (shared .threshold)
        th0 = float(node.threshold)

        def _set(v: Sequence[float], node=node) -> None:
            node.threshold = float(np.clip(v[0], 0.0, 1.0))

        def _restore(node=node, th0=th0) -> None:
            node.threshold = th0

        return _ParamSpec([th0], [(0.0, 1.0)], _set, _restore)

    return None  # MeanNode and any other parameter-free node


def tune_tree_params(
    tree,
    score_fn: Callable[[object], float],
    *,
    max_iter: Optional[int] = None,
    max_eval: Optional[int] = None,
) -> TuneResult:
    """Polish ``tree``'s continuous parameters in place; keep them only if better.

    Args:
        tree: an OKAPI ``Tree``. Mutated in place **only** on acceptance; on rejection
            (or no tunable params) it is left exactly as received.
        score_fn: maps a tree to a scalar where **higher is better** (e.g. OKAPI's
            objective-0 selection fitness). Called repeatedly; must be deterministic.
        max_iter: Powell ``maxiter`` (default ``max(20, 8 * n_params)``).
        max_eval: Powell ``maxfev`` (default ``min(400, 60 * n_params)``).

    Returns:
        :class:`TuneResult`. With no tunable parameters it is a no-op
        (``improved=False`` and ``score_before == score_after``).
    """
    tree.update_nodes()
    specs = [s for s in (_spec_for(n) for n in tree.nodes["op_nodes"]) if s is not None]
    n_params = sum(len(s.x0) for s in specs)
    score_before = float(score_fn(tree))
    if n_params == 0:
        return TuneResult(False, score_before, score_before, 0, 0)

    # Flatten parameters across specs, recording each spec's slice into the joint vector.
    x0: list[float] = []
    bounds: list[tuple[float | None, float | None]] = []
    slices: list[tuple[_ParamSpec, slice]] = []
    for s in specs:
        start = len(x0)
        x0.extend(s.x0)
        bounds.extend(s.bounds)
        slices.append((s, slice(start, start + len(s.x0))))

    def _apply(x: np.ndarray) -> None:
        for s, sl in slices:
            s.set_fn(x[sl])

    state = {"best_score": score_before, "best_x": np.asarray(x0, dtype=float), "n_eval": 0}

    def _neg(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        _apply(x)
        sc = float(score_fn(tree))
        state["n_eval"] += 1
        if sc > state["best_score"]:
            state["best_score"] = sc
            state["best_x"] = x.copy()
        return -sc

    from scipy.optimize import minimize

    d = len(x0)
    mi = max_iter if max_iter is not None else max(20, 8 * d)
    me = max_eval if max_eval is not None else min(400, 60 * d)
    minimize(
        _neg,
        np.asarray(x0, dtype=float),
        method="Powell",
        bounds=bounds,
        options={"maxiter": mi, "maxfev": me, "xtol": 1e-4, "ftol": 1e-4},
    )

    # Use the best evaluation actually seen (robust to a non-smooth objective such as
    # F1 through a 0.5 threshold), and keep it only if it strictly beats the start.
    if state["best_score"] > score_before:
        _apply(state["best_x"])
        return TuneResult(True, score_before, float(state["best_score"]), n_params, int(state["n_eval"]))
    for s in specs:
        s.restore_fn()
    return TuneResult(False, score_before, score_before, n_params, int(state["n_eval"]))
