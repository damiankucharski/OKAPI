"""Tests for memetic post-hoc parameter tuning (``okapi.tuning``, L1.2).

Two layers:

* **Spec layer** (``_spec_for``): per-node-type extract / set / restore in optimiser
  space, including the constraints each node enforces (logit-mean clamp, soft-median
  clamp, threshold ``[0, 1]``, weighted-mean simplex, weighted-logit-mean ``[0, W]``).
* **Driver layer** (``tune_tree_params``): finds a known optimum, is **accept-or-revert**
  (never worsens, restores exactly on no-gain), leaves *structure* untouched, is a no-op
  on parameter-free trees, is deterministic, and improves a *real* tree prediction.
"""

import numpy as np
import pytest

from okapi.node import (
    CloseThresholdNode,
    FarThresholdNode,
    LogitMeanNode,
    MeanNode,
    SoftMedianNode,
    ThresholdNode,
    ValueNode,
    WeightedLogitMeanNode,
    WeightedMeanNode,
)
from okapi.tree import Tree
from okapi.tuning import TuneResult, _spec_for, tune_tree_params


def _vn(value, name):
    return ValueNode(None, np.asarray(value, dtype=float), name)


def _mse(pred, target):
    return float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))


# --------------------------------------------------------------------------------------
# Spec layer: _spec_for per node type
# --------------------------------------------------------------------------------------


def test_spec_none_for_param_free_nodes():
    c, d = _vn(np.zeros((2, 1)), "c"), _vn(np.zeros((2, 1)), "d")
    assert _spec_for(MeanNode([c, d])) is None
    assert _spec_for(_vn(np.zeros((2, 1)), "v")) is None  # a ValueNode has no op params


def test_spec_logitmean_roundtrip_bounds_and_restore():
    node = LogitMeanNode([_vn(np.zeros((2, 1)), "c")], 2.0, 0.5)
    spec = _spec_for(node)
    assert spec.x0 == [2.0, 0.5]
    assert spec.bounds == [(0.05, 50.0), (-10.0, 10.0)]
    spec.set_fn([100.0, 3.0])  # temperature clamped to 50, shift free
    assert node.temperature == 50.0 and node.shift == 3.0
    spec.set_fn([0.0, -1.0])  # temperature clamped up to 0.05
    assert node.temperature == 0.05
    spec.restore_fn()
    assert node.temperature == 2.0 and node.shift == 0.5


def test_spec_softmedian_clamp_and_restore():
    node = SoftMedianNode([_vn(np.zeros((2, 1)), "c")], 0.3)
    spec = _spec_for(node)
    assert spec.x0 == [0.3] and spec.bounds == [(0.01, 5.0)]
    spec.set_fn([10.0])
    assert node.temperature == 5.0
    spec.set_fn([0.0])
    assert node.temperature == 0.01
    spec.restore_fn()
    assert node.temperature == 0.3


@pytest.mark.parametrize("cls", [CloseThresholdNode, FarThresholdNode])
def test_spec_threshold_subclasses_clamp_to_unit_interval(cls):
    node = cls([_vn(np.zeros((2, 1)), "c")], 0.3)
    assert isinstance(node, ThresholdNode)
    spec = _spec_for(node)
    assert spec.x0 == [0.3] and spec.bounds == [(0.0, 1.0)]
    spec.set_fn([2.0])
    assert node.threshold == 1.0
    spec.set_fn([-1.0])
    assert node.threshold == 0.0
    spec.restore_fn()
    assert node.threshold == 0.3


def test_spec_weightedlogitmean_clamp_and_restore():
    c, d = _vn(np.zeros((2, 1)), "c"), _vn(np.zeros((2, 1)), "d")
    node = WeightedLogitMeanNode([c, d], [0.5, 1.5, 2.0])
    spec = _spec_for(node)
    assert spec.x0 == [0.5, 1.5, 2.0]
    assert spec.bounds == [(0.0, 50.0)] * 3
    spec.set_fn([100.0, -1.0, 2.0])  # clamp to [0, 50]
    assert node._weights == [50.0, 0.0, 2.0]
    spec.restore_fn()
    assert node._weights == [0.5, 1.5, 2.0]


def test_spec_weightedmean_stays_on_simplex_and_restores_exactly():
    c, d = _vn(np.zeros((2, 1)), "c"), _vn(np.zeros((2, 1)), "d")
    orig = [0.2, 0.3, 0.5]
    node = WeightedMeanNode([c, d], list(orig))
    spec = _spec_for(node)
    # x0 = log(weights): re-applying it reproduces the original simplex point.
    spec.set_fn(spec.x0)
    np.testing.assert_allclose(node._weights, orig, atol=1e-9)
    # An arbitrary theta still yields a valid probability simplex (sums to 1, positive).
    spec.set_fn([3.0, -2.0, 0.1])
    assert abs(sum(node._weights) - 1.0) < 1e-9
    assert all(w > 0.0 for w in node._weights)
    spec.restore_fn()
    assert node._weights == orig


# --------------------------------------------------------------------------------------
# Driver layer: tune_tree_params
# --------------------------------------------------------------------------------------


def _logit_tree(temperature, shift):
    a = _vn([[0.6], [0.7]], "A")
    a.add_child(LogitMeanNode([_vn([[0.8], [0.2]], "C"), _vn([[0.5], [0.9]], "D")], temperature, shift))
    return Tree.create_tree_from_root(a), a.children[0]


def test_noop_on_parameter_free_tree():
    a = _vn([[0.6], [0.7]], "A")
    a.add_child(MeanNode([_vn([[0.8], [0.2]], "C"), _vn([[0.5], [0.9]], "D")]))
    tree = Tree.create_tree_from_root(a)
    before = tree.predict()
    res = tune_tree_params(tree, lambda t: 0.0)
    assert isinstance(res, TuneResult)
    assert res.improved is False and res.n_params == 0 and res.n_eval == 0
    np.testing.assert_allclose(tree.predict(), before, atol=1e-12)


def test_finds_known_optimum():
    tree, op = _logit_tree(0.5, 0.0)
    target_t, target_s = 5.0, 2.0
    score = lambda t: -((op.temperature - target_t) ** 2 + (op.shift - target_s) ** 2)  # noqa: E731
    n_before = tree.nodes_count
    res = tune_tree_params(tree, score)
    assert res.improved is True
    assert res.score_after > res.score_before
    assert op.temperature == pytest.approx(target_t, abs=0.1)
    assert op.shift == pytest.approx(target_s, abs=0.1)
    assert tree.nodes_count == n_before  # structure untouched


def test_accept_or_revert_never_worsens_and_restores_exactly():
    tree, op = _logit_tree(2.0, 0.5)
    # Score is maximal exactly at the starting params -> no strict gain is possible.
    score = lambda t: -((op.temperature - 2.0) ** 2 + (op.shift - 0.5) ** 2)  # noqa: E731
    res = tune_tree_params(tree, score)
    assert res.improved is False
    assert res.score_after == res.score_before
    assert op.temperature == 2.0 and op.shift == 0.5  # restored bit-exactly


def test_structure_is_invariant_under_tuning():
    a = _vn([[0.6], [0.4], [0.7]], "A")
    inner = WeightedLogitMeanNode([_vn([[0.8], [0.3], [0.2]], "C"), _vn([[0.5], [0.6], [0.9]], "D")], [1.0, 1.0, 1.0])
    a.add_child(inner)
    tree = Tree.create_tree_from_root(a)
    tree.update_nodes()
    op_types_before = sorted(type(n).__name__ for n in tree.nodes["op_nodes"])
    val_ids_before = sorted(n.id for n in tree.nodes["value_nodes"])
    n_before = tree.nodes_count

    tune_tree_params(tree, lambda t: float(-_mse(t.predict(), np.array([[1.0], [0.0], [1.0]]))))

    tree.update_nodes()
    assert tree.nodes_count == n_before
    assert sorted(type(n).__name__ for n in tree.nodes["op_nodes"]) == op_types_before
    assert sorted(n.id for n in tree.nodes["value_nodes"]) == val_ids_before


def test_tuning_is_deterministic():
    results = []
    for _ in range(2):
        tree, op = _logit_tree(0.5, 0.0)
        tune_tree_params(tree, lambda t, op=op: -((op.temperature - 4.0) ** 2 + op.shift**2))
        results.append((op.temperature, op.shift))
    assert results[0] == results[1]


def test_improves_a_real_weighted_mean_prediction():
    # Parent slot (A) carries the good signal; both children are misleading. Optimal
    # tuning shifts weight onto the parent, so the convex blend approaches the target.
    y = np.array([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])
    good = np.where(y > 0.5, 0.9, 0.1)
    bad1 = np.where(y > 0.5, 0.2, 0.8)
    bad2 = np.full_like(y, 0.5)
    a = _vn(good, "A")
    a.add_child(WeightedMeanNode([_vn(bad1, "C"), _vn(bad2, "D")], [1 / 3, 1 / 3, 1 / 3]))
    tree = Tree.create_tree_from_root(a)
    op = a.children[0]

    mse_before = _mse(tree.predict(), y)
    res = tune_tree_params(tree, lambda t: float(-_mse(t.predict(), y)))
    mse_after = _mse(tree.predict(), y)

    assert res.improved is True
    assert mse_after < mse_before
    assert op._weights[0] > 1 / 3  # weight moved onto the trustworthy parent slot
    assert abs(sum(op._weights) - 1.0) < 1e-9  # simplex preserved
