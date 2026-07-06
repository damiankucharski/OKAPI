"""Unit tests for ``WeightedLogitMeanNode`` -- per-input weighting in *logit*
space (the calibrated-stacking link), the natural join of ``LogitMeanNode``
(global temperature on the mean logit) and ``WeightedMeanNode`` (per-input
weights in probability space).

These pin: the binary ``sigmoid(Sum w_i logit x_i)`` and multiclass weighted
product maths; that equal weights ``t/K`` reproduce ``LogitMeanNode(t)`` exactly
(the strict-superset claim); overflow safety; the GP hooks (create_node /
mutate_params / copy / code); and -- specific to this node -- the *unnormalised*
weight bookkeeping on add/remove/replace and the copy_subtree round-trip, plus an
engine-evolution smoke test that the weight vector never desyncs from the children
under real crossover + mutation.
"""

import numpy as np
import pytest

from okapi.globals import _passthrough, set_postprocessing_function
from okapi.node import (
    LogitMeanNode,
    MeanNode,
    ValueNode,
    WeightedLogitMeanNode,
    WeightedMeanNode,
    check_if_both_types_operators,
)
from okapi.okapi import Okapi
from okapi.operators import WEIGHTED_LOGIT_MEAN
from okapi.pareto import maximize
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def _default_passthrough_postprocessing():
    """The global postprocessing function is shared state; ensure the default
    passthrough so the maths assertions hold regardless of test ordering."""
    set_postprocessing_function(_passthrough)
    yield
    set_postprocessing_function(_passthrough)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    return np.log(p) - np.log(1.0 - p)


def _identity(x):
    return x


def _acc(prediction, gt):
    p = np.asarray(prediction)
    y = np.asarray(gt).ravel().astype(int)
    yhat = (p[:, 0] > 0.5).astype(int) if p.shape[1] == 1 else p.argmax(axis=1)
    return float((yhat == y).mean())


# ------------------------------- op() maths (binary) -------------------------------


def test_binary_matches_weighted_logit_sum_reference():
    rng = np.random.default_rng(0)
    K = 3
    x = rng.uniform(0.01, 0.99, size=(K, 7, 1))
    w = [0.5, 1.5, 0.8]
    out = WeightedLogitMeanNode(None, w).op(x)
    ref = _sigmoid((np.array(w).reshape(K, 1, 1) * _logit(x)).sum(axis=0))
    np.testing.assert_allclose(out, ref, atol=1e-9)
    assert out.shape == (7, 1)


def test_equal_weights_t_over_K_recover_logit_mean_node():
    # The strict-superset claim: all weights == t/K is exactly LogitMeanNode(t, 0).
    rng = np.random.default_rng(3)
    K = 4
    x = rng.uniform(0.02, 0.98, size=(K, 6, 1))
    t = 2.3
    out = WeightedLogitMeanNode(None, [t / K] * K).op(x)
    ref = LogitMeanNode(None, t, 0.0).op(x)
    np.testing.assert_allclose(out, ref, atol=1e-9)


def test_idempotent_when_weights_sum_to_one_and_inputs_equal():
    K = 4
    x = np.full((K, 5, 1), 0.8)
    out = WeightedLogitMeanNode(None, [1.0 / K] * K).op(x)
    np.testing.assert_allclose(out, 0.8, atol=1e-9)
    assert out.shape == (5, 1)


def test_unit_weights_give_logit_sum_and_sharpen():
    # weights all 1 -> Sum logit = K*logit for equal inputs -> more confident than the input
    K, p = 3, 0.7
    x = np.full((K, 1, 1), p)
    out = WeightedLogitMeanNode(None, [1.0] * K).op(x).item()
    ref = _sigmoid(K * _logit(np.array(p)))
    assert out == pytest.approx(float(ref))
    assert out > p


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_binary_extremes_finite_and_in_unit_interval(p):
    x = np.full((4, 3, 1), p)  # clipped internally -> no inf
    out = WeightedLogitMeanNode(None, [2.0, 2.0, 2.0, 2.0]).op(x)
    assert np.isfinite(out).all()
    assert (out > 0).all() and (out < 1).all()


def test_all_zero_weights_collapse_to_half_not_nan():
    # Degenerate but legal (mutation can floor every weight at 0): Sum = 0 -> sigmoid(0).
    x = np.full((3, 4, 1), 0.9)
    out = WeightedLogitMeanNode(None, [0.0, 0.0, 0.0]).op(x)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, 0.5, atol=1e-9)


# ----------------------------- op() maths (multiclass) -----------------------------


def test_multiclass_matches_weighted_product_prenorm():
    rng = np.random.default_rng(2)
    K = 3
    x = rng.uniform(0.01, 0.99, size=(K, 5, 4))
    x = x / x.sum(axis=-1, keepdims=True)
    w = [0.7, 1.2, 0.5]
    out = WeightedLogitMeanNode(None, w).op(x)
    ref = np.exp((np.array(w).reshape(K, 1, 1) * np.log(x)).sum(axis=0))
    np.testing.assert_allclose(out, ref, atol=1e-9)
    assert out.shape == (5, 4)
    assert (out > 0).all()


# --------------------------------- tree integration --------------------------------


def test_in_tree_calculate_binary():
    # A -> WLMN -> [C, D]; _concat stacks [A, C, D]; parent A is weight slot 0.
    a = np.array([[0.6], [0.7]])
    c = np.array([[0.8], [0.2]])
    d = np.array([[0.5], [0.9]])
    A = ValueNode(None, a, "A")
    C = ValueNode(None, c, "C")
    D = ValueNode(None, d, "D")
    w = [0.9, 1.1, 0.7]
    A.add_child(WeightedLogitMeanNode([C, D], w))

    out = A.calculate()
    stacked = np.stack([a, c, d], axis=0)
    ref = _sigmoid((np.array(w).reshape(3, 1, 1) * _logit(stacked)).sum(axis=0))
    np.testing.assert_allclose(out, ref, atol=1e-9)
    assert out.shape == (2, 1)


@pytest.mark.parametrize("shape", [(5, 1), (4, 3), (4, 3, 2)])
def test_calculate_streaming_matches_op_concat(shape):
    rng = np.random.default_rng(10)
    a = rng.uniform(0.01, 0.99, size=shape)
    c = rng.uniform(0.01, 0.99, size=shape)
    d = rng.uniform(0.01, 0.99, size=shape)
    weights = [0.9, 1.1, 0.7]
    node = WeightedLogitMeanNode(None, weights)
    ref = node.op(np.stack([a, c, d], axis=0))

    A = ValueNode(None, a, "A")
    C = ValueNode(None, c, "C")
    D = ValueNode(None, d, "D")
    A.add_child(WeightedLogitMeanNode([C, D], weights))

    np.testing.assert_allclose(A.calculate(), ref, atol=1e-9)


# ------------------------------------ GP plumbing ----------------------------------


def test_create_node_positive_weights_correct_length_and_children():
    c = ValueNode(None, np.zeros((2, 1)), "c")
    node = WeightedLogitMeanNode.create_node([c])
    assert isinstance(node, WeightedLogitMeanNode)
    assert len(node._weights) == 2  # parent slot + one child
    assert all(w > 0.0 for w in node._weights)
    assert node.children == [c]
    assert c.parent is node


def test_mutate_params_returns_true_moves_and_stays_bounded():
    np.random.seed(0)
    node = WeightedLogitMeanNode(None, [1.0, 1.0, 1.0])
    before = list(node._weights)
    assert node.mutate_params() is True
    assert node._weights != before
    assert len(node._weights) == 3
    for _ in range(300):
        node.mutate_params(0.5)
        assert len(node._weights) == 3
        assert all(0.0 <= w <= 50.0 for w in node._weights)


def test_copy_preserves_weights_and_detaches():
    node = WeightedLogitMeanNode(None, [0.3, 1.7, 2.1])
    cp = node.copy()
    assert isinstance(cp, WeightedLogitMeanNode)
    assert cp._weights == [0.3, 1.7, 2.1]
    assert cp._weights is not node._weights  # independent list
    assert cp.children == [] and cp.parent is None


def test_code_reflects_weights_for_dedup():
    assert WeightedLogitMeanNode(None, [1.0, 0.5]).code == "WLMN[1.0,0.5]"
    # equal to one decimal -> identical code (deduplicated by Okapi.run_iteration)
    assert WeightedLogitMeanNode(None, [1.04, 0.48]).code == WeightedLogitMeanNode(None, [0.96, 0.52]).code
    # materially different weights -> different code
    assert WeightedLogitMeanNode(None, [1.0, 0.5]).code != WeightedLogitMeanNode(None, [2.0, 0.5]).code


def test_registered_as_operator_alias():
    assert WEIGHTED_LOGIT_MEAN is WeightedLogitMeanNode


def test_is_operator_for_crossover_typecheck():
    assert check_if_both_types_operators(WeightedLogitMeanNode, MeanNode) is True
    assert check_if_both_types_operators(WeightedLogitMeanNode, WeightedMeanNode) is True


# ----------------------------- weight bookkeeping (structural) ----------------------


def test_add_child_appends_free_weight_without_rescaling():
    c1 = ValueNode(None, np.zeros((2, 1)), "c1")
    node = WeightedLogitMeanNode([c1], [1.0, 0.5])  # parent, c1
    c2 = ValueNode(None, np.zeros((2, 1)), "c2")
    node.add_child(c2)
    assert len(node.children) == 2
    assert len(node._weights) == 3
    assert node._weights[:2] == [1.0, 0.5]  # existing weights untouched (unnormalised)
    assert node._weights[2] > 0.0


def test_remove_child_pops_the_aligned_weight():
    c1 = ValueNode(None, np.zeros((2, 1)), "c1")
    c2 = ValueNode(None, np.zeros((2, 1)), "c2")
    node = WeightedLogitMeanNode([c1, c2], [0.9, 0.3, 0.7])  # parent, c1, c2
    node.remove_child(c1)  # c1 is slot 1
    assert node.children == [c2]
    assert node._weights == [0.9, 0.7]  # parent (slot 0) and c2's weight remain, aligned


def test_length_assertion_fires_on_desync():
    node = WeightedLogitMeanNode([ValueNode(None, np.zeros((2, 1)), "c")], [1.0, 0.5])
    node._weights.append(0.3)  # deliberately desync
    with pytest.raises(AssertionError):
        node._weight_length_assertion()


def test_copy_subtree_roundtrip_preserves_eval_and_length():
    a = np.array([[0.6], [0.4]])
    c = np.array([[0.7], [0.2]])
    d = np.array([[0.3], [0.9]])
    A = ValueNode(None, a, "A")
    C = ValueNode(None, c, "C")
    D = ValueNode(None, d, "D")
    A.add_child(WeightedLogitMeanNode([C, D], [0.8, 1.2, 0.6]))
    tree = Tree.create_tree_from_root(A)

    out_orig = np.asarray(tree.evaluation)
    copied = tree.copy()  # uses copy() ([] children + full weights) + copy_subtree re-attach
    out_copy = np.asarray(copied.evaluation)
    np.testing.assert_allclose(out_copy, out_orig, atol=1e-12)

    op = copied.root.children[0]
    assert isinstance(op, WeightedLogitMeanNode)
    assert len(op._weights) == len(op.children) + 1 == 3


# ------------------------------------ engine smoke ---------------------------------


def test_engine_evolves_with_wlmn_without_weight_desync(tmp_path):
    # Real crossover + structural mutation must keep every WLMN's weight vector in
    # sync with its children (the bookkeeping above, exercised end-to-end).
    rng = np.random.default_rng(0)
    n_models = 8
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir(parents=True)
    for k in range(n_models):
        np.save(fit_dir / f"m{k}.npy", rng.random((200, 1)).astype(np.float64))
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, (rng.random(200) > 0.5).astype(np.int64))

    ok = Okapi(
        preds_source=fit_dir,
        gt_path=gt_path,
        population_size=8,  # OKAPI seeds one tree per model, so must be <= n_models
        population_multiplier=3,
        tournament_size=4,
        minimize_node_count=False,
        objective_functions=(_acc,),
        objectives=(maximize,),
        allowed_ops=(MeanNode, WeightedLogitMeanNode),
        backend="numpy",
        seed=1,
        postprocessing_function=_identity,
    )

    # Seed WLMN-bearing trees so crossover (copy_subtree) and structural mutation
    # (append/prune children) actually manipulate WLMN weight vectors from gen 0.
    # Any desync trips _weight_length_assertion mid-train and fails this test;
    # surviving to the final Pareto front is not required (selection may cull them).
    ids, models = ok.ids, ok.models
    for _ in range(4):
        pick = rng.choice(n_models, size=3, replace=False)
        root = ValueNode(None, models[pick[0]], ids[pick[0]])
        kids = [ValueNode(None, models[pick[1]], ids[pick[1]]),
                ValueNode(None, models[pick[2]], ids[pick[2]])]
        root.add_child(WeightedLogitMeanNode.create_node(kids))
        ok.population.append(Tree.create_tree_from_root(root))
    ok.population_size = len(ok.population)
    ok.fitnesses = None

    ok.train(25)

    assert ok.population
    for t in ok.population:
        pred = np.asarray(t.predict())
        assert np.isfinite(pred).all()
        for n in t.nodes["op_nodes"]:
            if isinstance(n, WeightedLogitMeanNode):
                assert len(n._weights) == len(n.children) + 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
