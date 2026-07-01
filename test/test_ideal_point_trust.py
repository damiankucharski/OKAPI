"""Tests for ``IdealPointTrustNode`` (#30) — supervised, position-correct trust blend.

Covers: the supervised weighting (concentrates on the input closest to the truth, at any
position), the temperature limits (hard-pick vs mean), the **fit-vs-predict** mechanism
(weights computed + cached when the eval-context ``y`` is present, reused when it is absent,
uniform fallback otherwise), ``copy()`` carrying temperature + cache (required because
prediction copies the tree), the GP plumbing (mutate/code/create_node, binary + multiclass),
the global eval-context, and the engine wiring (``_calculate_fitnesses`` sets **and clears**
the context, and a #30 tree's weights get cached during a fitness pass).
"""

import numpy as np
import pytest

from okapi.globals import clear_eval_context, get_eval_context, set_eval_context
from okapi.node import IdealPointTrustNode, ValueNode
from okapi.okapi import Okapi
from okapi.pareto import maximize
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def _clear_ctx():
    """No test should leak the eval-context into the next one."""
    clear_eval_context()
    yield
    clear_eval_context()


def _vn(value, name):
    return ValueNode(None, np.asarray(value, dtype=float), name)


def _tree(parent_val, child_vals, temperature=0.3):
    """Root parent (slot 0) -> IdealPointTrust over the given children."""
    a = _vn(parent_val, "A")
    children = [_vn(v, f"C{i}") for i, v in enumerate(child_vals)]
    a.add_child(IdealPointTrustNode(children, temperature))
    return Tree.create_tree_from_root(a), a.children[0]


def _mse(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


# --- binary truth fixtures: one diamond (~y) among agreeing-wrong garbage ---
Y = np.array([[1.0], [0.0], [1.0], [0.0]])
DIAMOND = np.array([[0.95], [0.05], [0.95], [0.05]])
G1 = np.array([[0.1], [0.9], [0.1], [0.9]])
G2 = np.array([[0.2], [0.8], [0.2], [0.8]])


# --------------------------------------------------------------------------------------
# Supervised weighting
# --------------------------------------------------------------------------------------


def test_weights_concentrate_on_the_diamond_even_as_a_child():
    # parent=garbage, the diamond is a CHILD (slot 1) -> position-correct trust must still find it
    tree, op = _tree(G1, [DIAMOND, G2])
    set_eval_context(Y)
    out = tree.predict()
    assert np.argmax(op._cached_weights) == 1  # the diamond's slot wins
    assert _mse(out, Y) < _mse(G1, Y) and _mse(out, Y) < _mse(G2, Y)  # closer to truth than the garbage


def test_temperature_zero_hard_picks_the_most_competent_input():
    tree, _ = _tree(G1, [DIAMOND, G2], temperature=0.01)
    set_eval_context(Y)
    np.testing.assert_allclose(tree.predict(), DIAMOND, atol=0.02)


def test_large_temperature_approaches_the_plain_mean():
    tree, _ = _tree(G1, [DIAMOND, G2], temperature=50.0)
    set_eval_context(Y)
    np.testing.assert_allclose(tree.predict(), (G1 + DIAMOND + G2) / 3.0, atol=0.02)


# --------------------------------------------------------------------------------------
# Fit-vs-predict caching
# --------------------------------------------------------------------------------------


def test_predict_without_y_reuses_fit_cached_weights_on_new_inputs():
    tree, op = _tree(G1, [DIAMOND, G2])
    set_eval_context(Y)
    tree.predict()  # fit pass: compute + cache weights
    cached = list(op._cached_weights)
    clear_eval_context()  # prediction time: no y

    new_p = np.array([[0.3], [0.4], [0.5], [0.6]])
    new_0 = np.array([[0.7], [0.1], [0.2], [0.9]])
    new_1 = np.array([[0.5], [0.5], [0.5], [0.5]])
    tree.root.value = new_p
    op.children[0].value, op.children[1].value = new_0, new_1
    out = tree.predict()

    manual = cached[0] * new_p + cached[1] * new_0 + cached[2] * new_1
    np.testing.assert_allclose(out, manual, atol=1e-9)  # cached weights reused, applied to NEW inputs


def test_uniform_fallback_without_y_and_without_usable_cache():
    mean = (G1 + DIAMOND + G2) / 3.0
    tree, op = _tree(G1, [DIAMOND, G2])
    np.testing.assert_allclose(tree.predict(), mean, atol=1e-9)  # no y, no cache -> neutral mean
    assert op._cached_weights is None
    op._cached_weights = [0.5, 0.5]  # stale cache of the wrong length -> still neutral
    np.testing.assert_allclose(tree.predict(), mean, atol=1e-9)


def test_copy_carries_temperature_and_cache_and_detaches():
    op = IdealPointTrustNode([_vn(np.zeros((2, 1)), "c")], 0.7)
    op._cached_weights = [0.2, 0.8]
    cp = op.copy()
    assert isinstance(cp, IdealPointTrustNode)
    assert cp.temperature == 0.7
    assert cp._cached_weights == [0.2, 0.8] and cp._cached_weights is not op._cached_weights
    assert cp.children == [] and cp.parent is None
    assert IdealPointTrustNode(None, 0.3).copy()._cached_weights is None


# --------------------------------------------------------------------------------------
# GP plumbing
# --------------------------------------------------------------------------------------


def test_mutate_params_moves_and_stays_bounded():
    np.random.seed(0)
    op = IdealPointTrustNode(None, 0.3)
    assert op.mutate_params() is True
    for _ in range(300):
        op.mutate_params(0.5)
        assert 0.01 <= op.temperature <= 5.0


def test_code_reflects_temperature_for_dedup():
    assert IdealPointTrustNode(None, 0.3).code == "IPT[0.3]"
    assert IdealPointTrustNode(None, 1.04).code == IdealPointTrustNode(None, 0.96).code  # both round to 1.0
    assert IdealPointTrustNode(None, 0.3).code != IdealPointTrustNode(None, 2.0).code


def test_create_node_defaults():
    c = _vn(np.zeros((2, 1)), "c")
    n = IdealPointTrustNode.create_node([c])
    assert isinstance(n, IdealPointTrustNode) and n.children == [c] and n.temperature == 0.3


def test_multiclass_concentrates_on_diamond_and_preserves_shape():
    y = np.array([0, 1, 2, 0])
    diamond = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.9, 0.05, 0.05]])
    uniform = np.full((4, 3), 1 / 3)
    wrong = np.array([[0.05, 0.9, 0.05]] * 4)
    tree, op = _tree(uniform, [diamond, wrong])
    set_eval_context(y)
    out = tree.predict()
    assert np.argmax(op._cached_weights) == 1
    assert out.shape == (4, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


# --------------------------------------------------------------------------------------
# Shape-aware Brier: segmentation / multilabel must use element-wise distance, not one-hot
# (regression for the STARCOP OOM: (S,H,W) preds have shape[-1] != 1 but are NOT multiclass)
# --------------------------------------------------------------------------------------


def _softmax_brier_blend(inputs, y, temp):
    d = np.array([_mse(x, y) for x in inputs])
    z = -d / max(temp, 1e-9)
    z = z - z.max()
    w = np.exp(z)
    w = w / w.sum()
    return w, sum(wi * x for wi, x in zip(w, inputs))


def test_segmentation_shaped_predictions_use_elementwise_brier():
    # Per-pixel maps (S, H, W): shape[-1] != 1 but NOT multiclass. The old `shape[-1] == 1`
    # routing sent these to the one-hot branch -> eye(width)[every_pixel] (the STARCOP 730 GB
    # OOM). Must now use element-wise Brier, keep per-pixel shape, and still find the diamond.
    y = np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])  # (2,2,3)
    diamond = np.clip(y * 0.9 + 0.05, 0.0, 1.0)
    g1, g2 = 1.0 - diamond, np.full_like(y, 0.5)
    tree, op = _tree(g1, [diamond, g2], temperature=0.3)
    set_eval_context(y)
    out = tree.predict()
    assert out.shape == y.shape  # per-pixel shape preserved (no one-hot reshape)
    assert np.argmax(op._cached_weights) == 1  # diamond slot wins
    w, manual = _softmax_brier_blend([g1, diamond, g2], y, 0.3)
    np.testing.assert_allclose(op._cached_weights, w, atol=1e-6)
    np.testing.assert_allclose(out, manual, atol=1e-6)


def test_multilabel_shaped_predictions_use_elementwise_brier():
    y = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])  # (N, C) multi-hot == pred shape
    diamond = np.clip(y * 0.9 + 0.05, 0.0, 1.0)
    g1 = 1.0 - diamond
    tree, op = _tree(g1, [diamond], temperature=0.3)
    set_eval_context(y)
    out = tree.predict()
    assert out.shape == y.shape
    assert op._cached_weights[1] > op._cached_weights[0]  # diamond child beats the wrong parent
    _, manual = _softmax_brier_blend([g1, diamond], y, 0.3)
    np.testing.assert_allclose(out, manual, atol=1e-6)


def test_streaming_blend_identical_with_eager_free_on():
    # The two-pass streaming blend (which frees inputs one at a time under eager-free) must give
    # the same output as the default path.
    from okapi.node import Node

    tree_a, _ = _tree(G1, [DIAMOND, G2], temperature=0.3)
    set_eval_context(Y)
    out_off = tree_a.predict()
    tree_b, _ = _tree(G1, [DIAMOND, G2], temperature=0.3)
    set_eval_context(Y)
    prev = Node._EAGER_FREE_EVALS
    Node._EAGER_FREE_EVALS = True
    try:
        out_on = tree_b.predict()
    finally:
        Node._EAGER_FREE_EVALS = prev
    np.testing.assert_allclose(out_on, out_off, atol=1e-9)


# --------------------------------------------------------------------------------------
# Global eval-context + engine wiring
# --------------------------------------------------------------------------------------


def test_eval_context_set_get_clear():
    assert get_eval_context() is None
    y = np.array([1, 0, 1])
    set_eval_context(y)
    assert get_eval_context() is y
    clear_eval_context()
    assert get_eval_context() is None


def _engine(tmp_path, n=200, n_models=6):
    rng = np.random.default_rng(0)
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir(parents=True)
    for k in range(n_models):
        np.save(fit_dir / f"m{k}.npy", rng.random((n, 1)).astype(np.float64))
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, (rng.random(n) > 0.5).astype(np.int64))

    def _acc(prediction, gt):
        p = np.asarray(prediction)
        y = np.asarray(gt).ravel().astype(int)
        yhat = (p[:, 0] > 0.5).astype(int) if p.shape[1] == 1 else p.argmax(axis=1)
        return float((yhat == y).mean())

    return Okapi(preds_source=fit_dir, gt_path=gt_path, population_size=6, population_multiplier=3,
                 tournament_size=4, minimize_node_count=False, objective_functions=(_acc,),
                 objectives=(maximize,), backend="numpy", seed=0, postprocessing_function=lambda x: x)


def test_calculate_fitnesses_sets_then_clears_context_and_caches_weights(tmp_path):
    ok = _engine(tmp_path, n=200)
    rng = np.random.default_rng(1)
    a = _vn(rng.random((200, 1)), "A")
    a.add_child(IdealPointTrustNode([_vn(rng.random((200, 1)), "C"), _vn(rng.random((200, 1)), "D")]))
    tree = Tree.create_tree_from_root(a)

    assert get_eval_context() is None
    ok._calculate_fitnesses([tree])
    assert get_eval_context() is None  # set during the pass, cleared in finally (no leak to predict)

    op = tree.root.children[0]
    assert op._cached_weights is not None and len(op._cached_weights) == 3  # y was available during fit
