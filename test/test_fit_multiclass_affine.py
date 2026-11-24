import numpy as np
import pytest

from okapi.functions import scale_vector_to_sum_1
from okapi.globals import _passthrough, use_postprocessing_function
from okapi.multiclass_affine import fit_multiclass_affine_tree
from okapi.node import MaxNode, MulticlassAffineNode, ValueNode
from okapi.tree import Tree


def brier_score(p, y):
    return -np.square(p - np.eye(p.shape[1])[y]).sum(axis=1).mean()


def test_cross_class_cycle_needs_full_map_on_this_three_class_fixture():
    source = np.tile(np.arange(3), 8)
    p = np.eye(3)[source]
    y = (source + 1) % 3
    scores = {}
    for mode in ("diagonal", "full"):
        root = ValueNode(None, p, 0)
        calls = []

        def objective(prediction, labels, calls=calls):
            assert labels is y
            calls.append(brier_score(prediction, labels))
            return calls[-1]

        tree, info = fit_multiclass_affine_tree([root], y, mode=mode, objective=objective, l2_strength=0.01)
        head = tree.root.children[-1]
        assert isinstance(head, MulticlassAffineNode) and head.mode == mode
        with use_postprocessing_function(scale_vector_to_sum_1):
            predictions = tree.predict()
        assert info["score"] == brier_score(predictions, y)
        assert info["score"] in calls and info["evaluations"] == len(calls) <= info["max_evaluations"]
        variance = p.var(axis=0)
        weight_square = np.square(head.coefficients[0])
        expected = (variance if mode == "diagonal" else variance[:, None]) * weight_square
        assert info["penalty"] == pytest.approx(0.01 * expected.sum() / len(y))
        assert info["penalized_score"] == pytest.approx(info["score"] - info["penalty"])
        assert not root.children and root.evaluation is None
        scores[mode] = info["score"]
        if mode == "full":
            np.testing.assert_array_equal(predictions.argmax(axis=1), y)
    assert scores["full"] > -0.01
    assert scores["diagonal"] < -0.3


def test_diagonal_root_contains_original_program_and_rebinds_to_unseen_probabilities():
    rng = np.random.default_rng(207)
    p, q = [rng.dirichlet(np.ones(3), 40) for _ in range(2)]
    parent = ValueNode([MaxNode([ValueNode(None, q, "second")])], p, "first")
    extra = ValueNode(None, q, "second")
    y = np.maximum(p, q).argmax(axis=1)
    with use_postprocessing_function(_passthrough):
        tree, info = fit_multiclass_affine_tree([parent, extra], y, mode="diagonal", objective=brier_score, max_evaluations=47)
    assert info["evaluations"] == 47
    with use_postprocessing_function(scale_vector_to_sum_1):
        assert brier_score(tree.predict(), y) == info["score"]
        fresh = {key: rng.dirichlet(np.ones(3), 9) for key in ("first", "second")}
        actual = tree.do_pred_on_another_tensors(current_tensors=fresh)
        subtree = Tree(parent.copy_subtree()).do_pred_on_another_tensors(current_tensors=fresh)
        head = tree.root.children[-1]
        logits = head.intercept + subtree * head.coefficients[0] + fresh["second"] * head.coefficients[1]
        expected = np.exp(logits - logits.max(axis=1, keepdims=True))
        np.testing.assert_allclose(actual, expected / expected.sum(axis=1, keepdims=True), atol=1e-14)
    assert parent.parent is None and len(parent.children) == 1
    assert extra.parent is None


def test_fits_bias_with_all_constant_features_and_missing_class():
    p = np.tile([0.2, 0.3, 0.5], (21, 1))
    y = np.array([0] * 7 + [2] * 14)
    tree, info = fit_multiclass_affine_tree([ValueNode(None, p, "x")], y, mode="full", objective=brier_score, max_evaluations=200)
    assert info["score"] > -0.46
    assert info["penalty"] == 0
    with use_postprocessing_function(scale_vector_to_sum_1):
        np.testing.assert_allclose(tree.predict()[0], [1 / 3, 0, 2 / 3], atol=0.035)


def test_discontinuous_callback_budget_and_reproducibility():
    p = np.tile(np.eye(3), (3, 1))
    y = p.argmax(axis=1)
    roots = [ValueNode(None, p, "provider")]
    calls = []

    def objective(prediction, labels):
        value = (prediction.argmax(axis=1) == labels).mean()
        calls.append(value)
        return value

    tree, one = fit_multiclass_affine_tree(roots, y, objective=objective, max_evaluations=1)
    assert one["evaluations"] == len(calls) == 1
    with use_postprocessing_function(scale_vector_to_sum_1):
        np.testing.assert_allclose(tree.predict(), 1 / 3)
    tree, limited = fit_multiclass_affine_tree(roots, y, objective=objective, max_evaluations=9)
    assert limited["evaluations"] == len(calls) - 1 <= 9
    assert limited["score"] == 1.0
    repeated, info = fit_multiclass_affine_tree(roots, y, objective=objective, max_evaluations=9)
    assert info == limited
    assert tree.structural_key() == repeated.structural_key()


@pytest.mark.parametrize(
    "p, y, kwargs",
    [
        ([[0.7, 0.7]], [0], {}),
        ([[0.5, 0.5]], [2], {}),
        ([[0.5, 0.5]], [0], {"mode": "shared"}),
        ([[0.5, 0.5]], [0], {"max_evaluations": 0}),
        ([[0.5, 0.5]], [0], {"l2_strength": -1}),
    ],
)
def test_invalid_inputs(p, y, kwargs):
    with pytest.raises(ValueError):
        fit_multiclass_affine_tree([ValueNode(None, np.asarray(p), 0)], np.asarray(y), objective=brier_score, **kwargs)
