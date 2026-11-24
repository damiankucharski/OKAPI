from inspect import signature

import numpy as np
import pytest
from scipy.special import softmax

from okapi import Okapi
from okapi.backend.backend import Backend
from okapi.globals import use_postprocessing_function
from okapi.node import MulticlassAffineNode, ValueNode
from okapi.operators import MAX, MULTICLASS_AFFINE
from okapi.random_state import use_rng
from okapi.tree import Tree


@pytest.fixture(params=["numpy", "pytorch"])
def backend(request, monkeypatch):
    if request.param == "pytorch":
        pytest.importorskip("torch")
    monkeypatch.setattr(Backend, "_current_backend", Backend.get_backend())
    Backend.set_backend(request.param)
    with use_postprocessing_function(lambda p: p):
        yield Backend.get_backend()


def manual_parameters(mode):
    bias = np.array([-0.21, 0.33, -0.14])
    full = np.array([[[0.8, -0.1, 0.3], [0.9, 0.7, -0.5], [-0.4, 0.2, 0.5]],
                     [[-0.2, 0.8, 0.6], [0.3, 0.4, -1.1], [0.6, -0.7, 0.3]]])
    return (np.diagonal(full, axis1=1, axis2=2).copy() if mode == "diagonal" else full), bias


def reference(columns, weights, bias):
    # Explicit scalar class loops give an independent check of the matrix axes.
    x = np.asarray(columns, dtype=np.float64)
    scores = np.broadcast_to(bias, x.shape[1:]).copy()
    for provider in range(len(x)):
        for target in range(len(bias)):
            if weights.ndim == 2:
                scores[:, target] += x[provider, :, target] * weights[provider, target]
            else:
                for source in range(len(bias)):
                    scores[:, target] += x[provider, :, source] * weights[provider, source, target]
    return softmax(scores, axis=-1)


@pytest.mark.parametrize("mode", ["diagonal", "full"])
def test_predict_copy_and_saved_architecture_on_fresh_rows(backend, mode, tmp_path):
    x = np.array([[[1, 0, 0], [.5, .2, .3], [.1, .6, .3]],
                  [[0, 0, 1], [.9, .1, 0], [.7, .25, .05]]], dtype=np.float32)
    weights, bias = manual_parameters(mode)
    root, companion = [ValueNode(None, backend.tensor(col), i) for i, col in enumerate(x)]
    node = MULTICLASS_AFFINE([companion], weights, bias, mode)
    root.add_child(node)
    tree = Tree(root)
    copy = tree.copy()
    key = tree.structural_key()
    code = node.code
    expected = reference(x, weights, bias)
    for prediction in (node.op(backend.tensor(x)), tree.predict(), copy.predict()):
        actual = backend.to_numpy(prediction)
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-16)
    np.testing.assert_allclose(expected.sum(axis=1), 1)
    assert copy.structural_key() == key

    path = tmp_path / "affine.pkl"
    tree.save_tree_architecture(path)
    loaded = Tree.load_tree_architecture(path)
    assert all(n.value is None and n.evaluation is None for n in loaded.value_nodes)
    assert loaded.structural_key() == key
    assert loaded.root.children[-1].code == code
    fresh = np.stack((x[0, :2, ::-1], x[1, :2][:, [1, 2, 0]]))
    rebound = loaded.do_pred_on_another_tensors(current_tensors={i: backend.tensor(v) for i, v in enumerate(fresh)})
    np.testing.assert_allclose(backend.to_numpy(rebound), reference(fresh, weights, bias), rtol=2e-15, atol=2e-16)

    node.intercept[0] = np.nextafter(node.intercept[0], np.inf)
    assert tree.structural_key() != key and copy.structural_key() == key
    assert node.code != code and copy.root.children[-1].code == code
    node.intercept[0] = bias[0]
    node.coefficients.flat[0] = np.nextafter(weights.flat[0], np.inf)
    assert tree.structural_key() != key and copy.structural_key() == key
    assert node.code != code and copy.root.children[-1].code == code


def test_full_node_transfers_evidence_across_classes(backend):
    # Both rows assign class 0 exactly zero. A diagonal predictor with
    # constant scores for classes 1 and 2 cannot distinguish them.
    x = np.array([[[0, 1, 0], [0, 0, 1]]], dtype=np.float32)
    weights = np.zeros((1, 3, 3))
    weights[0, 1, 0], weights[0, 2, 0] = 4, -4
    direct = MULTICLASS_AFFINE(None, weights, np.zeros(3), "full")
    p = backend.to_numpy(direct.op(backend.tensor(x)))
    np.testing.assert_allclose(p[:, 0], np.exp([4., -4.]) / (np.exp([4., -4.]) + 2))
    assert p[0, 0] > .9 and p[1, 0] < .01


def test_parent_evaluation_and_each_postprocessing_step(backend):
    x = np.array([[[.4, .1, .5]], [[.15, .8, .05]], [[.55, .05, .4]]])
    root, a, b = [ValueNode(None, backend.tensor(v), i) for i, v in enumerate(x)]
    weights, bias = manual_parameters("diagonal")
    seen = []

    def normalize(p):
        seen.append(backend.to_numpy(p).copy())
        return p / p.sum(axis=1, keepdims=True)

    root.add_child(MAX([a]))
    root.add_child(MULTICLASS_AFFINE([b], weights, bias))
    with use_postprocessing_function(normalize):
        result = backend.to_numpy(Tree(root).predict())
    parent = np.maximum(x[0], x[1])
    expected = reference([parent / parent.sum(axis=1, keepdims=True), x[2]], weights, bias)
    assert len(seen) == 2
    np.testing.assert_allclose(seen[0], parent)
    np.testing.assert_allclose(seen[1], expected)
    np.testing.assert_allclose(result, expected)


def test_extreme_intercepts_produce_finite_probabilities(backend):
    node = MULTICLASS_AFFINE(None, [[0, 0, 0]], [-1e4, 1e4, 1e4])
    p = backend.to_numpy(node.op(backend.tensor(np.ones((1, 3, 3)) / 3)))
    np.testing.assert_array_equal(p, np.broadcast_to([0., .5, .5], (3, 3)))


@pytest.mark.parametrize("mode", ["diagonal", "full"])
def test_opt_in_parameter_mutation_and_operand_slots(backend, mode):
    assert MULTICLASS_AFFINE is MulticlassAffineNode
    assert MULTICLASS_AFFINE not in signature(Okapi).parameters["allowed_ops"].default
    cols = np.array([[[1., 0, 0]], [[0, 1, 0]], [[0, 0, 1]]])
    leaves = [ValueNode(None, backend.tensor(x), i) for i, x in enumerate(cols)]
    weights, bias = manual_parameters(mode)
    op = MULTICLASS_AFFINE([leaves[1]], weights, bias, mode)
    with use_rng(np.random.RandomState(91)):
        op.add_child(leaves[2])
        np.testing.assert_array_equal(op.coefficients[:2], weights)
        added = op.coefficients[-1].copy()
        assert op.remove_child(leaves[1]) is leaves[1]
        np.testing.assert_array_equal(op.coefficients, [weights[0], added])
        assert not op.mutate_params(0) and op.mutate_params(.2)
        assert not np.array_equal(op.intercept, bias)
        assert not np.array_equal(op.coefficients[-1], added)
        initialized = MULTICLASS_AFFINE.create_node([leaves[1], leaves[2]])
    assert initialized.coefficients.shape == (3, 3)
    assert initialized.op(backend.tensor(cols)).shape == (1, 3)


@pytest.mark.parametrize(("weights", "bias", "mode"), [
    ([], [0, 0], "diagonal"), ([[0, 0]], [0, 0], "other"), ([[0]], [0], "diagonal"),
    ([[0, np.inf]], [0, 0], "diagonal"), ([[0, 0]], [0, np.nan], "diagonal"),
    (np.zeros((1, 2, 3)), [0, 0], "full"), (np.zeros((2, 2)), [0, 0], "diagonal"),
])
def test_rejects_invalid_parameters_or_arity(weights, bias, mode):
    with pytest.raises(ValueError):
        MULTICLASS_AFFINE([], weights, bias, mode)
