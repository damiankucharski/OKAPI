from inspect import signature

import numpy as np
import pytest
from scipy.special import expit, logit

from okapi import Okapi
from okapi.backend.backend import Backend
from okapi.globals import _passthrough, use_postprocessing_function
from okapi.node import BinaryAffineNode, MeanNode, ValueNode
from okapi.operators import BINARY_AFFINE, CLOSE_THRESHOLD, FAR_THRESHOLD, MAX, MEAN, MIN, WEIGHTED_MEAN
from okapi.tree import Tree


@pytest.fixture(params=("numpy", "torch"))
def backend(request, monkeypatch):
    if request.param == "torch":
        pytest.importorskip("torch")
    monkeypatch.setattr(Backend, "_current_backend", Backend.get_backend())
    Backend.set_backend(request.param)
    return Backend.get_backend()


@pytest.mark.parametrize("transform", ["logit", "probability"])
def test_signed_reduction_respects_slot_order_bias_and_copy(backend, transform):
    a = np.array([[0.0], [0.4], [1.0]])
    b = np.array([[0.2], [0.7], [0.5]])
    weights, bias = [-0.75, 1.5], 0.2
    # The incoming parent already holds a mean of the first two inputs.
    left = (a + b) / 2
    features = np.stack([left, b])
    if transform == "logit":
        features = logit(np.clip(features, 1e-6, 1 - 1e-6))
    expected = expit(weights[0] * features[0] + weights[1] * features[1] + bias)
    leaf = ValueNode(None, backend.tensor(b), "b")
    root = ValueNode([MeanNode([leaf])], backend.tensor(a), "a")
    root.add_child(BinaryAffineNode([leaf.copy()], weights, bias, transform))
    tree = Tree(root)
    with use_postprocessing_function(_passthrough):
        direct = BinaryAffineNode(None, weights, bias, transform).op(backend.tensor(np.stack([left, b])))
        for actual in (tree.predict(), tree.copy().predict(), direct):
            np.testing.assert_allclose(backend.to_numpy(actual), expected, rtol=2e-14)
    assert tree.copy().op_nodes[-1].coefficients is not tree.op_nodes[-1].coefficients


def test_gp_child_bookkeeping_and_default_palette():
    init = signature(Okapi).parameters["allowed_ops"].default
    assert init == (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD)
    assert BINARY_AFFINE is BinaryAffineNode
    a, b, c = [ValueNode(None, np.array([0.5]), i) for i in range(3)]
    node = BinaryAffineNode([a], [-0.25, 0.75])
    node.add_child(b)
    added = node.coefficients[-1]
    node.remove_child(a)
    node.replace_child(b, c)
    assert node.children == [c] and node.coefficients == [-0.25, added]
    assert node.mutate_params(0.0) is False


@pytest.mark.parametrize(
    "weights,intercept,transform",
    [
        ([np.inf], 0.1, "logit"),
        ([0.5], np.nan, "logit"),
        ([0.5], 0.1, "other"),
        ([0.5, 0.5], 0.0, "logit"),
    ],
)
def test_reject_invalid_head(weights, intercept, transform):
    with pytest.raises(ValueError):
        BinaryAffineNode([], weights, intercept, transform)
