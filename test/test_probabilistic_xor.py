from inspect import signature
from itertools import product

import numpy as np
import pytest

from okapi import Okapi
from okapi.backend.backend import Backend
from okapi.globals import postprocessing_function
from okapi.node import Node, ProbabilisticXorNode, ValueNode
from okapi.operators import CLOSE_THRESHOLD, FAR_THRESHOLD, MAX, MEAN, MIN, PROBABILISTIC_XOR, WEIGHTED_MEAN
from okapi.tree import Tree


@pytest.fixture(params=["numpy", "pytorch"])
def backend(request, monkeypatch):
    if request.param == "pytorch":
        pytest.importorskip("torch")
    monkeypatch.setattr(Backend, "_current_backend", Backend.get_backend())
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)
    Backend.set_backend(request.param)
    return Backend.get_backend()


def odd_probability(inputs):
    p = np.asarray(inputs, dtype=np.float64)
    result = np.zeros(p.shape[1:])
    for bits in product((False, True), repeat=len(p)):
        if sum(bits) % 2:
            result += np.prod([prob if bit else 1 - prob for bit, prob in zip(bits, p, strict=True)], axis=0)
    return result


def tree_of(values, operator=PROBABILISTIC_XOR):
    root, *children = [ValueNode(None, value, index) for index, value in enumerate(values)]
    root.add_child(operator.create_node(children))
    return Tree(root)


@pytest.mark.parametrize("arity", [1, 2, 3, 5])
@pytest.mark.parametrize("column", [False, True])
def test_direct_and_tree_match_bernoulli_enumeration(backend, arity, column):
    values = np.random.default_rng(17).uniform(size=(arity, 7))
    values[:, :2] = (0, 1)
    if column:
        values = values[..., None]
    expected = odd_probability(values)
    direct = PROBABILISTIC_XOR(None).op(backend.tensor(values))
    tree = tree_of([backend.tensor(row) for row in values])
    for output in (direct, tree.predict()):
        actual = backend.to_numpy(output)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


def test_multiple_occurrences_apply_the_probability_operator_per_input(backend):
    p = np.array([0., .2, .5, .9, 1.])
    tensor = backend.tensor(p)
    root = ValueNode([PROBABILISTIC_XOR([ValueNode(None, tensor, "same")])], tensor, "same")
    np.testing.assert_allclose(backend.to_numpy(Tree(root).predict()), 2 * p * (1 - p))
    np.testing.assert_array_equal(backend.to_numpy(tensor), p)


def test_small_float32_tail_survives_endpoint_cancellation(backend):
    # The third column rounds the intermediate odd probability to one. The
    # remaining even mass is the entire probability of the final odd event.
    inputs = np.array([[1e-12, 1., 1.], [2e-12, 1., 1e-12], [0., 1e-12, 1.]], dtype=np.float32)
    expected = odd_probability(inputs)
    direct = PROBABILISTIC_XOR(None).op(backend.tensor(inputs))
    tree = tree_of([backend.tensor(row) for row in inputs])
    for value in (direct, tree.predict()):
        np.testing.assert_allclose(backend.to_numpy(value), expected, atol=0, rtol=2e-6)


@pytest.mark.parametrize("left,right", [((3,), (3, 1)), ((3, 1), (3,)), ((3,), (4,)), ((3, 2), (3, 2))])
def test_tree_rejects_misaligned_or_multiclass_scores(backend, left, right):
    tree = tree_of([backend.tensor(np.full(shape, .5)) for shape in (left, right)])
    with pytest.raises(ValueError):
        tree.predict()


def test_direct_operator_rejects_stacked_multiclass(backend):
    with pytest.raises(ValueError):
        PROBABILISTIC_XOR(None).op(backend.tensor(np.full((2, 3, 2), .5)))


def test_xor_consumes_preceding_evaluation_and_postprocesses_once(backend, monkeypatch):
    def provider(p, name):
        return ValueNode(None, backend.tensor(np.array([p])), name)

    raw_outputs = []

    def postprocess(x):
        raw_outputs.append(backend.to_numpy(x).item())
        return .5 * x + .1

    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", postprocess)
    root = provider(.2, "a")
    root.add_child(MAX([provider(.8, "b")]))
    root.add_child(PROBABILISTIC_XOR([provider(.3, "c")]))
    np.testing.assert_allclose(backend.to_numpy(Tree(root).predict()), [.35])
    np.testing.assert_allclose(raw_outputs, [.8, .5])


def test_eager_evaluation_frees_consumed_caches(backend, monkeypatch):
    monkeypatch.setattr(Node, "_EAGER_FREE_EVALS", True)
    tree = tree_of([backend.tensor(np.array([p])) for p in (.2, .3)])
    tree.root.evaluation = backend.tensor(np.array([.7]))
    out = tree.op_nodes[0].calculate()
    np.testing.assert_allclose(backend.to_numpy(out), [.58])
    assert all(leaf.evaluation is None for leaf in tree.value_nodes)
    np.testing.assert_allclose(backend.to_numpy(tree.root.value), [.2])


def test_operator_identity_copy_and_opt_in(backend):
    assert PROBABILISTIC_XOR is ProbabilisticXorNode
    defaults = signature(Okapi).parameters["allowed_ops"].default
    assert defaults == (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD)
    inputs = [backend.tensor(np.array([.2])), backend.tensor(np.array([.6]))]
    tree = tree_of(inputs)
    copied = tree.copy()
    assert isinstance(copied.op_nodes[0], ProbabilisticXorNode) and copied.op_nodes[0] is not tree.op_nodes[0]
    assert copied.structural_key() == tree.structural_key() != tree_of(inputs, MEAN).structural_key()
    np.testing.assert_allclose(backend.to_numpy(copied.predict()), [.56])
