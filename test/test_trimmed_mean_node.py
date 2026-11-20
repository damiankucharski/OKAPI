import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.globals import postprocessing_function
from okapi.node import Node, ValueNode
from okapi.operators import TRIMMED_MEAN
from okapi.tree import Tree


@pytest.fixture(params=["numpy", "pytorch"])
def backend(request):
    previous = Backend.get_backend()
    Backend.set_backend(request.param)
    try:
        yield Backend.get_backend()
    finally:
        Backend._current_backend = previous


@pytest.mark.parametrize("count", [1, 2, 3, 4, 9])
def test_trimmed_mean_agrees_with_sorted_reference(backend, count):
    rng = np.random.default_rng(41)
    # Quantization produces frequent ties at either end.
    values = rng.integers(0, 5, size=(count, 3, 7)).astype(np.float32) / 4
    sorted_values = np.sort(values, axis=0)
    expected = (sorted_values if count < 3 else sorted_values[1:-1]).mean(axis=0)
    result = TRIMMED_MEAN(None).op(backend.tensor(values))
    np.testing.assert_allclose(backend.to_numpy(result), expected)


def test_three_inputs_preserve_small_positive_middle(backend):
    values = np.array([[1, 0], [0, 1], [1e-12, 0]], dtype=np.float32)
    result = TRIMMED_MEAN(None).op(backend.tensor(values))
    np.testing.assert_array_equal(backend.to_numpy(result), values[-1])


def test_tree_uses_parent_evaluation_and_eager_cleanup(backend, monkeypatch):
    values = [backend.tensor(np.array([v], dtype=np.float32)) for v in (0.0, 0.3, 0.7)]
    root, left, right = [ValueNode(None, v, i) for i, v in enumerate(values)]
    operator = TRIMMED_MEAN.create_node([left, right])
    root.add_child(operator)
    tree = Tree(root)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: 2 * x)
    np.testing.assert_allclose(backend.to_numpy(tree.copy().predict()), [0.6])
    assert operator.copy().code == operator.code == "TRIM1"
    monkeypatch.setattr(Node, "_EAGER_FREE_EVALS", True)
    root.evaluation = backend.tensor(np.array([0.9], dtype=np.float32))
    np.testing.assert_allclose(backend.to_numpy(operator.calculate()), [1.4])
    assert all(node.evaluation is None for node in (root, left, right))
    np.testing.assert_array_equal(backend.to_numpy(root.value), [0.0])
