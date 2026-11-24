import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.functions import scale_vector_to_sum_1
from okapi.globals import _passthrough, use_postprocessing_function
from okapi.node import BinaryAffineNode, MeanNode, ValueNode, WeightedLogitMeanNode
from okapi.pattern_logit import _feature_variance, fit_pattern_tree
from okapi.tree import Tree


def test_binary_intercept_and_callback_count():
    y = np.array([0, 1, 1, 1] * 8)
    x = np.full((len(y), 1), 0.5)
    x.setflags(write=False)
    predictions = []

    def score(p, labels):
        assert labels is y
        predictions.append(p.copy())
        return -np.mean((p[:, 0] - labels) ** 2)

    root = ValueNode(None, x, "constant")
    tree, info = fit_pattern_tree([root], y, objective=score)
    assert len(predictions) == info["evaluations"]
    with use_postprocessing_function(_passthrough):
        p = tree.predict()
    assert isinstance(tree.root.children[-1], BinaryAffineNode)
    assert np.any([np.array_equal(candidate, p) for candidate in predictions])
    assert -np.mean((p[:, 0] - y) ** 2) == info["score"]
    np.testing.assert_allclose(p, 0.75, atol=0.002)
    assert root.evaluation is None and not root.children


def test_binary_signed_subtree_with_discontinuous_objective():
    y = np.array([0, 1] * 8)
    x = np.where(y[:, None], 0.03, 0.93).astype(np.float32)
    extra = np.full_like(x, 0.5)
    child = ValueNode(None, extra, "half")
    first = ValueNode([MeanNode([child])], x, "inverse")
    first.evaluation = np.zeros_like(x)  # stale cache on source must be left alone
    roots = [first, child]
    source = x.copy()

    def accuracy(p, y):
        return np.mean((p[:, 0] > 0.5) == y)

    tree, info = fit_pattern_tree(roots, y, objective=accuracy)
    assert tree.root.children[-1].coefficients[0] < 0
    assert tree.root.children[-1].coefficients[1] == 0
    assert info["score"] == 1
    with use_postprocessing_function(_passthrough):
        assert accuracy(tree.predict(), y) == 1
        for node in tree.nodes["value_nodes"]:
            node.value = (1 - node.value)[:4]
        assert accuracy(tree.predict(), 1 - y[:4]) == 1
    np.testing.assert_array_equal(x, source)
    np.testing.assert_array_equal(first.evaluation, np.zeros_like(x))
    assert first.children == [child.parent]


def test_signed_multiclass_and_exported_checkpoint_match():
    rng = np.random.default_rng(941)
    y = np.arange(60) % 3
    anti = np.full((len(y), 3), 0.48, dtype=np.float32)
    anti[np.arange(len(y)), y] = 0.04
    noise = rng.dirichlet(np.ones(3), len(y)).astype(np.float32)
    for col in (anti, noise):
        col.setflags(write=False)
    roots = [ValueNode(None, input_, i) for i, input_ in enumerate((anti, noise))]
    seen = []

    def objective(p, labels):
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-7)
        seen.append(p.copy())
        rows = np.arange(len(labels))
        return float(p[rows, labels].mean())

    # The fitter scopes multiclass normalization even when the outer task is binary.
    with use_postprocessing_function(_passthrough):
        tree, info = fit_pattern_tree(roots, y, objective=objective, multiclass=True)
    assert info["evaluations"] == len(seen)
    head = tree.root.children[-1]
    assert isinstance(head, WeightedLogitMeanNode) and head._weights[0] < 0
    with use_postprocessing_function(scale_vector_to_sum_1):
        p = tree.predict()
        assert np.array_equal(p.argmax(axis=1), y)
        assert p[np.arange(len(y)), y].mean() == info["score"]
        assert any(np.array_equal(p, trial) for trial in seen)
        for node in tree.nodes["value_nodes"]:
            node.value = node.value[::-1]
        np.testing.assert_array_equal(tree.predict(), p[::-1])
    assert all(root.parent is None and root.evaluation is None for root in roots)


def test_wide_bank_evaluation_bound_and_determinism():
    x = np.array([[0.1], [0.4], [0.55], [0.8]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    roots = [ValueNode(None, x, i) for i in range(100)]

    def score(p, y):
        return -np.mean((p[:, 0] - y) ** 2)

    tree, info = fit_pattern_tree(roots, labels, objective=score)
    duplicate, duplicate_info = fit_pattern_tree(roots, labels, objective=score)
    assert info == duplicate_info
    assert info["evaluations"] < 1450
    assert tree.structural_key() == duplicate.structural_key()


@pytest.mark.parametrize("x,y,multi", [
    (np.ones((3, 1)), [0, 0, 2], False),
    (np.ones((3, 1)), [0, 0.5, 1], False),
    (np.ones((3, 1)), [0, 1, 0], True),
    (np.zeros((3, 2)), [0, 1, -1], True),
    (np.full((3, 2), np.nan), [0, 1, 1], True),
])
def test_rejects_invalid_inputs(x, y, multi):
    root = ValueNode(None, x, 0)
    with pytest.raises(ValueError):
        fit_pattern_tree([root], y, objective=lambda p, labels: 0.0, multiclass=multi)


def test_one_native_evaluation_per_input(monkeypatch):
    calls = []
    original = Tree.predict

    def predict_once(self, *args, **kwargs):
        calls.append(self.root.id)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Tree, "predict", predict_once)
    x = np.array([[0.2], [0.8]])
    fit_pattern_tree([ValueNode(None, x, i) for i in range(3)], np.array([0, 1]), objective=lambda p, y: 0.0)
    assert calls == [0, 1, 2]


@pytest.mark.parametrize("multi,expected_weights,expected_calls", [
    (False, [-0.578125, -0.46875], 136),
    (True, [-0.015625, -0.3125], 82),
])
def test_zero_strength_preserves_original_search(multi, expected_weights, expected_calls):
    rng = np.random.default_rng(319)
    x = [rng.dirichlet([2, 2, 2], size=30) if multi else rng.uniform(0.05, 0.95, (30, 1)) for _ in range(2)]
    y = rng.integers(3 if multi else 2, size=30)
    target = np.eye(3)[y] if multi else y[:, None]
    roots = [ValueNode(None, col, i) for i, col in enumerate(x)]
    traces = []
    for kwargs in ({}, {"l2_strength": 0.0}):
        predictions = []

        def score(p, labels, predictions=predictions):
            predictions.append(p.copy())
            return -np.mean((p - target) ** 2)

        tree, info = fit_pattern_tree(roots, y, objective=score, multiclass=multi, **kwargs)
        head = tree.root.children[-1]
        assert (head._weights if multi else head.coefficients) == expected_weights
        if not multi:
            assert head.intercept == 0.1875
        assert info.keys() == {"score", "evaluations"}
        assert len(predictions) == info["evaluations"] == expected_calls
        traces.append(predictions)
    np.testing.assert_array_equal(*traces)


@pytest.mark.parametrize("multi", [False, True])
def test_l2_shrinks_separable_fit_and_matches_two_backends(multi, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(Backend, "_current_backend", Backend.get_backend())
    y = np.tile(np.arange(3 if multi else 2), 8)
    if multi:
        x = np.full((len(y), 3), 0.1)
        x[np.arange(len(y)), y] = 0.8
        log_p = np.log(x)
        variance = (log_p - log_p.mean(axis=1, keepdims=True)).var(axis=0).mean()
    else:
        x = np.where(y[:, None], 0.8, 0.2)
        variance = np.log(x / (1 - x)).var()
    target = np.eye(3)[y] if multi else y[:, None]
    fitted = []
    for name in ("numpy", "torch"):
        Backend.set_backend(name)
        backend = Backend.get_backend()
        root = ValueNode(None, backend.tensor(x), "provider")
        calls = []

        def score(p, labels, calls=calls, backend=backend):
            calls.append(1)
            return -float(np.mean(np.square(backend.to_numpy(p) - target)))

        slopes = []
        for strength in (0.0, 2.0):
            calls.clear()
            tree, info = fit_pattern_tree([root], y, objective=score, multiclass=multi, l2_strength=strength)
            assert info["evaluations"] == len(calls)
            head = tree.root.children[-1]
            slopes.append((head._weights if multi else head.coefficients)[0])
            if strength:
                with use_postprocessing_function(scale_vector_to_sum_1 if multi else _passthrough):
                    pred = backend.to_numpy(tree.predict())
                expected = strength / len(y) * slopes[-1] ** 2 * variance
                assert info["penalty"] == pytest.approx(expected)
                assert info["score"] == -float(np.mean(np.square(pred - target)))
                assert info["penalized_score"] == info["score"] - info["penalty"]
                fitted.append(pred)
        assert 0 < slopes[1] < slopes[0]
    np.testing.assert_allclose(*fitted, atol=1e-13)


def test_multiclass_penalty_ignores_per_row_common_log_offsets():
    logits = np.array([[0.2, 1.4, 2.1], [0.0, 1.0, 4.1], [-2.5, 0.5, 0.94]])
    shifted = logits + np.array([[2.0], [-6.0], [21.0]])
    expected = (logits - logits.mean(axis=1, keepdims=True)).var(axis=0).mean()
    assert _feature_variance(logits, True) == pytest.approx(expected)
    assert _feature_variance(shifted, True) == pytest.approx(expected)


def test_regularized_discontinuous_score_and_unpenalized_intercept():
    y = np.repeat([0, 1], 8)
    x = np.where(y[:, None], 0.1, 0.9)

    def threshold_accuracy(p, labels):
        return int(np.sum((p[:, 0] >= 0.7) == labels))

    tree, info = fit_pattern_tree([ValueNode(None, x, 0)], y, objective=threshold_accuracy, l2_strength=1.0)
    assert info["score"] == threshold_accuracy(tree.predict(), y) == len(y)
    assert 0 < info["penalty"] < info["score"]
    # With a constant logit, only the intercept is needed to fit prevalence.
    root = ValueNode(None, np.full((4, 1), 0.5), 0)

    def brier(p, labels):
        return -np.mean((p[:, 0] - labels) ** 2)

    tree, info = fit_pattern_tree([root], np.array([0, 1, 1, 1]), objective=brier, l2_strength=100.0)
    assert info["penalty"] == 0.0 and info["score"] == info["penalized_score"]
    np.testing.assert_allclose(tree.predict(), 0.75, atol=0.002)


@pytest.mark.parametrize("strength", [-0.1, np.inf, np.nan, "3", [0.1]])
def test_l2_requires_finite_nonnegative_scalar(strength):
    root = ValueNode(None, np.array([[0.5]]), 0)
    with pytest.raises(ValueError, match="l2_strength"):
        fit_pattern_tree([root], np.array([1]), objective=lambda p, y: 0.0, l2_strength=strength)
