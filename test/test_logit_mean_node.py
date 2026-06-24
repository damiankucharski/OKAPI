"""Unit tests for ``LogitMeanNode`` — a logit-space fusion operator with a
learnable temperature and shift — and its GP plumbing (create_node /
mutate_params / copy / code), mirroring the conventions in test_node.py.

The Bayes-optimal fusion of calibrated, conditionally-independent detectors is
additive in logit space; these tests pin the operator's maths (binary sigmoid of
the mean logit; temperature=K recovers the logit *sum*; multiclass geometric
mean), its overflow safety, and the hooks the evolutionary search relies on.
"""

import numpy as np
import pytest

from okapi.globals import _passthrough, set_postprocessing_function
from okapi.node import (
    LogitMeanNode,
    MeanNode,
    ValueNode,
    check_if_both_types_operators,
)
from okapi.operators import LOGIT_MEAN


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


# ------------------------------- op() maths (binary) -------------------------------


def test_binary_idempotent_when_inputs_equal():
    # all K inputs equal p -> output p (temperature=1, shift=0)
    x = np.full((3, 5, 1), 0.8)
    out = LogitMeanNode(None, 1.0, 0.0).op(x)
    np.testing.assert_allclose(out, 0.8, atol=1e-9)
    assert out.shape == (5, 1)


def test_binary_matches_reference_for_random_temperature_shift():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.01, 0.99, size=(4, 7, 1))
    t, s = 1.7, -0.4
    out = LogitMeanNode(None, t, s).op(x)
    ref = _sigmoid(t * _logit(x).mean(axis=0) + s)
    np.testing.assert_allclose(out, ref, atol=1e-9)


def test_binary_temperature_equals_K_recovers_logit_sum():
    # the independent-evidence optimum is the sum of logits; mean*K == sum
    rng = np.random.default_rng(1)
    K = 3
    x = rng.uniform(0.05, 0.95, size=(K, 6, 1))
    out = LogitMeanNode(None, float(K), 0.0).op(x)
    ref_sum = _sigmoid(_logit(x).sum(axis=0))
    np.testing.assert_allclose(out, ref_sum, atol=1e-9)


def test_binary_higher_temperature_is_more_confident():
    x = np.full((3, 1, 1), 0.8)  # mean logit > 0
    low = LogitMeanNode(None, 1.0, 0.0).op(x).item()
    high = LogitMeanNode(None, 5.0, 0.0).op(x).item()
    assert high > low > 0.5


def test_binary_shift_moves_decision_boundary():
    # at p that gives mean logit 0, output is 0.5; a positive shift pushes it up
    x = np.full((2, 1, 1), 0.5)
    assert LogitMeanNode(None, 1.0, 0.0).op(x).item() == pytest.approx(0.5)
    assert LogitMeanNode(None, 1.0, 1.0).op(x).item() > 0.5
    assert LogitMeanNode(None, 1.0, -1.0).op(x).item() < 0.5


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_binary_extremes_finite_and_in_unit_interval(p):
    x = np.full((4, 3, 1), p)  # clipped internally -> no inf
    out = LogitMeanNode(None, 5.0, 0.0).op(x)
    assert np.isfinite(out).all()
    assert (out > 0).all() and (out < 1).all()


# ----------------------------- op() maths (multiclass) -----------------------------


def test_multiclass_matches_geometric_mean_prenorm():
    rng = np.random.default_rng(2)
    x = rng.uniform(0.01, 0.99, size=(3, 5, 4))
    x = x / x.sum(axis=-1, keepdims=True)
    out = LogitMeanNode(None, 1.0, 0.0).op(x)
    ref = np.exp(np.log(x).mean(axis=0))  # geometric mean, before renormalisation
    np.testing.assert_allclose(out, ref, atol=1e-9)
    assert out.shape == (5, 4)
    assert (out > 0).all()


# --------------------------------- tree integration --------------------------------


def test_in_tree_calculate_binary():
    # A -> B(LogitMean) -> [C, D]; _concat stacks [A, C, D] -> sigmoid(mean logit)
    a = np.array([[0.6], [0.7]])
    c = np.array([[0.8], [0.2]])
    d = np.array([[0.5], [0.9]])
    A = ValueNode(None, a, "A")
    C = ValueNode(None, c, "C")
    D = ValueNode(None, d, "D")
    A.add_child(LogitMeanNode([C, D], 1.0, 0.0))

    out = A.calculate()
    ref = _sigmoid(_logit(np.stack([a, c, d], axis=0)).mean(axis=0))
    np.testing.assert_allclose(out, ref, atol=1e-9)
    assert out.shape == (2, 1)


# ------------------------------------ GP plumbing ----------------------------------


def test_create_node_positive_temperature_and_children():
    c = ValueNode(None, np.zeros((2, 1)), "c")
    node = LogitMeanNode.create_node([c])
    assert isinstance(node, LogitMeanNode)
    assert node.temperature > 0.0
    assert node.children == [c]
    assert c.parent is node


def test_mutate_params_returns_true_and_moves_params():
    np.random.seed(0)
    node = LogitMeanNode(None, 1.0, 0.0)
    t0, s0 = node.temperature, node.shift
    assert node.mutate_params() is True
    assert (node.temperature != t0) or (node.shift != s0)


def test_mutate_params_keeps_temperature_in_bounds():
    node = LogitMeanNode(None, 0.05, 0.0)
    for _ in range(200):
        node.mutate_params(0.5)
        assert 0.05 <= node.temperature <= 50.0


def test_copy_preserves_params_and_detaches():
    node = LogitMeanNode(None, 2.3, -0.7)
    cp = node.copy()
    assert isinstance(cp, LogitMeanNode)
    assert cp.temperature == 2.3 and cp.shift == -0.7
    assert cp.children == [] and cp.parent is None


def test_code_reflects_params_for_dedup():
    assert LogitMeanNode(None, 1.0, 0.0).code == "LMN[1.0,0.0]"
    # equal to one decimal -> identical code (deduplicated)
    assert LogitMeanNode(None, 1.04, 0.42).code == LogitMeanNode(None, 0.96, 0.38).code
    # materially different params -> different code
    assert LogitMeanNode(None, 1.0, 0.0).code != LogitMeanNode(None, 2.0, 0.0).code


def test_registered_as_operator_alias():
    assert LOGIT_MEAN is LogitMeanNode


def test_is_operator_for_crossover_typecheck():
    assert check_if_both_types_operators(LogitMeanNode, MeanNode) is True
    assert check_if_both_types_operators(LogitMeanNode, LogitMeanNode) is True
