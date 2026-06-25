"""Unit tests for the trust/robust fusion operators and ValueNode metadata:
``TrustGatedBlend`` (trust-weighted blend, data-driven) and ``SoftMedianNode``
(symmetric learnable robust reducer), plus the global model-metadata registry that
delivers trust to value nodes.
"""

import numpy as np
import pytest

from okapi.globals import (
    _passthrough,
    clear_model_metadata,
    set_model_metadata,
    set_postprocessing_function,
)
from okapi.node import (
    MeanNode,
    SoftMedianNode,
    TrustGatedBlend,
    ValueNode,
    check_if_both_types_operators,
)


@pytest.fixture(autouse=True)
def _reset_globals():
    set_postprocessing_function(_passthrough)
    clear_model_metadata()
    yield
    set_postprocessing_function(_passthrough)
    clear_model_metadata()


def _vn(val, id_, trust=None):
    md = None if trust is None else {"trust": trust}
    return ValueNode(None, np.asarray(val, dtype=float), id_, md)


# ------------------------------ ValueNode metadata ------------------------------


def test_valuenode_metadata_explicit_global_and_copy():
    v = _vn([[0.8]], "x", trust=0.9)
    assert v.metadata == {"trust": 0.9}
    assert v.copy().metadata == {"trust": 0.9}  # copy preserves
    set_model_metadata({"m3.npy": {"trust": 0.4}})
    assert ValueNode(None, np.zeros((2, 1)), "m3.npy").metadata == {"trust": 0.4}
    assert ValueNode(None, np.zeros((2, 1)), "unknown").metadata == {}  # default empty


# ------------------------------- TrustGatedBlend -------------------------------


def test_tgb_recovers_high_trust_and_starves_low():
    A = _vn([[0.8], [0.8]], "A", 0.9)
    A.add_child(TrustGatedBlend([_vn([[0.2], [0.2]], "C", 0.0), _vn([[0.1], [0.1]], "D", 0.0)]))
    np.testing.assert_allclose(A.calculate(), 0.8, atol=1e-9)


def test_tgb_equal_trust_equals_mean():
    A = _vn([[0.6]], "A", 0.5)
    A.add_child(TrustGatedBlend([_vn([[0.9]], "C", 0.5)]))
    np.testing.assert_allclose(A.calculate(), 0.75, atol=1e-9)


def test_tgb_zero_total_trust_is_uniform_mean():
    A = _vn([[0.2]], "A", 0.0)
    A.add_child(TrustGatedBlend([_vn([[0.8]], "C", 0.0)]))
    np.testing.assert_allclose(A.calculate(), 0.5, atol=1e-9)


def test_tgb_reads_trust_from_global_registry():
    set_model_metadata({"a.npy": {"trust": 1.0}, "b.npy": {"trust": 0.0}})
    A = ValueNode(None, np.array([[0.9]]), "a.npy")
    A.add_child(TrustGatedBlend([ValueNode(None, np.array([[0.1]]), "b.npy")]))
    np.testing.assert_allclose(A.calculate(), 0.9, atol=1e-9)  # low-trust input starved


def test_tgb_leaf_only_guard_neutralizes_internal_child():
    # Leaf-only trust guard (L2.3c): per-model trust is valid only where the streamed
    # tensor IS the base model. Parent A and leaf C both have trust 0 -> contribute
    # nothing. E is an INTERNAL child (it owns an operator subtree) carrying a *stale*
    # base-model trust of 0; the guard must give it a NEUTRAL weight so its fused value
    # (0.9) wins. Pre-fix (trust applied to all) -> total weight 0 -> uniform mean 0.3.
    A = _vn([[0.0]], "A", 0.0)
    C = _vn([[0.0]], "C", 0.0)
    E = _vn([[0.9]], "E", 0.0)  # stale base-model trust sitting on an internal node
    E.add_child(MeanNode([_vn([[0.9]], "F", 0.0)]))  # E.calculate() = mean(0.9, 0.9) = 0.9 (fused)
    A.add_child(TrustGatedBlend([C, E]))
    np.testing.assert_allclose(A.calculate(), 0.9, atol=1e-9)


def test_tgb_parent_slot_keeps_its_trust():
    # The parent slot streams parent.value (always the base model), so its trust is
    # valid and must NOT be neutralized even though the parent node has a child (the TGB).
    A = _vn([[0.8]], "A", 1.0)
    A.add_child(TrustGatedBlend([_vn([[0.0]], "C", 0.0)]))
    np.testing.assert_allclose(A.calculate(), 0.8, atol=1e-9)


def test_tgb_plumbing():
    assert TrustGatedBlend(None).code == "TGB"
    assert isinstance(TrustGatedBlend.create_node([]), TrustGatedBlend)
    assert isinstance(TrustGatedBlend(None).copy(), TrustGatedBlend)
    assert check_if_both_types_operators(TrustGatedBlend, MeanNode) is True


# -------------------------------- SoftMedianNode --------------------------------


def test_softmedian_downweights_outlier():
    out = SoftMedianNode(None, 0.05).op(np.array([[[0.1]], [[0.9]], [[0.9]]]))
    np.testing.assert_allclose(out, 0.9, atol=1e-3)  # outlier 0.1 starved -> median


def test_softmedian_large_temperature_approaches_mean():
    x = np.array([[[0.0]], [[0.6]], [[0.9]]])
    np.testing.assert_allclose(SoftMedianNode(None, 100.0).op(x), np.mean([0.0, 0.6, 0.9]), atol=1e-2)


def test_softmedian_plumbing():
    n = SoftMedianNode.create_node([])
    assert isinstance(n, SoftMedianNode) and n.temperature > 0.0
    assert n.mutate_params() is True
    assert SoftMedianNode(None, 1.23).copy().temperature == 1.23
    assert SoftMedianNode(None, 0.34).code == "SMD[0.3]"
    assert check_if_both_types_operators(SoftMedianNode, MeanNode) is True
