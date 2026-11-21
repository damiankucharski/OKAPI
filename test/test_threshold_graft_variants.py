import random

import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function
from okapi.node import CloseThresholdNode, FarThresholdNode, MeanNode, ThresholdNode, ValueNode
from okapi.random_state import use_rng
from okapi.threshold_graft import propose_threshold_graft, threshold_graft_variant
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def binary_numpy(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)


def bindings():
    return {-7: np.array([.95, .05, .43, .57]), 7: np.array([.42, .58, .90, .10]),
            "7": np.array([.31, .41, .39, .29]), "root": np.array([.53, .7, .75, .12])}


def build_tree(providers, nested, gate_type=None, cutoff=.21, operand_id=7):
    anchor = ValueNode(None, providers[-7], -7)
    if gate_type is not None:
        anchor.add_child(gate_type([ValueNode(None, providers[operand_id], operand_id)], cutoff))
    if nested:
        # A separate occurrence of provider +7 remains in this other branch.
        prefix = MeanNode([ValueNode(None, providers["7"], "7")])
        second = MeanNode([ValueNode(None, providers[7], 7), anchor])
        root = ValueNode([prefix, second], providers["root"], "root")
    else:
        root = anchor
    return Tree(root, mutation_chance=.37)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("gate_type", [CloseThresholdNode, FarThresholdNode])
@pytest.mark.parametrize("reset, duplicate", [(False, False), (True, False), (False, True), (True, True)])
def test_variants_change_only_grafted_fields_on_fit_and_rebound_panels(nested, gate_type, reset, duplicate):
    providers = bindings()
    iid = {key: (1 - value[:3])[::-1] for key, value in providers.items()}
    source = build_tree(providers, nested, gate_type)
    path = (1, 1) if nested else ()
    source_key = source.structural_key()
    source_prediction = source.predict(clear_cache=False)
    source_cache = source.root.evaluation

    variant = threshold_graft_variant(source, path, reset_threshold=reset, duplicate_anchor=duplicate)
    operand_id = -7 if duplicate else 7
    cutoff = .5 if reset else .21
    expected = build_tree(providers, nested, gate_type, cutoff, operand_id)
    assert variant.structural_key() == expected.structural_key()
    assert variant.nodes_count == source.nodes_count and variant.depth == source.depth
    assert variant.mutation_chance == source.mutation_chance
    assert source.structural_key() == source_key and source.root.evaluation is source_cache
    assert all(node.evaluation is None for node in variant.value_nodes)
    assert not {id(node) for node in source.root.get_nodes()} & {id(node) for node in variant.root.get_nodes()}
    assert all(node.value is providers[node.id] for node in variant.value_nodes)

    for panel in (providers, iid):
        a, b = panel[-7], panel[operand_id]
        distance_a, distance_b = (a-cutoff)**2, (b-cutoff)**2
        use_a = distance_a <= distance_b if gate_type is CloseThresholdNode else distance_a >= distance_b
        selected = np.where(use_a, a, b)
        if nested:
            selected = ((panel["root"] + panel["7"]) / 2 + panel[7] + selected) / 3
        prediction = variant.predict() if panel is providers else variant.do_pred_on_another_tensors(current_tensors=panel)
        np.testing.assert_array_equal(prediction, selected)
        if duplicate:
            np.testing.assert_array_equal(prediction, build_tree(panel, nested).predict())

    anchor = variant.root.children[1].children[1] if nested else variant.root
    anchor.children[0].threshold = .99
    anchor.children[0].children[0].id = "changed"
    assert source.structural_key() == source_key
    np.testing.assert_array_equal(source.evaluation, source_prediction)


def test_path_from_public_proposal_can_recover_ungrafted_predictions():
    providers = bindings()
    source = build_tree(providers, False)
    graft = propose_threshold_graft(source, {-7: providers[-7], 7: providers[7]}, [1, 0, 1, 0],
                                    seed=12, search_operands=False)
    assert graft.loss < graft.initial_loss
    variant = threshold_graft_variant(graft.tree, graft.site_path, duplicate_anchor=True)
    np.testing.assert_array_equal(variant.predict(), source.predict())
    assert [v.id for v in variant.value_nodes] == [-7, -7]


@pytest.mark.parametrize("path", [None, "", (True,), (-1,), (1.0,), (99,), (), (1, 1, 0), (1, 1, 0, 0)])
def test_invalid_path_or_address_of_gate_or_operand_is_rejected(path):
    source = build_tree(bindings(), True, FarThresholdNode)
    with pytest.raises((TypeError, ValueError)):
        threshold_graft_variant(source, path)


def test_ambiguous_or_non_native_local_shapes_are_rejected():
    p = bindings()

    class CustomClose(CloseThresholdNode):
        pass

    leaf = lambda: ValueNode(None, p[7], 7)
    for updates in [
        [], [MeanNode([leaf()])], [CloseThresholdNode([], .3)],
        [FarThresholdNode([leaf(), leaf()], .6)], [CloseThresholdNode([leaf()], .3), MeanNode([leaf()])],
        [CloseThresholdNode([ValueNode([MeanNode([leaf()])], p[7], 7)], .3)],
        [ThresholdNode([leaf()], .3)], [CustomClose([leaf()], .3)],
    ]:
        source = Tree(ValueNode(updates, p[-7], -7))
        with pytest.raises(ValueError):
            threshold_graft_variant(source, (), duplicate_anchor=True)


def test_duplicate_runs_real_gate_postprocessing(monkeypatch):
    p = bindings()
    source = build_tree(p, False, FarThresholdNode)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda values: values / 2)
    duplicate = threshold_graft_variant(source, (), duplicate_anchor=True)
    np.testing.assert_array_equal(duplicate.predict(), p[-7] / 2)


@pytest.mark.parametrize("flags", [{"reset_threshold": 1}, {"duplicate_anchor": "False"}])
def test_reject_ambiguous_options(flags):
    with pytest.raises(TypeError):
        threshold_graft_variant(build_tree(bindings(), False, FarThresholdNode), (), **flags)


def test_no_random_draw_or_prediction_during_construction(monkeypatch):
    source = build_tree(bindings(), True, FarThresholdNode)

    def unexpected(*args, **kwargs):
        pytest.fail("Construction must not draw a seed or evaluate any provider")

    monkeypatch.setattr(ValueNode, "calculate", unexpected)
    monkeypatch.setattr(np.random, "default_rng", unexpected)
    py_state, np_state = random.getstate(), np.random.get_state()
    scoped = np.random.RandomState(33)
    scoped_state = scoped.get_state()
    with use_rng(scoped):
        threshold_graft_variant(source, (np.int32(1), 1), reset_threshold=np.bool_(True), duplicate_anchor=True)
    assert random.getstate() == py_state
    for old, current in ((np_state, np.random.get_state()), (scoped_state, scoped.get_state())):
        assert old[0] == current[0] and old[2:] == current[2:]
        np.testing.assert_array_equal(old[1], current[1])
