import numpy as np
import pytest

from experiments.point_ablation.moves import MatchedPointControls
from okapi import Okapi
from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function
from okapi.interventions import operator_paths
from okapi.mutation import replace_operator_mutation
from okapi.node import CloseThresholdNode, FarThresholdNode, MaxNode, MeanNode, MinNode, ValueNode, WeightedMeanNode
from okapi.pareto import maximize
from okapi.random_state import use_rng
from okapi.tree import Tree

PALETTE = (MeanNode, MinNode, MaxNode, WeightedMeanNode, FarThresholdNode, CloseThresholdNode)


class Controls(MatchedPointControls, Okapi):
    pass


@pytest.fixture(autouse=True)
def numpy_backend(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)


def create_tree(kind, data):
    root = ValueNode(None, data[0], "a")
    inner = ValueNode([MinNode([ValueNode(None, data[2], "c")])], data[1], "b")
    op = kind([inner], .2) if kind is FarThresholdNode else kind([inner])
    root.add_child(op)
    root.add_child(MaxNode([ValueNode(None, data[2], "c")]))
    t = Tree(root, mutation_chance=.83)
    t.nodes["op_nodes"].reverse()
    return t


@pytest.mark.parametrize("kind", [FarThresholdNode, MeanNode])
def test_one_draw_has_identical_location_proposal_and_native_rng_across_all_policies(kind):
    data = np.random.default_rng(163).random((3, 19, 1))
    parent = create_tree(kind, data)
    source = parent.structural_key()
    events = {}
    rngs = {}
    outputs = {}
    for mode in ("retype", "identity", "refresh"):
        engine = object.__new__(Controls)
        engine.models, engine.ids = data, ["a", "b", "c"]
        engine.max_depth = engine.max_nodes = None
        engine.allowed_ops = PALETTE
        engine.point_policy, captured = mode, []
        engine.point_event_callback = captured.append
        rng = np.random.RandomState(3)
        with use_rng(rng):
            outputs[mode] = engine._apply_structural_mutation(replace_operator_mutation, parent)
        (event,) = captured
        events[mode], rngs[mode] = event, rng.get_state()
    assert parent.structural_key() == source
    assert outputs["retype"].structural_key() != source and outputs["identity"].structural_key() == source
    assert len({(e.path, e.original_family, e.proposed_family) for e in events.values()}) == 1
    for mode in ("identity", "refresh"):
        np.testing.assert_array_equal(rngs[mode][1], rngs["retype"][1])
        assert rngs[mode][2:] == rngs["retype"][2:]
    e = events["refresh"]
    assert e.path == (0,) and e.changed == (kind is FarThresholdNode)
    old = dict(operator_paths(parent))[e.path]
    new = dict(operator_paths(outputs["refresh"]))[e.path]
    assert type(old) is type(new) and old is not new
    assert [v.id for v in old.children] == [v.id for v in new.children]


def test_full_retype_and_no_fourth_move_replay_their_stock_native_counterparts(tmp_path):
    rng = np.random.default_rng(381)
    files = []
    for k in range(5):
        path = tmp_path / f"{k}.npy"
        np.save(path, rng.random((50, 1)))
        files.append(path)
    label_path = tmp_path / "gt.npy"
    np.save(label_path, rng.integers(2, size=(50, 1)))

    def run(kind, enable):
        engine = kind(
            files, label_path, population_size=5, population_multiplier=2, tournament_size=2,
            objective_functions=(lambda p, y: 1 - float(np.mean((p - y) ** 2)),), objectives=(maximize,),
            allowed_ops=PALETTE, backend="numpy", postprocessing_function=lambda x: x,
            seed=7121, operator_point_mutation=enable,
        )
        events = []
        if isinstance(engine, MatchedPointControls):
            engine.point_event_callback = events.append
        snapshots = []
        for _ in range(40):
            engine.run_iteration()
            state = engine.rng.get_state()
            snapshots.append((
                [t.structural_key() for t in engine.population], engine.fitnesses.tobytes(),
                state[0], state[1].tobytes(), state[2:],
            ))
        return snapshots, events

    for enabled in (False, True):
        native, _ = run(Okapi, enabled)
        controlled, observed = run(Controls, enabled)
        assert controlled == native
        assert bool(observed) == enabled
