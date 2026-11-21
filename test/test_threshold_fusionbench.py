"""End-to-end equality between a full native tree scan and the optional event solver."""

import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function
from okapi.node import ValueNode
from okapi.operators import CLOSE_THRESHOLD, FAR_THRESHOLD, MEAN, PROBABILISTIC_XOR
from okapi.threshold_refinement import refine_threshold_coordinate
from okapi.tree import Tree


@pytest.mark.parametrize("operator", [CLOSE_THRESHOLD, FAR_THRESHOLD])
def test_marker_gate_inside_complement_matches_complete_dyadic_native_scan(monkeypatch, operator):
    solver = pytest.importorskip("fusionbench.eval.gate_events").fit_threshold
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)
    rng = np.random.default_rng(713)
    inputs = {i: rng.integers(1, 16, size=(192, 1)) / 16 for i in range(4)}
    inputs[4] = np.ones((192, 1))
    # The second sequential operation gets the output of the first.
    # A final deterministic XOR with one complements the selected value.
    root = ValueNode(None, inputs[0], 0)
    root.add_child(MEAN([ValueNode(None, inputs[1], 1)]))
    gate = operator([ValueNode(None, inputs[j], j) for j in (2, 3)], 0.8125)
    root.add_child(gate)
    root.add_child(PROBABILISTIC_XOR([ValueNode(None, inputs[4], 4)]))
    tree = Tree(root)
    gate_index = tree.op_nodes.index(gate)
    source_key = tree.structural_key()

    def loss_for(threshold, labels):
        candidate = tree.copy()
        candidate.op_nodes[gate_index].threshold = threshold
        return np.mean((candidate.predict().ravel() - labels) ** 2)

    oracle = tree.copy()
    oracle.op_nodes[gate_index].threshold = 0.375
    y = (oracle.predict().ravel() >= 0.5).astype(int)
    # Upstream means lie on the 1/32 grid. Ties are on 1/64 and every open
    # region has a representative on 1/128, including the endpoint regions.
    grid = np.arange(129) / 128
    losses = [loss_for(t, y) for t in grid]
    result = refine_threshold_coordinate(tree, inputs, y, operator_index=gate_index, fit_threshold=solver)
    assert result.full_tree_evaluations == 5
    assert result.solver_result.method == "exact_event_sweep"
    assert result.solver_result.n_thresholds > 2
    assert result.loss < result.initial_loss
    assert result.loss == pytest.approx(min(losses), abs=1e-15)
    assert result.solver_result.fit_loss == pytest.approx(result.loss, abs=1e-15)
    assert tree.structural_key() == source_key
