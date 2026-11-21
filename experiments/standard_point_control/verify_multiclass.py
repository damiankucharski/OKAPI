"""Witness checks for per-node simplex normalization and the native point-mutation switch."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fusionbench.eval.metrics import brier_score, normalize_proba
from fusionbench.methods.okapi import OKAPIMethod, _brier_objective
from fusionbench.scenarios.registry import SCENARIOS

from okapi import Okapi, mutation
from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.functions import scale_vector_to_sum_1
from okapi.globals import postprocessing_function
from okapi.mutation import tree_with_operator_type
from okapi.node import CloseThresholdNode, FarThresholdNode, MaxNode, MeanNode, MinNode, ValueNode, WeightedMeanNode
from okapi.pareto import maximize
from okapi.random_state import use_rng
from okapi.tree import Tree

PALETTE = (MeanNode, MinNode, MaxNode, WeightedMeanNode, FarThresholdNode, CloseThresholdNode)


def simplex(x):
    total = x.sum(axis=1, keepdims=True)
    uniform = np.full_like(x, 1 / x.shape[1])
    return np.divide(x, total, out=uniform, where=total != 0)


def prediction(node, tensors, multiclass):
    """Independent evaluation by node and ordered slot on class and example axes."""
    incoming = tensors[node.id]
    for op in node.children:
        values = np.stack([incoming, *(prediction(v, tensors, multiclass) for v in op.children)])
        if isinstance(op, WeightedMeanNode):
            scores = np.sum(values * np.asarray(op._weights)[:, None, None], axis=0)
        elif isinstance(op, (CloseThresholdNode, FarThresholdNode)):
            choose = np.argmin if op.close else np.argmax
            indices = choose((values - op.threshold) ** 2, axis=0)
            scores = np.take_along_axis(values, indices[None, :, :], axis=0)[0]
        else:
            reduce = {MeanNode: np.mean, MinNode: np.min, MaxNode: np.max}[type(op)]
            scores = reduce(values, axis=0)
        incoming = simplex(scores) if multiclass else scores
    return incoming


@pytest.fixture(autouse=True)
def scoped_backend(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", scale_vector_to_sum_1)


def fixture(multiclass, seed=59):
    rng = np.random.default_rng(seed)
    classes = 5 if multiclass else 2
    x = rng.dirichlet(np.ones(classes), size=(48, 4))
    x[0] = np.eye(classes)[np.arange(4) % classes]
    y = rng.integers(classes, size=len(x))
    tensors = {f"m{i}.npy": x[:, i] if multiclass else x[:, i, 1:] for i in range(x.shape[1])}
    return tensors, y, classes


def tree_for(tensors):
    leaves = [ValueNode(None, value, name) for name, value in tensors.items()]
    root, second, third, fourth = leaves
    root.add_child(MinNode([second]))
    third.add_child(MinNode([fourth]))
    root.add_child(MaxNode([third]))
    return Tree(root)


@pytest.mark.parametrize("family", PALETTE)
@pytest.mark.parametrize("path", [(0,), (1, 0, 0)])
def test_native_retyping_respects_every_class_and_postprocesses_nested_updates(family, path):
    tensors, y, nc = fixture(True)
    with use_rng(np.random.RandomState(83)):
        tree = tree_with_operator_type(tree_for(tensors), path, family)
    p = tree.predict()
    assert p.shape == (48, nc) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=2e-15, rtol=0)
    np.testing.assert_allclose(p, prediction(tree.root, tensors, True), atol=2e-15, rtol=0)
    full_brier = np.mean(np.sum((p - np.eye(nc)[y]) ** 2, axis=1))
    assert brier_score(y, p, nc) == pytest.approx(full_brier)
    assert _brier_objective(p, y) == pytest.approx(1 - full_brier / 2)


def test_final_simplex_alone_differs_from_native_per_operator_normalization(monkeypatch):
    tensors, _, nc = fixture(True)
    tree = tree_for(tensors)
    p = tree.predict()
    np.testing.assert_array_equal(p[0], np.full(nc, 1 / nc))
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)
    late = normalize_proba(tree.predict())
    assert np.max(np.abs(late - p)) > .05


def test_registry_class_axes():
    assert len(SCENARIOS) == 19
    multi = {}
    for name, kind in SCENARIOS.items():
        scenario = kind()
        x, y, _ = scenario._generate(32, np.random.default_rng(97))
        assert x.shape == (len(y), len(scenario.model_ids()), scenario.n_classes)
        np.testing.assert_allclose(x.sum(axis=2), 1, rtol=1e-6, atol=1e-6)
        if scenario.task == "multiclass":
            multi[name] = scenario.n_classes
        else:
            assert scenario.task == "binary" and scenario.n_classes == 2
    assert multi == {"saturated": 4, "multiclass_committee": 5}


@pytest.mark.parametrize("name", ["committee", "saturated", "multiclass_committee"])
@pytest.mark.parametrize("enabled", [False, True])
def test_actual_native_mutation_switch_and_archived_predictions(tmp_path, name, enabled):
    scenario = SCENARIOS[name]()
    nc = scenario.n_classes
    multiclass = scenario.task == "multiclass"

    def draw(seed):
        x, y, _ = scenario._generate(48, np.random.default_rng(seed))
        tensors = {f"m{i}.npy": x[:, i] if multiclass else x[:, i, 1:] for i in range(x.shape[1])}
        return tensors, y

    tensors, y = draw(59)
    query, _ = draw(61)
    paths = []
    for name, values in tensors.items():
        path = Path(tmp_path) / name
        np.save(path, values)
        paths.append(path)
    labels = Path(tmp_path) / "labels.npy"
    np.save(labels, y)
    post = scale_vector_to_sum_1 if multiclass else (lambda x: x)
    n_point_edits = 0
    native_edit = mutation.replace_operator_mutation

    def counted_edit(*args, **kwargs):
        nonlocal n_point_edits
        n_point_edits += 1
        return native_edit(*args, **kwargs)

    with patch.object(mutation, "replace_operator_mutation", new=counted_edit):
        gp = Okapi(
            paths, labels, population_size=4, population_multiplier=2, tournament_size=2,
            objective_functions=(_brier_objective,), objectives=(maximize,), allowed_ops=PALETTE,
            minimize_node_count=False, backend="numpy", postprocessing_function=post,
            operator_point_mutation=enabled, seed=7121,
        )
        gp.train(100)
        assert (n_point_edits > 0) == enabled
    for tree in gp.population:
        p = tree.do_pred_on_another_tensors(current_tensors=query)
        np.testing.assert_allclose(p, prediction(tree.root, query, multiclass), atol=2e-15, rtol=0)
        if multiclass:
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-15, rtol=0)
        else:
            full = np.column_stack((1 - p[:, 0], p[:, 0]))
            assert _brier_objective(p, y) == pytest.approx(1 - brier_score(y, full, nc) / 2)


@pytest.mark.parametrize("name", ["committee", "saturated", "multiclass_committee"])
def test_iteration_replay_matches_standard_adapter_train_including_every_fitness_batch(name):
    scenario = SCENARIOS[name]()
    x, y, _ = scenario._generate(48, np.random.default_rng(487))
    seed = 1207
    adapter = OKAPIMethod(seed=seed, n_restarts=1, generations=3, fitness="brier")
    native_score = Okapi._calculate_fitnesses
    trace = []

    def traced_score(engine, trees=None):
        trees = engine.population if trees is None else trees
        result = native_score(engine, trees)
        trace.append((tuple(t.structural_key() for t in trees), result.copy()))
        return result

    try:
        with patch.object(Okapi, "_calculate_fitnesses", new=traced_score):
            adapter.fit(x, y, task=scenario.task, n_classes=scenario.n_classes)
            stock_trace = list(trace)
            trace.clear()
            base = Path(adapter._tmp.name)
            pop = max(2, min(adapter.population_size, x.shape[1]))
            gp = Okapi(
                base / "fit", base / "gt.npy", population_size=pop,
                population_multiplier=adapter.population_multiplier,
                tournament_size=max(2, min(adapter.tournament_size, pop)),
                minimize_node_count=adapter.minimize_node_count,
                objective_functions=(_brier_objective,), objectives=(maximize,), allowed_ops=PALETTE,
                backend="numpy", seed=seed, operator_point_mutation=False,
                postprocessing_function=scale_vector_to_sum_1 if scenario.task == "multiclass" else (lambda a: a),
            )
            adapter._seed_population(gp, np.random.default_rng(seed))
            for _ in range(adapter.generations):
                gp.run_iteration()
        assert len(trace) == len(stock_trace) and len(trace) >= adapter.generations
        for (keys1, scores1), (keys2, scores2) in zip(trace, stock_trace, strict=True):
            assert keys1 == keys2
            np.testing.assert_array_equal(scores1, scores2)
        assert [t.structural_key() for t in gp.population] == [t.structural_key() for t in adapter.okapi.population]
        np.testing.assert_array_equal(gp.fitnesses, adapter.okapi.fitnesses)
        assert gp.rng.get_state()[2:] == adapter.okapi.rng.get_state()[2:]
        np.testing.assert_array_equal(gp.rng.get_state()[1], adapter.okapi.rng.get_state()[1])
    finally:
        adapter._tmp.cleanup()
