from types import SimpleNamespace

import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function
from okapi.node import CloseThresholdNode, FarThresholdNode, MeanNode, ValueNode
from okapi.okapi import Okapi
from okapi.pareto import minimize
from okapi.threshold_graft import propose_threshold_graft
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def binary_numpy(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)


def fit_data():
    # A is the better singleton. B repairs A's two weak, incorrect decisions;
    # neither single provider can achieve the graft's Brier.
    return {"a": np.array([.95, .05, .43, .57]),
            "b": np.array([.42, .58, .90, .10]),
            "c": np.full(4, .5)}, np.array([1, 0, 1, 0])


def baseline(prediction, labels):
    return float(np.mean((prediction.ravel() - labels.ravel()) ** 2))


def grid_solver(operands, costs, *, close, original):
    grid = [original, 0., .25, .75, 1.]
    pick = np.argmin if close else np.argmax
    losses = [costs[np.arange(len(costs)), pick((operands-t)**2, axis=1)].mean() for t in grid]
    best = int(np.argmin(losses))
    return SimpleNamespace(threshold=grid[best], fit_loss=losses[best], n_thresholds=len(grid),
                           n_events=2, n_row_events=len(operands))


def test_native_singleton_becomes_real_graft_and_reenters_training(tmp_path):
    fit, y = fit_data()
    paths = []
    for key, values in fit.items():
        path = tmp_path / f"{key}.npy"
        np.save(path, values[:, None])
        paths.append(path)
    labels = tmp_path / "labels.npy"
    np.save(labels, y)
    engine = Okapi(paths, labels, 3, 2, 2, backend="numpy", seed=17,
                   objective_functions=(baseline,), objectives=(minimize,))
    index = int(np.argmin([baseline(tree.predict(), y) for tree in engine.population]))
    source = engine.population[index]
    assert source.root.id == "a.npy" and source.op_nodes == []
    graft = propose_threshold_graft(source, engine.train_tensors, y, seed=33, search_operands=True)
    assert isinstance(graft.tree, Tree)
    assert graft.tree.root.id == source.root.id
    assert graft.tree.nodes_count == source.nodes_count + 2 == 3
    assert isinstance(graft.tree.root.children[0], FarThresholdNode)
    assert graft.tree.root.children[0].children[0].id == graft.selected_operand == "b.npy"
    assert graft.selected_direction == "far" and graft.selected_threshold == .5
    expected = np.array([.95, .05, .90, .10])
    np.testing.assert_allclose(graft.tree.predict().ravel(), expected)
    assert graft.loss == pytest.approx(baseline(expected, y))
    assert graft.loss < min(baseline(values, y) for values in fit.values())
    assert source.nodes_count == 1 and graft.full_tree_evaluations == 5

    engine.population[index] = graft.tree
    engine.fitnesses = None
    engine.run_iteration()
    assert len(engine.population) == len(engine.fitnesses) == engine.population_size
    assert engine.fitnesses[:, 0].min() <= graft.loss + .0005
    for tree in engine.population:
        pred = tree.do_pred_on_another_tensors(current_tensors=engine.train_tensors)
        assert pred.shape == (4, 1) and np.isfinite(pred).all()


def test_factorial_prefix_and_cost_match_real_predictions(monkeypatch):
    p, y = fit_data()
    p.update(d=np.array([.31, .39, .66, .72]), root=np.full(4, .45))
    branches = [ValueNode(None, p[key], key) for key in ("a", "b")]
    tree = Tree(ValueNode([MeanNode(branches)], p["root"], "root"))
    real_predict = Tree.predict
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(self)
        return real_predict(self, *args, **kwargs)

    monkeypatch.setattr(Tree, "predict", counted)
    reference = None
    for all_operands in (False, True):
        for tuned in (False, True):
            calls.clear()
            solvers = []

            def solver(x, costs, *, close, original):
                solvers.append((x.copy(), close))
                return grid_solver(x, costs, close=close, original=original)

            result = propose_threshold_graft(tree, p, y, seed=48, search_operands=all_operands,
                                             fit_threshold=solver if tuned else None)
            site = tuple(result.site_path), result.anchor_id, tuple(result.operand_order)
            if reference is None:
                reference = site
            assert site == reference
            assert result.anchor_id in ("a", "b")
            assert set(result.operand_order) == set(p) - {result.anchor_id}
            choices = result.operand_order if all_operands else result.operand_order[:1]
            count = len(choices)
            assert len(calls) == result.full_tree_evaluations == 1 + (8 if tuned else 2)*count
            assert result.n_proposals == 2*count
            if tuned:
                assert result.threshold_candidates == 2*count*5
                assert result.solver_events == 2*count*2 and result.solver_row_events == 2*count*len(y)
                for values, _ in solvers:
                    np.testing.assert_array_equal(values[:, 0], p[result.anchor_id])
                assert [(values[:, 1].tobytes(), close) for values, close in solvers] == [
                    (p[provider].tobytes(), close) for provider in choices for close in (True, False)]
            else:
                assert len(solvers) == 0
    other = propose_threshold_graft(tree, p, 1-y, seed=48, search_operands=True)
    assert (tuple(other.site_path), other.anchor_id, tuple(other.operand_order)) == reference


def test_nested_anchor_preserves_structure_and_binds_private_values():
    p, y = fit_data()
    stale = np.array([.123, .321])
    source = Tree(ValueNode([MeanNode([ValueNode(None, stale, "a")])], stale, "c"), mutation_chance=.37)
    cached = source.evaluation
    result = propose_threshold_graft(source, p, y, seed=39, search_operands=True, max_nodes=5, max_depth=5)
    assert result.site_path == (0, 0) and result.anchor_id == "a"
    assert result.selected_operand == "b" and result.tree.root.id == "c"
    assert isinstance(result.tree.op_nodes[0], MeanNode)
    anchor = result.tree.root.children[0].children[0]
    assert anchor.id == "a" and isinstance(anchor.children[0], FarThresholdNode)
    assert anchor.children[0].children[0].id == "b"
    assert result.tree.depth == result.tree.nodes_count == 5
    assert result.tree.mutation_chance == source.mutation_chance
    expected = (p["c"] + np.array([.95, .05, .9, .1]))/2
    np.testing.assert_allclose(result.tree.predict(), expected)
    fresh = {k: 1-value for k, value in p.items()}
    np.testing.assert_allclose(result.tree.do_pred_on_another_tensors(current_tensors=fresh), 1-expected)
    assert source.nodes_count == 3 and source.root.children[0].children[0].children == []
    assert source.evaluation is cached and len(cached) == 2
    assert all(val.value is stale for val in source.value_nodes)
    assert not {id(n) for n in result.tree.root.get_nodes()} & {id(n) for n in source.root.get_nodes()}


@pytest.mark.parametrize("tuned", [False, True])
def test_tie_keeps_whole_original_tree(tuned):
    x = np.array([.3, .7, .4, .9])
    tree = Tree(ValueNode(None, np.full(2, .12), "a"))
    result = propose_threshold_graft(tree, {"a": x, "alias": x.copy()}, [0, 1, 0, 1], seed=11,
                                     search_operands=True, fit_threshold=grid_solver if tuned else None)
    assert result.tree.nodes_count == 1 and result.tree.root.id == "a"
    assert result.initial_loss == result.loss
    assert result.selected_operand is result.selected_direction is result.selected_threshold is None
    np.testing.assert_array_equal(result.tree.predict(), x)
    assert tree.root.value.shape == (2,)


@pytest.mark.parametrize(("provider_count", "caps"), [(1, {}), (2, {"max_nodes": 2}), (2, {"max_depth": 2})])
def test_no_viable_proposal_leaves_rebound_incumbent(provider_count, caps):
    p, y = fit_data()
    p = dict(list(p.items())[:provider_count])
    tree = Tree(ValueNode(None, np.array([.25]), "a"))
    result = propose_threshold_graft(tree, p, y, seed=2, search_operands=True, **caps)
    assert result.n_proposals == 0 and result.full_tree_evaluations == 1
    if caps:
        assert result.site_path is result.anchor_id is None
        assert result.operand_order == ()
    assert result.tree.nodes_count == 1 and result.loss == result.initial_loss
    assert result.selected_operand is result.selected_direction is result.selected_threshold is None
    np.testing.assert_array_equal(result.tree.predict(), p["a"])


def test_shape_validation_and_missing_fit_identity():
    tree = Tree(ValueNode(None, np.array([.25]), "a"))
    p, y = fit_data()
    for bad in (np.ones((4, 2)), np.ones(3), np.array([0., .1, np.nan, .2])):
        with pytest.raises(ValueError):
            propose_threshold_graft(tree, {**p, "a": bad}, y, seed=0, search_operands=True)
    with pytest.raises(ValueError):
        propose_threshold_graft(tree, {"b": p["b"]}, y, seed=0, search_operands=True)
    with pytest.raises(ValueError):
        propose_threshold_graft(tree, p, [0, 0, 2, 1], seed=0, search_operands=True)


def test_optional_public_solver_returns_native_verified_prediction():
    fit_threshold = pytest.importorskip("fusionbench.eval.gate_events").fit_threshold
    p, y = fit_data()
    source = Tree(ValueNode(None, p["a"], "a"))
    result = propose_threshold_graft(source, p, y, seed=1, search_operands=True, fit_threshold=fit_threshold)
    assert result.full_tree_evaluations == 17 and result.n_proposals == 4
    assert result.threshold_candidates > result.n_proposals
    assert result.loss < result.initial_loss
    assert result.loss == pytest.approx(baseline(result.tree.predict(), y))
    assert isinstance(result.tree.op_nodes[-1], (CloseThresholdNode, FarThresholdNode))


def test_first_improving_operand_wins_equal_loss_and_caps_filter_sites_before_draw():
    p, y = fit_data()
    source = Tree(ValueNode(None, p["a"], "a"))
    aliases = {"a": p["a"], "b": p["b"], "alias_b": p["b"].copy()}
    tied = propose_threshold_graft(source, aliases, y, seed=12, search_operands=True)
    assert tied.loss < tied.initial_loss and tied.selected_operand == tied.operand_order[0]
    assert tied.selected_direction == "far"

    # Only the left leaf is shallow enough for a graft under depth 5.
    shallow = ValueNode(None, p["a"], "a")
    deep = ValueNode([MeanNode([ValueNode(None, p["b"], "b")])], p["c"], "c")
    source = Tree(ValueNode([MeanNode([shallow, deep])], p["c"], "c"))
    for seed in (11, 12, 13):
        result = propose_threshold_graft(source, p, y, seed=seed, search_operands=False, max_depth=5)
        assert result.site_path == (0, 0) and result.anchor_id == "a"
        assert result.n_proposals == 2 and result.tree.depth <= 5


@pytest.mark.parametrize(("key", "value"), [
    ("search_operands", "False"), ("search_operands", 1),
    ("max_nodes", True), ("max_nodes", 3.5), ("max_nodes", 0),
    ("max_depth", np.bool_(True)), ("max_depth", 3.0), ("max_depth", -1),
])
def test_invalid_search_policy_or_budget_is_rejected(key, value):
    providers, labels = fit_data()
    tree = Tree(ValueNode(None, providers["a"], "a"))
    options = {"search_operands": True, key: value}
    with pytest.raises((TypeError, ValueError)):
        propose_threshold_graft(tree, providers, labels, seed=3, **options)


@pytest.mark.parametrize(("field", "value"), [
    ("n_thresholds", -1), ("n_thresholds", True),
    ("n_events", -.5), ("n_events", -2),
    ("n_row_events", np.bool_(True)), ("n_row_events", -1),
])
def test_invalid_solver_budget_is_rejected(field, value):
    providers, labels = fit_data()
    tree = Tree(ValueNode(None, providers["a"], "a"))

    def solver(*args, **kwargs):
        result = grid_solver(*args, **kwargs)
        setattr(result, field, value)
        return result

    with pytest.raises((TypeError, ValueError)):
        propose_threshold_graft(tree, providers, labels, seed=3, search_operands=False, fit_threshold=solver)


def test_numpy_boolean_policy_and_integer_budgets_are_accepted():
    providers, labels = fit_data()
    tree = Tree(ValueNode(None, providers["a"], "a"))

    def solver(*args, **kwargs):
        result = grid_solver(*args, **kwargs)
        result.n_thresholds, result.n_events, result.n_row_events = map(np.int32, (5, 0, 4))
        return result

    result = propose_threshold_graft(tree, providers, labels, seed=3, search_operands=np.bool_(False),
                                     fit_threshold=solver, max_nodes=np.int32(3), max_depth=np.int64(3))
    assert result.n_proposals == 2
    assert (result.threshold_candidates, result.solver_events, result.solver_row_events) == (10, 0, 8)
