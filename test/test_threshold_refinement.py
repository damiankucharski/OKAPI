from types import SimpleNamespace

import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function
from okapi.node import CloseThresholdNode, FarThresholdNode, MaxNode, MeanNode, MinNode, ThresholdNode, ValueNode
from okapi.threshold_refinement import refine_threshold_coordinate
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def numpy_backend(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)


def enumerate_grid(operands, downstream_losses, *, close, original):
    # A small fixed grid suffices for the wrapper tests. The event solver is an
    # independent dependency supplied by callers.
    thresholds = [original, 0., .25, .5, .75, 1.]
    pick = np.argmin if close else np.argmax
    losses = []
    for threshold in thresholds:
        chosen = pick((operands - threshold) ** 2, axis=1)
        losses.append(downstream_losses[np.arange(len(operands)), chosen].mean())
    best = int(np.argmin(losses))
    return SimpleNamespace(threshold=thresholds[best], fit_loss=losses[best], n_thresholds=len(thresholds))


def leaf(name, children=()):
    return ValueNode(children, np.array([.333]), name)


@pytest.mark.parametrize(("gate_class", "original"), [(CloseThresholdNode, .05), (FarThresholdNode, .95)])
def test_nested_gate_sees_real_operands_and_full_tree_losses(gate_class, original, monkeypatch):
    # The source deliberately holds a different split and populated caches.
    def process(x):
        return .1 + .8 * x

    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", process)
    inner_child = leaf("a", [MaxNode([leaf("extra")])])
    target = gate_class([inner_child, leaf("b")], original)
    nested = leaf("nested", [MeanNode([leaf("prefix")]), target, MinNode([leaf("cap")])])
    pre = MeanNode([leaf("pre")])
    tree = Tree(leaf("base", [pre, MeanNode([nested]), MaxNode([leaf("floor")])]), mutation_chance=.37)
    # replace_at moves the replacement to the back of the public op_nodes list.
    # The tree copy rebuilds that list from traversal order.
    tree.replace_at(pre, MeanNode(pre.children))
    position = tree.op_nodes.index(target)
    assert type(tree.copy().op_nodes[position]) is not gate_class
    source_state = tree.evaluation

    rng = np.random.default_rng(167)
    fit = {v.id: rng.random((90, 1)) for v in tree.value_nodes}
    y = rng.integers(2, size=90)
    prefix = process((fit["nested"] + fit["prefix"]) / 2)
    child = process(np.maximum(fit["a"], fit["extra"]))
    operands = np.column_stack([prefix, child, fit["b"]])
    ancestor_input = process((fit["base"] + fit["pre"]) / 2)
    forced_predictions = process(np.maximum(process((ancestor_input + process(np.minimum(process(operands), fit["cap"]))) / 2),
                                            fit["floor"]))
    expected_costs = (forced_predictions - y[:, None]) ** 2
    seen = []
    real_predict = Tree.predict

    def counted(self, *args, **kwargs):
        seen.append(self)
        return real_predict(self, *args, **kwargs)

    monkeypatch.setattr(Tree, "predict", counted)
    results = []

    def solver(values, costs, *, close, original):
        np.testing.assert_allclose(values, operands, rtol=0, atol=1e-15)
        np.testing.assert_allclose(costs, expected_costs, rtol=0, atol=1e-15)
        assert close == (gate_class is CloseThresholdNode)
        assert np.any(np.abs(costs - (values-y[:, None])**2) > .01)
        results.append(enumerate_grid(values, costs, close=close, original=original))
        return results[-1]

    result = refine_threshold_coordinate(tree, fit, y[:, None], operator_index=position, fit_threshold=solver)
    assert type(result.tree.op_nodes[position]) is gate_class
    assert result.solver_result is results[0] and result.solver_result.n_thresholds == 6
    assert len(seen) == result.full_tree_evaluations == 5
    assert result.tree.mutation_chance == tree.mutation_chance == .37
    assert result.loss <= result.initial_loss
    np.testing.assert_allclose(result.realized_loss, result.solver_result.fit_loss, atol=1e-15)
    np.testing.assert_allclose(result.loss, ((result.tree.predict().ravel()-y)**2).mean(), atol=1e-15)
    assert tree.evaluation is source_state and target.threshold == original
    assert all(v.value.shape == (1,) for v in tree.value_nodes)
    assert all(v.evaluation is None for v in result.tree.value_nodes)
    assert all(not type(node).__name__.startswith("_Selected") for node in result.tree.op_nodes)


@pytest.mark.parametrize(("target", "labels", "suggestion", "accepted", "realized"), [
    (ThresholdNode([leaf("b")], 0., True), [1, 1], 1., 1., 0.),
    (FarThresholdNode([leaf("b")], 1.), [1, 1], 0., 0., 0.),
    (CloseThresholdNode([leaf("b")], 0.), [0, 0], .2, 0., 0.),
    (CloseThresholdNode([leaf("b")], 0.), [0, 0], 1., 0., 1.),
])
def test_strict_improvement_and_original_slot_order(target, labels, suggestion, accepted, realized):
    tree = Tree(leaf("a", [target]))
    fit = {"a": np.zeros(2), "b": np.ones(2)}

    def solver(x, costs, **kwargs):
        np.testing.assert_array_equal(x, [[0, 1], [0, 1]])
        return SimpleNamespace(threshold=suggestion)

    result = refine_threshold_coordinate(tree, fit, labels, operator_index=0, fit_threshold=solver)
    assert result.tree.op_nodes[0].threshold == accepted
    assert result.realized_loss == realized and result.loss == min(realized, result.initial_loss)
    assert result.full_tree_evaluations == 4
    assert result.tree.op_nodes[0] is not target


def test_float32_precision_mismatch_verified_by_ordinary_final_score():
    a = np.float32(.5)
    b = np.nextafter(a, np.float32(1))
    midpoint = (float(a) + float(b)) / 2
    tree = Tree(leaf("b", [CloseThresholdNode([leaf("a")], float(a))]))

    def solver(x, costs, **kwargs):
        assert x.dtype == np.float32
        # Binary64 sees a tie and chooses slot 0. The float32 gate rounds
        # its scalar to a, and therefore chooses slot 1.
        return SimpleNamespace(threshold=midpoint, fit_loss=costs[0, 0])

    result = refine_threshold_coordinate(tree, {"a": np.array([a]), "b": np.array([b])}, [1],
                                         operator_index=0, fit_threshold=solver)
    assert result.solver_result.fit_loss < result.initial_loss
    assert result.loss == result.realized_loss == result.initial_loss
    assert result.tree.op_nodes[0].threshold == a


def test_single_operand_torch_backend_and_labels(monkeypatch):
    torch = pytest.importorskip("torch")
    from okapi.backend.pytorch import PyTorchBackend

    monkeypatch.setattr(Backend, "_current_backend", PyTorchBackend)
    tree = Tree(leaf("a", [FarThresholdNode([], .4)]))
    x = torch.tensor([[.2], [.8]], requires_grad=True)
    result = refine_threshold_coordinate(tree, {"a": x}, torch.tensor([0, 1]), operator_index=0, fit_threshold=enumerate_grid)
    assert result.full_tree_evaluations == 3 and result.tree.op_nodes[0].threshold == .4
    np.testing.assert_allclose(result.loss, .04, atol=1e-7)


def test_invalid_targets_shapes_and_callback_results():
    class CustomThreshold(CloseThresholdNode):
        pass

    fit = {"a": np.array([.2, .3])}
    y = [0, 1]

    def run(tree, providers=fit, labels=y, operator_index=0, solver=enumerate_grid):
        return refine_threshold_coordinate(tree, providers, labels, operator_index=operator_index, fit_threshold=solver)

    for op in (MeanNode([]), CustomThreshold([], .5)):
        with pytest.raises(ValueError, match="built-in"):
            run(Tree(leaf("a", [op])))
    tree = Tree(leaf("a", [CloseThresholdNode([], .5)]))
    for idx in (-1, 2):
        with pytest.raises(IndexError, match="outside"):
            run(tree, operator_index=idx)
    with pytest.raises(TypeError, match="integer"):
        run(tree, operator_index=True)
    with pytest.raises(ValueError, match="Missing"):
        run(tree, {})
    for labels in ([0, 2], [], [[0, 1]]):
        with pytest.raises(ValueError, match="binary label"):
            run(tree, labels=labels)
    for values in (np.ones((2, 2)), np.ones(1)):
        with pytest.raises(ValueError, match="matching"):
            run(tree, {"a": values})
    for values in (np.array([np.nan, 1]), np.array([-.2, .2])):
        with pytest.raises(ValueError, match="finite probabilities"):
            run(tree, {"a": values})
    broken = Tree(leaf("a", [ThresholdNode([], .5)]))
    broken.op_nodes[0].threshold = np.nan
    with pytest.raises(ValueError, match="Original threshold"):
        run(broken)
    broken.op_nodes[0].threshold = .5
    broken.op_nodes[0].close = "close"
    with pytest.raises(ValueError, match="boolean"):
        run(broken)
    for proposed in (-.01, np.nan, np.inf, [0, 1], None):
        with pytest.raises(ValueError, match="threshold"):
            run(tree, solver=lambda *a, t=proposed, **kw: SimpleNamespace(threshold=t))


def test_custom_unhashable_equal_operator_does_not_displace_selected_gate():
    class CustomMean(MeanNode):
        __hash__ = None

        def __eq__(self, other):
            return isinstance(other, (MeanNode, CloseThresholdNode))

        def copy(self):
            return CustomMean(None)

    tree = Tree(leaf("a", [CustomMean([]), CloseThresholdNode([leaf("b")], 0.)]))
    fit = {"a": np.zeros(3), "b": np.ones(3)}
    result = refine_threshold_coordinate(tree, fit, [1, 1, 1], operator_index=1, fit_threshold=enumerate_grid)
    assert isinstance(result.tree.op_nodes[0], CustomMean)
    assert result.loss == 0 and result.tree.op_nodes[1].threshold > .5
