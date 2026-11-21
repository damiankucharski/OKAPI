import numpy as np
import pytest

from okapi import Okapi
from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function
from okapi.islands import migrate_ring
from okapi.node import (
    CloseThresholdNode,
    FarThresholdNode,
    LogitMeanNode,
    MaxNode,
    MeanNode,
    MinNode,
    ThresholdNode,
    TrimmedMeanNode,
    ValueNode,
    WeightedLogitMeanNode,
    WeightedMeanNode,
)
from okapi.pareto import minimize
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def numpy_backend(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda scores: scores)


def providers():
    return {name: np.full((2, 1), float(name == "C")) for name in "ABCD"}


def split_children(x, count):
    root, *leaves = [ValueNode(None, x[name], name) for name in "ABCD"]
    root.add_child(MinNode(leaves[:count]))
    root.add_child(MaxNode(leaves[count:]))
    return Tree(root, mutation_chance=0)


def wrap(operator, *params):
    x = np.zeros((2, 1))
    return Tree(ValueNode([operator([ValueNode(None, x, "B")], *params)], x, "A"))


def test_topology_distinguishes_former_alias_and_ignores_tensor_bindings():
    x = providers()
    first, second = split_children(x, 2), split_children(x, 1)
    assert repr(first) == repr(second)
    np.testing.assert_array_equal(first.predict(), np.zeros((2, 1)))
    np.testing.assert_array_equal(second.predict(), np.ones((2, 1)))
    key = first.structural_key()
    assert isinstance(key, tuple) and len({key, second.structural_key()}) == 2
    rebound = {name: np.full((5, 1), .6) for name in x}
    _, copy = first.do_pred_on_another_tensors(current_tensors=rebound, return_tree=True)
    assert first.copy().structural_key() == copy.structural_key() == key
    assert copy.root.value.shape == (5, 1)


@pytest.mark.parametrize("operator,args,other", [
    (ThresholdNode, (.414, True), (.415, True)),
    (ThresholdNode, (.414, False), (.415, False)),
    (CloseThresholdNode, (.414,), (.415,)),
    (FarThresholdNode, (.414,), (.415,)),
    (WeightedMeanNode, ([.414, .586],), ([.415, .585],)),
    (WeightedLogitMeanNode, ([1.011, 2.011],), ([1.011, 2.012],)),
    (LogitMeanNode, (1., .123), (1., .124)),
    (LogitMeanNode, (1., .123), (np.nextafter(1., 2.), .123)),
])
def test_exact_builtin_parameters_survive_display_rounding(operator, args, other):
    x, y = wrap(operator, *args), wrap(operator, *other)
    assert repr(x) == repr(y)
    assert x.structural_key() != y.structural_key()
    assert x.structural_key() == x.copy().structural_key()


def test_builtin_types_child_order_and_provider_id_types():
    keys = {wrap(op).structural_key() for op in (MinNode, MaxNode, MeanNode, TrimmedMeanNode)}
    assert len(keys) == 4
    assert wrap(ThresholdNode, .4, True).structural_key() != wrap(ThresholdNode, .4, False).structural_key()
    assert wrap(CloseThresholdNode, .4).structural_key() != wrap(ThresholdNode, .4, True).structural_key()
    assert Tree(ValueNode(None, None, 1)).structural_key() != Tree(ValueNode(None, None, "1")).structural_key()
    first = split_children(providers(), 2)
    permuted = first.copy()
    permuted.root.children[0].children.reverse()
    assert first.structural_key() != permuted.structural_key()
    permuted = first.copy()
    permuted.root.children.reverse()
    assert first.structural_key() != permuted.structural_key()


def test_keys_snapshot_mixed_numeric_parameters_without_narrowing():
    base = np.float32(.25)
    weights = [np.float32(.5), base, .25]
    x = np.zeros((2, 1))
    root = ValueNode(None, x, "A")
    root.add_child(WeightedMeanNode([ValueNode(None, x, "B"), ValueNode(None, x, "C")], weights))
    tree = Tree(root)
    original = tree.structural_key()
    assert original == tree.copy().structural_key()
    weights[1] = np.nextafter(base, np.float32(1))
    weights[2] = .5 - float(weights[1])
    assert tree.structural_key() != original
    assert wrap(LogitMeanNode, np.float32(1), 0.).structural_key() != wrap(LogitMeanNode, 1., 0.).structural_key()


def test_opaque_custom_nodes_do_not_alias_each_other_or_builtins():
    class CustomMean(MeanNode):
        __hash__ = None

        def __init__(self, children, offset):
            super().__init__(children)
            self.offset = offset

        def calculate(self):
            return super().calculate() + self.offset

    a, b, c = (wrap(CustomMean, offset) for offset in (0., .25, .25))
    assert repr(a) == repr(b) == repr(c) == repr(wrap(MeanNode))
    assert len({t.structural_key() for t in (a, b, c, wrap(MeanNode))}) == 4
    before = a.structural_key()
    a.op_nodes[0].offset = .375
    assert a.structural_key() != before

    class CustomValue(ValueNode):
        def calculate(self):
            return 1 - super().calculate()

    roots = [cls(None, np.zeros((2, 1)), "A") for cls in (ValueNode, CustomValue, CustomValue)]
    assert len({Tree(root).structural_key() for root in roots}) == 3


@pytest.fixture
def engine(monkeypatch):
    def make(x, y, size, reuse=False):
        monkeypatch.setattr(Okapi, "_build_train_tensors", lambda *_: (x, y))
        calls = []

        def brier(p, labels):
            calls.append(p.copy())
            return np.mean((p - labels) ** 2)

        runner = Okapi("", "", size, 1, 1, objective_functions=(brier,), objectives=(minimize,),
                       minimize_node_count=False, reuse_survivor_fitness=reuse)
        return runner, calls

    return make


@pytest.mark.parametrize("reuse", [False, True])
def test_generation_scores_distinct_alias_once_and_suppresses_true_copy(engine, reuse):
    x = providers()
    y = np.array([[0.], [1.]])
    x["E"] = 1 - y
    gp, calls = engine(x, y, 2, reuse)
    old, novel = split_children(x, 2), split_children(x, 1)
    gp.population = [old, Tree(ValueNode(None, x["E"], "E"))]
    gp.fitnesses = gp._calculate_fitnesses().round(3)
    calls.clear()
    duplicate = old.copy()
    duplicate.mutation_chance = 0
    # Filling the offspring quota and turning off offspring mutation lets
    # run_iteration exercise its ordinary deduplication, scoring and selection.
    gp.additional_population = [duplicate, novel]
    gp.run_iteration()
    assert set(gp.population) == {old, novel}
    assert len(calls) == (1 if reuse else 3)
    assert sum(np.array_equal(p, novel.predict()) for p in calls) == 1
    np.testing.assert_array_equal(gp.fitnesses, [[.5], [.5]])


def test_ring_evaluates_alias_and_skips_a_true_copy(engine):
    x = providers()
    target = np.ones((2, 1))
    left, _ = engine(x, target, 1)
    right, calls = engine(x, target, 1)
    left.population = [split_children(x, 1)]
    right.population = [split_children(x, 2)]
    for island in (left, right):
        island.fitnesses = island._calculate_fitnesses().round(3)
    calls.clear()
    event = migrate_ring([left, right])[1]
    assert event.evaluated and event.admitted and len(calls) == 1
    assert right.population[0] is not left.population[0]
    assert right.population[0].structural_key() == left.population[0].structural_key()
    duplicate = migrate_ring([left, right])[1]
    assert duplicate.reason == "duplicate" and not duplicate.evaluated
    assert len(calls) == 1
