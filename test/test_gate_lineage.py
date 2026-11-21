import numpy as np
import pytest

from okapi import Okapi
from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.crossover import crossover
from okapi.gate_lineage import gate_descendants, mark_gate_origin, tree_with_gate_cutoff
from okapi.globals import postprocessing_function
from okapi.interventions import tree_with_bypass
from okapi.mutation import mutate_parameters
from okapi.node import CloseThresholdNode, FarThresholdNode, MeanNode, ThresholdNode, ValueNode
from okapi.random_state import use_rng
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def numpy_backend(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)


def leaf(name):
    scores = {"parent": [.275, .875], "left": [.620, .310], 3: [.730, .080]}
    return ValueNode(None, np.asarray(scores.get(name, [.440, .550]))[:, None], name)


def gate_tree(gate_type=FarThresholdNode, *args):
    root = leaf("parent")
    root.add_child(gate_type([leaf("left"), leaf(3)], .389721643, *args))
    return Tree(root, mutation_chance=.73)


@pytest.mark.parametrize("gate_type,args,direction", [
    (ThresholdNode, (False,), "far"), (ThresholdNode, (True,), "close"),
    (FarThresholdNode, (), "far"), (CloseThresholdNode, (), "close"),
])
def test_mark_and_tree_copy_preserve_only_gate_origin(gate_type, args, direction):
    source = gate_tree(gate_type, *args)
    source.op_nodes[0].scratch = {"temporary": [1, 2]}
    before = source.evaluation
    marked = mark_gate_origin(source, (0,), token="seed")
    assert source.evaluation is before and gate_descendants(source, token="seed") == ()
    assert marked.mutation_chance == source.mutation_chance and marked.root is not source.root
    assert gate_descendants(marked, token="other") == ()
    for copy in (marked, marked.copy(), Tree(marked.root.copy_subtree())):
        hits = gate_descendants(copy, token="seed")
        assert isinstance(hits, tuple) and len(hits) == 1
        hit = hits[0]
        assert (hit.path, hit.direction, hit.threshold, hit.operand_ids) == ((0,), direction, .389721643, ("parent", "left", 3))
        assert not hasattr(copy.op_nodes[0], "scratch")
        assert copy.structural_key() == source.structural_key()
        np.testing.assert_array_equal(copy.predict(), before)


def test_retagging_changes_only_selected_occurrence_and_tracks_updated_paths():
    root = gate_tree().root
    nested = gate_tree(CloseThresholdNode).root
    root.add_child(MeanNode([nested]))
    path = (1, 0, 0)
    first = mark_gate_origin(Tree(root), (0,), token="one")
    both = mark_gate_origin(first, path, token="one")
    assert {h.path for h in gate_descendants(both, token="one")} == {(0,), path}
    retagged = mark_gate_origin(both, path, token="two")
    assert [h.path for h in gate_descendants(retagged, token="one")] == [(0,)]
    assert [h.path for h in gate_descendants(retagged, token="two")] == [path]
    assert len(gate_descendants(both, token="one")) == 2
    reduced = tree_with_bypass(retagged, [(0,)])
    assert gate_descendants(reduced, token="one") == ()
    assert [h.path for h in gate_descendants(reduced, token="two")] == [(0, 0, 0)]
    assert gate_descendants(tree_with_bypass(reduced, [(0,)]), token="two") == ()


def test_copying_marked_material_can_create_two_descendants_from_one_mark():
    ancestor = mark_gate_origin(gate_tree(), (0,), token="origin")
    branch = ancestor.root.copy_subtree()
    expanded = ancestor.copy()
    expanded.append_after(expanded.root, MeanNode([branch]))
    hits = gate_descendants(expanded, token="origin")
    assert [hit.path for hit in hits] == [(0,), (1, 0, 0)]
    assert all(hit.operand_ids == ("parent", "left", 3) for hit in hits)
    assert len(gate_descendants(ancestor, token="origin")) == 1


def test_changes_report_live_threshold_and_operand_ids_until_material_is_removed():
    original = mark_gate_origin(gate_tree(), (0,), token="one")
    with use_rng(np.random.RandomState(17)):
        mutated = mutate_parameters(original, mutation_strength=.07)
    gate = mutated.op_nodes[0]
    hit, = gate_descendants(mutated, token="one")
    assert hit.threshold == gate.threshold != original.op_nodes[0].threshold
    added = leaf("fresh")
    mutated.append_after(gate, added)
    assert gate_descendants(mutated, token="one")[0].operand_ids == ("parent", "left", 3, "fresh")
    mutated.prune_at(added)
    assert gate_descendants(mutated, token="one")[0].operand_ids == ("parent", "left", 3)
    old_key = mutated.structural_key()
    replacement = FarThresholdNode([leaf("left"), leaf(3)], gate.threshold)
    mutated.replace_at(gate, replacement)
    assert mutated.structural_key() == old_key and gate_descendants(mutated, token="one") == ()
    assert gate_descendants(original, token="one")[0].threshold == .389721643


def test_crossover_transfers_the_tag_only_to_offspring_receiving_the_gate():
    donor = mark_gate_origin(gate_tree(), (0,), token="one")
    recipient = gate_tree()
    assert donor.structural_key() == recipient.structural_key()
    assert gate_descendants(recipient, token="one") == ()
    # Each parent has exactly one operator. Their genotypes match; a genuine
    # operator crossover exchanges the marked and unmarked material.
    stream = np.random.RandomState(43)
    with use_rng(stream):
        lost, received = crossover(donor, recipient, node_type="op_nodes")
        descendants = [mutate_parameters(t) for t in (lost, received)]
    control_stream = np.random.RandomState(43)
    with use_rng(control_stream):
        controls = [mutate_parameters(t) for t in crossover(gate_tree(), recipient, node_type="op_nodes")]
    np.testing.assert_equal(stream.get_state(), control_stream.get_state())
    for descendant, control in zip(descendants, controls, strict=True):
        assert descendant.structural_key() == control.structural_key()
        np.testing.assert_array_equal(descendant.predict(), control.predict())
    assert lost.structural_key() == received.structural_key() == donor.structural_key()
    assert gate_descendants(lost, token="one") == ()
    assert gate_descendants(received, token="one") == gate_descendants(donor, token="one")
    assert gate_descendants(recipient, token="one") == ()


def test_marking_a_previously_edited_tree_preserves_native_sampling_order():
    source = gate_tree()
    source.append_after(source.op_nodes[0].children[0], MeanNode([leaf("deep")]))
    source.append_after(source.root, CloseThresholdNode([leaf("later")], .25))
    # The native append catalogues put the deep branch before the newer root
    # branch. A fresh breadth-first catalogue would reverse their order.
    breadth_first = Tree(source.root)
    assert [v.id for v in source.value_nodes] != [v.id for v in breadth_first.value_nodes]
    assert [type(op) for op in source.op_nodes] != [type(op) for op in breadth_first.op_nodes]
    marked = mark_gate_origin(source, (1,), token="latest")
    assert [v.id for v in marked.value_nodes] == [v.id for v in source.value_nodes]
    assert [type(op) for op in marked.op_nodes] == [type(op) for op in source.op_nodes]
    hit, = gate_descendants(marked, token="latest")
    assert (hit.path, hit.direction, hit.operand_ids) == ((1,), "close", ("parent", "later"))


def test_marking_one_seed_does_not_change_evolution_or_its_random_stream(monkeypatch):
    draw = np.random.default_rng(113)
    values = {i: draw.random((28, 1)) for i in range(5)}
    labels = draw.integers(2, size=(28, 1))
    monkeypatch.setattr(Okapi, "_build_train_tensors", lambda *_: (values, labels))

    def run(mark):
        engine = Okapi("", "", 5, 2, 2, seed=29, objective_functions=(lambda p, y: 1 - np.mean((p - y) ** 2),))
        gate = Tree(ValueNode([FarThresholdNode([ValueNode(None, values[1], 1)], .37)], values[0], 0), mutation_chance=.4)
        engine.population[0] = mark_gate_origin(gate, (0,), token="seed") if mark else gate
        score = engine._calculate_fitnesses
        evaluations = []

        def record(trees=None):
            batch = engine.population if trees is None else trees
            raw = score(batch)
            evaluations.append([(tree.structural_key(), tuple(row)) for tree, row in zip(batch, raw, strict=True)])
            return raw

        monkeypatch.setattr(engine, "_calculate_fitnesses", record)
        return engine, evaluations

    unmarked, reference = run(False)
    marked, observed = run(True)
    for _ in range(6):
        np.testing.assert_equal(unmarked.rng.get_state(), marked.rng.get_state())
        unmarked.run_iteration()
        marked.run_iteration()
        assert reference == observed
        np.testing.assert_array_equal(unmarked.fitnesses, marked.fitnesses)
        for a, b in zip(unmarked.population, marked.population, strict=True):
            assert a.structural_key() == b.structural_key()
            np.testing.assert_array_equal(a.predict(), b.predict())
    assert any(len(batch) > unmarked.population_size for batch in reference)
    assert any(row[0] != round(row[0], 3) for batch in reference for _, row in batch)
    np.testing.assert_equal(unmarked.rng.get_state(), marked.rng.get_state())


def test_invalid_tokens_paths_and_custom_threshold_subclasses_are_rejected():
    class DifferentGate(FarThresholdNode):
        pass

    source = gate_tree()
    with pytest.raises(TypeError):
        mark_gate_origin(source, (0,), token=8)
    with pytest.raises(TypeError):
        gate_descendants(source, token=None)
    for tree, path in ((source, ()), (source, (0, 0)), (source, (7,)), (gate_tree(DifferentGate), (0,)),
                       (Tree(ValueNode([MeanNode([leaf("b")])], np.array([.1]), "a")), (0,))):
        with pytest.raises(ValueError):
            mark_gate_origin(tree, path, token="source")
        with pytest.raises(ValueError):
            tree_with_gate_cutoff(tree, [path])


def test_joint_cutoff_reset_preserves_evolved_operands_and_order(monkeypatch):
    # Two nested three-input gates, then one unrelated ordered update.
    def build(values, name, ops=None):
        return ValueNode(ops, np.array(values, dtype=float), name)

    inputs = [[.1, .7, .3], [.3, .9, .4], [.8, .3, .6], [.7, .2, .9], [.4, .7, .1], [.7, .1, .9]]
    a, b, c, d, e, f = inputs
    inner = CloseThresholdNode([build(c, "c"), build(d, "d")], .2)
    outer = FarThresholdNode([build(b, "b", [inner]), build(e, "e")], .8)
    tree = Tree(build(a, "a", [outer, MeanNode([build(f, "f")])]), mutation_chance=.71)
    tree = mark_gate_origin(tree, (0,), token="persisted")
    tree = mark_gate_origin(tree, (0, 0, 0), token="persisted")
    before_key = tree.structural_key()
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x * .9)
    cached = tree.evaluation
    paths = [hit.path for hit in gate_descendants(tree, token="persisted")]
    reset = tree_with_gate_cutoff(tree, iter([*paths, paths[0]]))
    assert [hit.threshold for hit in gate_descendants(reset, token="persisted")] == [.5, .5]
    assert [hit.threshold for hit in gate_descendants(tree, token="persisted")] == [.8, .2]
    assert tree.evaluation is cached and tree.structural_key() == before_key
    assert reset.nodes_count == tree.nodes_count and reset.mutation_chance == tree.mutation_chance

    def select(scores, close):
        matrix = np.stack(scores)
        selector = np.argmin if close else np.argmax
        row = selector((matrix - .5) ** 2, axis=0)
        return .9 * matrix[row, np.arange(matrix.shape[1])]

    expected_inner = select([b, c, d], True)
    expected_outer = select([a, expected_inner, e], False)
    expected = .9 * (expected_outer + np.asarray(f)) / 2
    np.testing.assert_allclose(reset.predict(), expected)
    assert not np.allclose(cached, expected)
    no_hit = tree_with_gate_cutoff(tree, [])
    assert no_hit is not tree and no_hit.structural_key() == before_key
    np.testing.assert_array_equal(no_hit.predict(), cached)


@pytest.mark.parametrize("value", [-.1, 1.1, np.nan, np.inf, True, [.5], "0.5"])
def test_joint_cutoff_rejects_invalid_scalars(value):
    with pytest.raises(ValueError, match="cutoff"):
        tree_with_gate_cutoff(gate_tree(), [(0,)], threshold=value)


@pytest.mark.parametrize("path", [(), (8,), (0, 0), (-1,), (False,)])
def test_joint_cutoff_rejects_invalid_gate_paths(path):
    source = gate_tree()
    key = source.structural_key()
    with pytest.raises(ValueError, match="gate_path"):
        tree_with_gate_cutoff(source, [(0,), path], threshold=0)
    assert source.structural_key() == key
