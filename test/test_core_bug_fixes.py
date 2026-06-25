"""Regression tests for core OKAPI bugs found in the 2026-06-25 code review.

Each test is named after the bug it pins. They are written to FAIL on the
pre-fix code and PASS after the fix, so they double as documentation of the
defect. See the PR description / vault note for the full review.
"""

import numpy as np
import pytest
import torch
from loguru import logger
from matplotlib import pyplot as plt

from okapi.backend.pytorch import PyTorchBackend
from okapi.crossover import tournament_selection_indexes
from okapi.node import MeanNode, ThresholdNode, ValueNode
from okapi.pareto import maximize, minimize, plot_pareto_frontier
from okapi.population import choose_pareto
from okapi.tree import Tree


def _const(value: float) -> np.ndarray:
    return np.full((2, 2), float(value))


def _mean_tree(a: float, d: float, e: float) -> Tree:
    """root A -> MeanNode -> [D, E]; ids 'A','D','E'."""
    root = ValueNode(None, _const(a), "A")
    op = MeanNode(None)
    root.add_child(op)
    op.add_child(ValueNode(None, _const(d), "D"))
    op.add_child(ValueNode(None, _const(e), "E"))
    return Tree.create_tree_from_root(root)


# --- Bug 1: get_random_node never shuffled (tree.py) ----------------------------


def test_get_random_node_is_shuffled():
    """Pre-fix `order = np.arange(...)` always returned the first valid node,
    so variation operators only ever touched one position. After the fix it
    must sample across all value nodes."""
    root = ValueNode(None, _const(2), "A")
    op = MeanNode(None)
    root.add_child(op)
    for vid in ("D", "E", "G"):
        op.add_child(ValueNode(None, _const(3), vid))
    tree = Tree.create_tree_from_root(root)

    np.random.seed(0)
    seen = {tree.get_random_node("value_nodes").id for _ in range(200)}

    assert len(seen) > 1, "get_random_node always returns the same node — not shuffled"
    assert seen <= {"A", "D", "E", "G"}


# --- Bug 2: ThresholdNode assertion used `or` (always true) (node.py) ------------


@pytest.mark.parametrize("bad", [1.5, -0.5, 2.0, -0.001, 100.0])
def test_threshold_node_rejects_out_of_range(bad):
    with pytest.raises(AssertionError):
        ThresholdNode(None, bad)


@pytest.mark.parametrize("good", [0.0, 0.25, 0.5, 1.0])
def test_threshold_node_accepts_in_range(good):
    node = ThresholdNode(None, good)
    assert node.threshold == good


# --- Bug 3: do_pred_on_another_tensors overwrote current_tensors (tree.py) -------


def test_do_pred_on_another_tensors_uses_passed_tensors():
    """Pre-fix the unconditional `current_tensors = {}` discarded the argument,
    raising KeyError. The result must reflect the *passed* tensors."""
    tree = _mean_tree(1, 2, 3)

    new_tensors = {"A": _const(10), "D": _const(20), "E": _const(30)}
    result = tree.do_pred_on_another_tensors(current_tensors=new_tensors)

    reference = _mean_tree(10, 20, 30)
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference.evaluation))
    # It must actually use the new tensors, not the tree's original values.
    assert not np.allclose(np.asarray(result), np.asarray(tree.evaluation))


# --- Bug 4: choose_pareto could keep dominated trees when truncating ------------


def test_choose_pareto_excludes_dominated_when_truncating(monkeypatch):
    """With more Pareto trees than `n`, pre-fix sorted ALL trees by proximity and
    could keep a high-fitness *dominated* tree. Pareto set = {A,B,C}; D (0.8, 10
    nodes) is dominated by C (0.9, 5 nodes). Truncating to 2 must give {C,B}."""
    counts = {"A": 1, "B": 2, "C": 5, "D": 10}
    fitness = {"A": 0.3, "B": 0.5, "C": 0.9, "D": 0.8}
    ids = ["A", "B", "C", "D"]

    trees = [Tree(ValueNode(children=None, value=np.array([0.5]), id=i)) for i in ids]
    monkeypatch.setattr(Tree, "nodes_count", property(lambda t: counts[str(t.root.id)]))
    fitnesses = np.array([[fitness[i]] for i in ids])

    selected, _ = choose_pareto(trees, fitnesses, 2, [maximize])
    selected_ids = {t.root.id for t in selected}

    assert "D" not in selected_ids, "dominated tree D was selected over a Pareto tree"
    assert selected_ids == {"C", "B"}


# --- Bug 5: PyTorchBackend.to_numpy missing .cpu() (pytorch.py) -----------------


def test_to_numpy_handles_cpu_tensor():
    out = PyTorchBackend.to_numpy(torch.tensor([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_to_numpy_handles_gpu_tensor():
    """The actual fix target: pre-fix `.numpy()` raised on CUDA tensors."""
    out = PyTorchBackend.to_numpy(torch.tensor([1.0, 2.0]).cuda())
    np.testing.assert_array_equal(out, np.array([1.0, 2.0]))


# --- Bug 6: plot_pareto_frontier sort direction always ascending (pareto.py) ----


def test_plot_pareto_frontier_sorts_descending_for_maximize():
    pts = np.random.RandomState(0).rand(40, 2)
    fig, ax = plot_pareto_frontier(pts, [maximize, maximize])
    try:
        red = [ln for ln in ax.lines if ln.get_color() == "red"]
        assert red, "frontier line was not drawn"
        x = np.asarray(red[0].get_xdata())
        assert x.size >= 2
        assert np.all(np.diff(x) <= 1e-9), "maximize frontier should be sorted descending"
    finally:
        plt.close(fig)


def test_plot_pareto_frontier_sorts_ascending_for_minimize():
    pts = np.random.RandomState(1).rand(40, 2)
    fig, ax = plot_pareto_frontier(pts, [minimize, minimize])
    try:
        red = [ln for ln in ax.lines if ln.get_color() == "red"]
        assert red, "frontier line was not drawn"
        x = np.asarray(red[0].get_xdata())
        assert x.size >= 2
        assert np.all(np.diff(x) >= -1e-9), "minimize frontier should be sorted ascending"
    finally:
        plt.close(fig)


# --- Bug 7: tournament_selection_indexes warning text was backwards (crossover) -


def test_tournament_warning_says_large():
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        # pop = 3 < 2 * tournament_size (4) -> the warning fires
        tournament_selection_indexes(np.array([[0.1], [0.2], [0.3]]), tournament_size=2)
    finally:
        logger.remove(sink_id)

    joined = " ".join(messages)
    assert "large relative to the population" in joined
    assert "small related" not in joined
