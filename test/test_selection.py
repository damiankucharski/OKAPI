import numpy as np
import pytest

from okapi.globals import set_backend
from okapi.node import MaxNode, ValueNode
from okapi.selection import select_compact, select_tree
from okapi.tree import Tree


def test_common_domain_difficulty_cancels_and_uncertain_gap_allows_smaller_tree():
    # Group losses share a strong shift. A constant .01 disadvantage has
    # zero paired SE; a similarly sized average gap can vary across groups.
    best = np.array([.20, .70, .10, .80])
    losses = [best, best + .01, best + [.12, -.10, .02, .02]]
    result = select_compact(losses, sizes=[9, 1, 3])
    assert result.best_index == 0 and result.index == 2
    assert result.eligible == (0, 2)
    assert select_compact(losses, [9, 1, 3], multiplier=0).index == 0


def test_ties_prefer_smaller_then_earlier_candidate():
    assert select_compact([[.2, .1]] * 3, [7, 3, 3]).index == 1


def test_validation_predicts_on_group_tensors_without_modifying_training_state():
    set_backend("numpy")
    training_scores = np.array([.1, .3])
    base = Tree(ValueNode(None, training_scores, "a"))
    fused = Tree(ValueNode([MaxNode([ValueNode(None, training_scores, "b")])], training_scores, "a"))
    original = fused.evaluation
    groups = [({"a": np.zeros(n), "b": np.ones(n)}, np.ones(n)) for n in (3, 7)]
    chosen = select_tree([base, fused], groups, lambda p, y: ((p - y) ** 2).mean())
    assert chosen.index == 1
    assert fused.root.evaluation is original
    assert fused.root.value is training_scores


@pytest.mark.parametrize("scores,sizes", [([[.2]], [1]), ([[0, np.nan]], [1]), ([[.2, .1]], []), ([[0, 0]], [0])])
def test_reject_invalid_evidence(scores, sizes):
    with pytest.raises(ValueError):
        select_compact(scores, sizes)
