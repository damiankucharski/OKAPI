import numpy as np
import pytest
import torch

from okapi.globals import set_backend
from okapi.interventions import binary_decision_changes, operator_paths, predict_with_bypass, predict_with_substitution
from okapi.node import MaxNode, MinNode, ValueNode
from okapi.tree import Tree


def val(name, scores, ops=None):
    return ValueNode(ops, np.array(scores, dtype=float), name)


def test_repeated_rescue_is_individually_redundant():
    set_backend("numpy")
    rescue = val("rescue", [1, 1, 1])
    repeat = MaxNode([rescue.copy()])
    tree = Tree(val("core", [0, 0, 1], [MaxNode([rescue]), repeat]))
    intact = tree.evaluation
    for path, _ in operator_paths(tree):
        np.testing.assert_array_equal(intact, predict_with_bypass(tree, [path]))
    changes = binary_decision_changes(intact, predict_with_bypass(tree, [(0,), (1,)]), [1, 0, 1])
    assert changes.repairs.tolist() == [True, False, False]
    assert changes.harms.tolist() == [False, True, False]
    assert tree.evaluation is intact
    assert tree.root.children[1] is repeat


def test_descendant_bypass_preserves_following_sibling_update():
    set_backend("numpy")
    nested = val("rescue", [1, 0], [MinNode([val("safe", [0, 1])])])
    tree = Tree(val("core", [0, 0], [MaxNode([nested]), MaxNode([val("extra", [0, 1])])]))
    np.testing.assert_array_equal(predict_with_bypass(tree, [(0, 0, 0)]), [1, 1])
    np.testing.assert_array_equal(predict_with_bypass(tree, [(0,), (0, 0, 0)]), [0, 1])
    assert tree.nodes_count == 7
    with pytest.raises(ValueError, match="Paths"):
        predict_with_bypass(tree, [(0, 0)])


def test_direction_tie_and_shapes():
    set_backend("numpy")
    change = binary_decision_changes([[0.5], [0.5]], [0.4, 0.4], [1, 0])
    assert change.repairs.tolist() == [True, False]
    assert change.harms.tolist() == [False, True]
    with pytest.raises(ValueError, match="shapes"):
        binary_decision_changes([[0.6, 0.4]], [[0.4, 0.6]], [1])


def test_replacement_changes_all_occurrences_including_root():
    set_backend("numpy")
    root = val("a", [1, 0], [MaxNode([val("a", [1, 0])])])
    tree = Tree(root)
    out = predict_with_substitution(tree, {"a": np.array([0.2, 0.1])})
    np.testing.assert_array_equal(out, [0.2, 0.1])
    np.testing.assert_array_equal(tree.predict(), [1, 0])
    with pytest.raises(ValueError, match="Unknown provider"):
        predict_with_substitution(tree, {"missing": np.array([0, 0])})


def test_tensor_scores_and_numpy_labels():
    scores = torch.tensor([.5, .2], requires_grad=True)
    changes = binary_decision_changes(scores, torch.tensor([.4, .8]), np.array([1, 0]))
    np.testing.assert_array_equal(changes.repairs, [True, True])
