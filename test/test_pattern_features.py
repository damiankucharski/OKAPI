import numpy as np

from okapi.node import MinNode, ValueNode
from okapi.pattern_features import select_pattern_features
from okapi.tree import Tree


def test_picker_excludes_singletons_and_functional_copies():
    a = np.array([[0.2], [0.8], [0.4], [0.7]])
    b = 1 - a
    left = ValueNode(None, a, "a")
    right = ValueNode(None, b, "b")
    singleton = Tree(right.copy_subtree())
    left.add_child(MinNode([right]))
    useful = Tree(left)
    p = np.minimum(a, b)
    flat = np.full_like(p, 0.5)
    archive = [(singleton, b), (useful.copy(), a), (useful, p), (useful.copy(), p.copy())]
    selected, info = select_pattern_features(archive, [a, b], flat, np.zeros(len(a)),
                                             objective=lambda pred, y: -np.mean((pred[:, 0] - y) ** 2))
    assert selected == [useful]
    assert info["unique_novel_features"] == 1
    assert info["selected"][0]["archive_index"] == 2
