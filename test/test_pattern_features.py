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
    selected, info = select_pattern_features(archive, [a, b], flat, np.zeros(len(a)), objective=lambda pred, y: -np.mean((pred[:, 0] - y) ** 2))
    assert selected == [useful]
    assert info["unique_novel_features"] == 1
    assert info["selected"][0]["archive_index"] == 2


def identity_program(p, name):
    # Repeating the operand makes a native non-singleton with the same output.
    root = ValueNode(None, p, name)
    root.add_child(MinNode([root.copy_subtree()]))
    return Tree(root)


def test_multiclass_negative_feature_and_exact_callback_costs():
    y = np.array([0, 1, 2, 0, 2, 1])
    wrong = np.full((len(y), 3), 0.475)
    wrong[np.arange(len(y)), y] = 0.05
    tree = identity_program(wrong, "inverted")
    flat = np.full_like(wrong, 1 / 3)
    calls = []

    def score(p, labels):
        np.testing.assert_allclose(p.sum(axis=1), 1)
        calls.append(len(p))
        return np.mean(p[np.arange(len(y)), labels])

    selected, info = select_pattern_features(((tree, wrong) for _ in range(1)), [flat], flat, y, objective=score, report_indices=True)
    assert selected == [tree]
    assert info["selected"][0]["weight"] == -1.0
    assert info["selected"][0]["score"] > info["initial_score"]
    assert info["evaluations_by_archive_index"] == {0: 4}
    assert info["evaluations"] == len(calls) == 5


def test_duplicate_filter_checks_all_rows_and_stops_on_a_plateau():
    n = 64
    flat = np.full((n, 1), 0.5)
    first = np.full((n, 1), 0.75)
    other = first.copy()
    probes = np.linspace(0, n - 1, num=16, dtype=int)
    different_row = next(i for i in range(n) if i not in probes)
    other[different_row] = 0.25
    within_tolerance = first - 1e-13
    outputs = [first, within_tolerance, other]
    archive = [(identity_program(a, str(i)), a) for i, a in enumerate(outputs)]
    chosen, info = select_pattern_features(archive, [flat], flat, np.zeros(n), objective=lambda p, y: 1.0, report_indices=True)
    assert chosen == []
    assert info["unique_novel_features"] == 2
    assert info["evaluations_by_archive_index"] == {0: 4, 2: 4}
    assert info["evaluations"] == 9


def test_greedy_cap_order_and_counts_over_multiple_steps():
    bits = (np.arange(16)[:, None] >> np.arange(4)) & 1
    flat = np.full((16, 1), 0.5)
    probabilities = 0.25 + 0.5 * bits
    archive = []
    for i in range(4):
        p = probabilities[:, i : i + 1]
        archive.append((identity_program(p, str(i)), p))
    logit = np.log(probabilities / (1 - probabilities)).sum(axis=1)
    labels = 1.0 / (1 + np.exp(-logit))

    def score(prediction, target):
        return round(-np.mean((prediction[:, 0] - target) ** 2), 12)

    selected, info = select_pattern_features(archive, [flat], flat, labels, objective=score, report_indices=True)
    # Equivalent features tie at each stage, resolved by their supplied order.
    assert selected == [tree for tree, _ in archive[:3]]
    assert [r["archive_index"] for r in info["selected"]] == [0, 1, 2]
    assert info["evaluations_by_archive_index"] == {0: 4, 1: 8, 2: 12, 3: 12}
    assert info["evaluations"] == 37
