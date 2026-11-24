from itertools import combinations

import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.functions import scale_vector_to_sum_1
from okapi.globals import _passthrough, use_postprocessing_function
from okapi.node import MinNode, ValueNode
from okapi.pair_basis import build_min_pair_basis
from okapi.pattern_features import select_pattern_features


@pytest.fixture(params=["numpy", "torch"])
def backend(request, monkeypatch):
    if request.param == "torch":
        pytest.importorskip("torch")
    monkeypatch.setattr(Backend, "_current_backend", Backend.get_backend())
    Backend.set_backend(request.param)
    return Backend.get_backend()


def no_score(*args):
    pytest.fail("The raw-provider objective runs only when the cap is exceeded")


@pytest.mark.parametrize("multiclass", [False, True])
def test_native_pair_output_replay_isolation_and_rebinding(backend, multiclass):
    if multiclass:
        a = np.array([[1.0, 0, 0], [0.5, 0.4, 0.1]])
        b = np.array([[0.0, 1, 0], [0.1, 0.2, 0.7]])
        expected = np.array([[1 / 3] * 3, [0.25, 0.5, 0.25]])
    else:
        a, b = np.array([[0.7], [0.1]]), np.array([[0.42], [0.6]])
        expected = np.minimum(a, b)
    left = ValueNode(None, backend.tensor(a), "left")
    right = ValueNode(None, backend.tensor(b), "right")
    left.evaluation = backend.tensor(np.zeros_like(a))
    with use_postprocessing_function(lambda p: p * 0):
        archive, info = build_min_pair_basis([left, right], [0, 1], objective=no_score, multiclass=multiclass)
    assert info == {"provider_indices": [0, 1], "pairs": [[0, 1]], "evaluations": 0, "programs": 1}
    (tree, p), = archive
    assert isinstance(tree.root.children[0], MinNode) and tree.nodes_count == 3
    assert [node.id for node in tree.value_nodes] == ["left", "right"]
    np.testing.assert_allclose(backend.to_numpy(p), expected)
    task = scale_vector_to_sum_1 if multiclass else _passthrough
    with use_postprocessing_function(task):
        np.testing.assert_array_equal(backend.to_numpy(tree.predict()), backend.to_numpy(p))
        for node in tree.value_nodes:
            node.value = backend.tensor(backend.to_numpy(node.value)[::-1].copy())
        np.testing.assert_allclose(backend.to_numpy(tree.predict()), expected[::-1])
    assert left.parent is None and right.parent is None and left.children == []
    np.testing.assert_array_equal(backend.to_numpy(left.evaluation), np.zeros_like(a))
    np.testing.assert_array_equal(backend.to_numpy(left.value), a)


@pytest.mark.parametrize("count", [1, 12, 14])
def test_floating_rank_and_original_order_with_all_pairs(backend, count):
    # Providers 1 and 11 share the highest score. In the oversize bank,
    # tied providers 12 and 13 fall below the cutoff by original order.
    roots = [ValueNode(None, backend.tensor(np.full((2, 1), 0.9 if i in (1, 11) else 0.2)), str(i)) for i in range(count)]
    y = np.array([1, 0])
    seen = []

    def discovery_score(prediction, labels):
        assert labels is y
        seen.append(float(prediction[0, 0]))
        return prediction[0, 0]

    archive, info = build_min_pair_basis(roots, y, objective=discovery_score)
    retained = list(range(min(12, count)))
    assert info["provider_indices"] == retained
    assert info["pairs"] == [list(pair) for pair in combinations(retained, 2)]
    assert info["evaluations"] == len(seen) == (count if count > 12 else 0)
    assert info["programs"] == len(archive) == len(retained) * (len(retained) - 1) // 2
    if count > 12:
        roots[-1].value = backend.tensor(np.full((2, 1), 1.0))
        _, override = build_min_pair_basis(roots, y, objective=discovery_score)
        assert override["provider_indices"] == list(range(10)) + [11, 13]


def test_pair_suffix_audits_only_unique_archive_features():
    arrays = [np.array([[0.2], [0.75], [0.4], [0.8]]),
              np.array([[0.7], [0.1], [0.3], [0.6]]),
              np.array([[0.4], [0.9], [0.2], [0.4]])]
    roots = [ValueNode(None, a, i) for i, a in enumerate(arrays)]
    y = np.zeros(4)
    pair_archive, _ = build_min_pair_basis(roots, y, objective=no_score)
    gp_tree, gp_pred = pair_archive[0]
    combined = [(gp_tree.copy(), gp_pred.copy()), *pair_archive]
    flat = np.full_like(y[:, None], 0.5)

    def score(p, y):
        return -np.mean((p[:, 0] - y) ** 2)

    selected, reported = select_pattern_features(combined, arrays, flat, y, objective=score, report_indices=True)
    plain, counts = select_pattern_features(combined, arrays, flat, y, objective=score)
    actual = reported.pop("evaluations_by_archive_index")
    assert selected == plain and reported == counts
    assert set(actual) == {0, 2, 3}  # the GP archive already supplied index 1's feature
    assert sum(actual.values()) == counts["evaluations"] - 1
    assert all(calls > 0 and calls % 4 == 0 for calls in actual.values())


def test_invalid_inputs():
    roots = [ValueNode(None, np.full((3, 1), 0.4), i) for i in range(13)]
    with pytest.raises(ValueError, match="finite real scalar"):
        build_min_pair_basis(roots, [0, 1, 1], objective=lambda p, y: np.array([1, 1]))
    with pytest.raises(ValueError, match="align"):
        build_min_pair_basis(roots, [0, 1], objective=no_score)
    roots[0].add_child(MinNode([roots[1]]))
    with pytest.raises(ValueError, match="raw provider"):
        build_min_pair_basis(roots, [0, 1, 1], objective=no_score)
