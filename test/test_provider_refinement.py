import numpy as np
import pytest

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function, set_backend
from okapi.node import MeanNode, ValueNode
from okapi.refinement import refine_provider_assignments
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def configure_evaluation(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)


def repeated_tree(placeholder):
    child = ValueNode(None, placeholder, "a")
    return Tree(ValueNode([MeanNode([child])], placeholder, "a"), mutation_chance=0.35)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_reassigns_one_occurrence_and_rebinds_fit_data(backend):
    set_backend(backend)
    if backend == "torch":
        import torch

        array = torch.as_tensor
    else:
        array = np.asarray
    placeholder = array([0.4, 0.4])
    tree = repeated_tree(placeholder)
    cached = tree.evaluation
    providers = {"a": array([0., 0.]), "b": array([1., 1.])}
    result = refine_provider_assignments(tree, providers, array([0.5, 0.5]), lambda p, y: ((p - y) ** 2).mean(), max_evaluations=30)
    assert (result.evaluations, result.initial_loss, result.loss) == (3, 0.25, 0.)
    assert [node.id for node in result.tree.value_nodes] == ["b", "a"]
    assert result.tree.nodes_count == tree.nodes_count and result.tree.mutation_chance == 0.35
    np.testing.assert_array_equal(result.tree.predict(), [0.5, 0.5])
    new = {"a": array([0.2]), "b": array([0.8])}
    np.testing.assert_allclose(result.tree.do_pred_on_another_tensors(current_tensors=new), [0.5])
    assert [node.id for node in tree.value_nodes] == ["a", "a"]
    assert tree.evaluation is cached and tree.root.value is placeholder


def test_budget_reproducibility_and_ties():
    tree = repeated_tree(np.array([0.9]))
    providers = {key: np.array([i / 5]) for i, key in enumerate("abcde")}

    def sampled():
        predictions = []

        def constant_loss(p, _):
            predictions.append(float(p.item()))
            return 1.

        result = refine_provider_assignments(tree, providers, None, constant_loss, max_evaluations=5, seed=17)
        assert len(predictions) == result.evaluations == 5
        assert repr(result.tree) == repr(tree)
        return predictions

    assert sampled() == sampled()
    result = refine_provider_assignments(tree, {"a": providers["a"]}, None, lambda p, _: p.item(), max_evaluations=100)
    assert result.evaluations == 1


def test_requires_finite_objective_valid_budget_and_all_ids():
    tree = repeated_tree(np.array([0.2]))
    providers = {"a": np.array([0.1])}

    def score(p, _):
        return p.item()

    with pytest.raises(ValueError, match="at least"):
        refine_provider_assignments(tree, providers, None, score, max_evaluations=0)
    with pytest.raises(ValueError, match="Missing"):
        refine_provider_assignments(tree, {}, None, score, max_evaluations=1)
    with pytest.raises(ValueError, match="finite"):
        refine_provider_assignments(tree, providers, None, lambda *_: np.nan, max_evaluations=1)
