"""Tests for GPU memory optimizations:

1. `torch.no_grad()` wrapping in evaluation path
2. Eager child ValueNode.evaluation freeing during _concat
3. Streaming reductions bypassing the concat buffer for Mean/WeightedMean/Min/Max
"""
from contextlib import nullcontext
from unittest import mock

import numpy as np
import pytest
import torch

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.backend.pytorch import PyTorchBackend
from okapi.fitness import (
    accuracy_fitness,
    average_precision_fitness,
    roc_auc_score_fitness,
)
from okapi.node import (
    CloseThresholdNode,
    MaxNode,
    MeanNode,
    MinNode,
    Node,
    OperatorNode,
    ValueNode,
    WeightedMeanNode,
)
from okapi.tree import Tree


# ---------------- shared fixtures ----------------


@pytest.fixture
def restore_backend():
    """Restore the original backend after a test that switches it."""
    original = Backend.get_backend()
    yield
    Backend._current_backend = original


@pytest.fixture
def restore_eager_free_flag():
    """Restore Node._EAGER_FREE_EVALS after a test that toggles it."""
    original = getattr(Node, "_EAGER_FREE_EVALS", False)
    yield
    Node._EAGER_FREE_EVALS = original


def _make_mean_tree_torch(requires_grad: bool = False):
    """A -> MeanNode -> [C, D] using torch tensors."""
    a = torch.tensor([[0.2, 0.2], [0.3, 0.3]], requires_grad=requires_grad)
    c = torch.tensor([[0.3, 0.3], [0.4, 0.4]], requires_grad=requires_grad)
    d = torch.tensor([[0.4, 0.4], [0.5, 0.5]], requires_grad=requires_grad)

    A = ValueNode(None, a, 1)
    C = ValueNode(None, c, 2)
    D = ValueNode(None, d, 3)
    B_ = MeanNode([C, D])
    A.add_child(B_)
    return {"A": A, "B": B_, "C": C, "D": D, "a": a, "c": c, "d": d}


def _make_mean_tree_numpy():
    a = np.array([[2.0, 2.0], [3.0, 3.0]])
    c = np.array([[3.0, 3.0], [4.0, 4.0]])
    d = np.array([[4.0, 4.0], [5.0, 5.0]])

    A = ValueNode(None, a, 1)
    C = ValueNode(None, c, 2)
    D = ValueNode(None, d, 3)
    B_ = MeanNode([C, D])
    A.add_child(B_)
    return {"A": A, "B": B_, "C": C, "D": D}


# ============================================================
# Fix #1 — torch.no_grad() in evaluation path
# ============================================================


class TestNoGradWrapping:
    def test_tree_predict_returns_detached_tensor_with_grad_input(self, restore_backend):
        """If input ValueNodes carry requires_grad, tree.predict() output must be
        detached (grad_fn is None). We respect the user's requires_grad but do not
        build a graph through our operations."""
        Backend.set_backend("pytorch")
        nodes = _make_mean_tree_torch(requires_grad=True)
        tree = Tree.create_tree_from_root(nodes["A"])

        pred = tree.predict()

        assert pred.grad_fn is None, (
            "tree.predict() must return a tensor without a grad_fn; "
            "found one, meaning autograd is building a graph through evaluation"
        )

    def test_tree_predict_preserves_user_requires_grad_on_value_nodes(self, restore_backend):
        """The user's input tensors must be left untouched — same requires_grad as before."""
        Backend.set_backend("pytorch")
        nodes = _make_mean_tree_torch(requires_grad=True)
        tree = Tree.create_tree_from_root(nodes["A"])

        _ = tree.predict()

        assert nodes["a"].requires_grad is True
        assert nodes["c"].requires_grad is True
        assert nodes["d"].requires_grad is True

    def test_tree_predict_without_grad_inputs_still_works(self, restore_backend):
        """Sanity: no_grad wrapping does not break the common case of plain tensors."""
        Backend.set_backend("pytorch")
        nodes = _make_mean_tree_torch(requires_grad=False)
        tree = Tree.create_tree_from_root(nodes["A"])

        pred = tree.predict()

        expected = np.array([[0.3, 0.3], [0.4, 0.4]])
        np.testing.assert_allclose(pred.detach().cpu().numpy(), expected)

    @pytest.mark.parametrize(
        "fitness_fn",
        [average_precision_fitness, roc_auc_score_fitness, accuracy_fitness],
    )
    def test_fitness_functions_do_not_build_graph(self, restore_backend, fitness_fn):
        """Each fitness function, when called on a Tree whose leaf tensors require
        grad, must run under no_grad so tree.predict() inside returns a detached
        tensor and the returned score is a plain float."""
        Backend.set_backend("pytorch")
        nodes = _make_mean_tree_torch(requires_grad=True)
        tree = Tree.create_tree_from_root(nodes["A"])
        gt = torch.tensor([[1, 0], [1, 1]])

        observed = {}
        original_predict = Tree.predict

        def spy_predict(self, *a, **kw):
            out = original_predict(self, *a, **kw)
            observed["grad_fn"] = out.grad_fn
            return out

        with mock.patch.object(Tree, "predict", spy_predict):
            score = fitness_fn(tree, gt, task="multilabel")

        assert isinstance(score, float)
        assert observed["grad_fn"] is None


# ============================================================
# Fix #2 — Eager child ValueNode.evaluation freeing
# ============================================================


class TestEagerEvalFreeing:
    def test_flag_defaults_to_false(self):
        assert getattr(Node, "_EAGER_FREE_EVALS", False) is False

    def test_flag_off_preserves_child_evaluations(self):
        """With the flag off, existing behavior: after parent's calculate(), children's
        .evaluation is set (for inspection / test compatibility)."""
        nodes = _make_mean_tree_numpy()
        nodes["A"].calculate()

        assert nodes["C"].evaluation is not None
        assert nodes["D"].evaluation is not None

    def test_flag_on_nulls_child_evaluations_after_concat(self, restore_eager_free_flag):
        """With flag on, _concat eagerly frees each child's .evaluation as soon as
        the concat buffer has copied them in."""
        Node._EAGER_FREE_EVALS = True
        nodes = _make_mean_tree_numpy()
        nodes["A"].calculate()

        assert nodes["C"].evaluation is None
        assert nodes["D"].evaluation is None

    def test_flag_on_never_touches_value_of_base_value_node(self, restore_eager_free_flag):
        """_EAGER_FREE_EVALS must not touch ValueNode.value (the base tensor)."""
        Node._EAGER_FREE_EVALS = True
        nodes = _make_mean_tree_numpy()
        expected_c_value = nodes["C"].value.copy()
        expected_d_value = nodes["D"].value.copy()

        nodes["A"].calculate()

        np.testing.assert_array_equal(nodes["C"].value, expected_c_value)
        np.testing.assert_array_equal(nodes["D"].value, expected_d_value)

    def test_flag_on_nulls_parent_evaluation_when_used(self, restore_eager_free_flag):
        """If parent.evaluation was used by _concat (not parent.value), it should be
        nulled after concat. parent.value must remain intact."""
        Node._EAGER_FREE_EVALS = True
        a = np.array([[2.0, 2.0], [3.0, 3.0]])
        c = np.array([[3.0, 3.0], [4.0, 4.0]])
        d = np.array([[4.0, 4.0], [5.0, 5.0]])

        parent = ValueNode(None, a, 1)
        C = ValueNode(None, c, 2)
        D = ValueNode(None, d, 3)
        op = OperatorNode([C, D])
        parent.add_child(op)

        # Pre-populate parent.evaluation so _concat uses it instead of parent.value
        parent.evaluation = np.array([[1.0, 1.0], [1.0, 1.0]])

        _ = op._concat()

        assert parent.evaluation is None, "parent.evaluation should be freed after concat used it"
        np.testing.assert_array_equal(parent.value, a), "parent.value must be untouched"

    def test_tree_predict_activates_eager_free_during_traversal(self, restore_backend):
        """Tree.predict(clear_cache=True) must enable the flag DURING traversal so
        peak memory is reduced (not only at the end via _clean_evals). We probe
        with a ThresholdNode tree since reducing ops now stream and bypass
        _concat — ThresholdNode still uses _concat, so it's the reliable probe."""
        Backend.set_backend("pytorch")
        a = torch.tensor([[0.2, 0.2], [0.3, 0.3]])
        c = torch.tensor([[0.3, 0.3], [0.4, 0.4]])
        d = torch.tensor([[0.4, 0.4], [0.5, 0.5]])
        A = ValueNode(None, a, 1)
        C = ValueNode(None, c, 2)
        D = ValueNode(None, d, 3)
        op = CloseThresholdNode([C, D], 0.4)
        A.add_child(op)
        tree = Tree.create_tree_from_root(A)

        observed = {}
        original_concat = OperatorNode._concat

        def spy_concat(self):
            # Capture flag state DURING _concat (before post-traversal cleanup)
            observed.setdefault("flag_during_concat", Node._EAGER_FREE_EVALS)
            return original_concat(self)

        with mock.patch.object(OperatorNode, "_concat", spy_concat):
            _ = tree.predict(clear_cache=True)

        assert observed.get("flag_during_concat") is True, (
            "Node._EAGER_FREE_EVALS must be True while _concat runs inside tree.predict(), "
            "so intermediate evals are freed eagerly"
        )
        # Flag must be restored afterwards (try/finally semantics)
        assert Node._EAGER_FREE_EVALS is False


# ============================================================
# Fix #3 — Streaming reductions bypass the concat buffer
# ============================================================


class TestStreamingReductions:
    @staticmethod
    def _concat_call_counter():
        """Returns (counter, patched_fn) — patched_fn is a regular function so it
        binds `self` via the descriptor protocol, unlike a raw Mock."""
        counter = {"n": 0}
        original = OperatorNode._concat

        def counting(self):
            counter["n"] += 1
            return original(self)

        return counter, counting

    @pytest.mark.parametrize(
        "op_cls, weights",
        [
            (MeanNode, None),
            (MinNode, None),
            (MaxNode, None),
            (WeightedMeanNode, [0.3, 0.2, 0.5]),
        ],
    )
    def test_reducing_op_does_not_call_concat(self, op_cls, weights):
        """Reducing ops (mean/weighted_mean/min/max) must compute via a streaming
        path, bypassing _concat which would materialize an (N+1, ...) buffer."""
        a = np.array([[2.0, 2.0], [3.0, 3.0]])
        c = np.array([[3.0, 3.0], [4.0, 4.0]])
        d = np.array([[4.0, 4.0], [5.0, 5.0]])

        A = ValueNode(None, a, 1)
        C = ValueNode(None, c, 2)
        D = ValueNode(None, d, 3)
        if weights is not None:
            op = op_cls([C, D], weights)
        else:
            op = op_cls([C, D])
        A.add_child(op)

        counter, patched = self._concat_call_counter()
        with mock.patch.object(OperatorNode, "_concat", patched):
            A.calculate()

        assert counter["n"] == 0, (
            f"{op_cls.__name__}.calculate() must not call _concat (streaming path expected); "
            f"got {counter['n']} call(s)"
        )

    def test_threshold_still_uses_concat(self):
        """ThresholdNode needs the full stack for argmin/argmax, so it keeps the
        concat path."""
        a = np.array([[0.2, 0.2], [0.3, 0.3]])
        c = np.array([[0.3, 0.3], [0.4, 0.4]])
        d = np.array([[0.4, 0.4], [0.5, 0.5]])

        A = ValueNode(None, a, 1)
        C = ValueNode(None, c, 2)
        D = ValueNode(None, d, 3)
        op = CloseThresholdNode([C, D], 0.4)
        A.add_child(op)

        counter, patched = self._concat_call_counter()
        with mock.patch.object(OperatorNode, "_concat", patched):
            A.calculate()

        assert counter["n"] >= 1, "ThresholdNode must still use _concat"

    # Numerical correctness is already covered by test_node.py (test_mean,
    # test_weighted_mean, test_min_tree, test_max_tree, test_weighted_mean_child_*)
    # Those tests hit the same calculate() paths and act as regressions. We add
    # a torch-backend correctness spot-check here to confirm parity across backends.
    def test_mean_streaming_torch_matches_numpy(self, restore_backend):
        Backend.set_backend("pytorch")
        t = _make_mean_tree_torch(requires_grad=False)
        tree = Tree.create_tree_from_root(t["A"])
        pred_torch = tree.predict().detach().cpu().numpy()

        Backend.set_backend("numpy")
        n = _make_mean_tree_numpy()
        # Scale numpy fixture values to match torch fixture values
        n["A"].value = np.array([[0.2, 0.2], [0.3, 0.3]])
        n["C"].value = np.array([[0.3, 0.3], [0.4, 0.4]])
        n["D"].value = np.array([[0.4, 0.4], [0.5, 0.5]])
        n["A"].evaluation = n["C"].evaluation = n["D"].evaluation = None
        tree_n = Tree.create_tree_from_root(n["A"])
        pred_numpy = tree_n.predict()

        np.testing.assert_allclose(pred_torch, pred_numpy, rtol=1e-6)

    def test_min_streaming_correctness(self):
        a = np.array([[2.0, 2.0], [3.0, 3.0]])
        c = np.array([[3.0, 3.0], [4.0, 4.0]])
        d = np.array([[4.0, 4.0], [5.0, 5.0]])
        A = ValueNode(None, a, 1)
        op = MinNode([ValueNode(None, c, 2), ValueNode(None, d, 3)])
        A.add_child(op)
        A.calculate()
        np.testing.assert_array_equal(A.evaluation, np.array([[2.0, 2.0], [3.0, 3.0]]))

    def test_max_streaming_correctness(self):
        a = np.array([[2.0, 2.0], [3.0, 3.0]])
        c = np.array([[3.0, 3.0], [4.0, 4.0]])
        d = np.array([[4.0, 4.0], [5.0, 5.0]])
        A = ValueNode(None, a, 1)
        op = MaxNode([ValueNode(None, c, 2), ValueNode(None, d, 3)])
        A.add_child(op)
        A.calculate()
        np.testing.assert_array_equal(A.evaluation, np.array([[4.0, 4.0], [5.0, 5.0]]))


# ============================================================
# Backend additions: minimum / maximum / no_grad
# ============================================================


class TestBackendAdditions:
    @pytest.mark.parametrize("B", [NumpyBackend, PyTorchBackend])
    def test_minimum_elementwise(self, B):
        a = B.tensor([[1.0, 4.0], [3.0, 2.0]])
        b = B.tensor([[2.0, 3.0], [3.0, 1.0]])
        result = B.to_numpy(B.minimum(a, b))
        np.testing.assert_array_equal(result, np.array([[1.0, 3.0], [3.0, 1.0]]))

    @pytest.mark.parametrize("B", [NumpyBackend, PyTorchBackend])
    def test_maximum_elementwise(self, B):
        a = B.tensor([[1.0, 4.0], [3.0, 2.0]])
        b = B.tensor([[2.0, 3.0], [3.0, 1.0]])
        result = B.to_numpy(B.maximum(a, b))
        np.testing.assert_array_equal(result, np.array([[2.0, 4.0], [3.0, 2.0]]))

    def test_numpy_no_grad_is_nullcontext(self):
        ctx = NumpyBackend.no_grad()
        # Should be a context manager that does nothing
        with ctx:
            assert True

    def test_pytorch_no_grad_disables_autograd(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        with PyTorchBackend.no_grad():
            y = x * 2
            assert y.grad_fn is None
        # Outside the context, autograd is back on
        z = x * 2
        assert z.grad_fn is not None
