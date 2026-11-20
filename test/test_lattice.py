import numpy as np
import pytest
import torch

from okapi.globals import set_backend, set_postprocessing_function
from okapi.lattice import compile_binary_lattice
from okapi.node import MaxNode, MeanNode, MinNode, ValueNode
from okapi.tree import Tree


def test_sequential_absorption_and_duplicate_affine_compilation():
    set_backend("numpy")
    set_postprocessing_function(lambda x: x)
    rng = np.random.default_rng(82)
    raw = rng.normal(size=(128, 3))
    w, b = rng.normal(size=(2, 3)), rng.normal(size=2)
    p = 1 / (1 + np.exp(-raw @ w.T - b))
    left = ValueNode(None, p[:, 0], "a")
    alternative = ValueNode(None, p[:, 0], "a_copy")
    alternative.add_child(MinNode([ValueNode(None, p[:, 1], "b")]))
    left.add_child(MaxNode([alternative]))
    program = Tree.create_tree_from_root(left)
    affines = {"a": (w[0], b[0]), "a_copy": (w[0], b[0]), "b": (w[1], b[1])}
    circuit = compile_binary_lattice(program, affines)
    reference = program.predict()
    compiled = circuit(torch.from_numpy(raw)).sigmoid().detach().numpy()
    np.testing.assert_allclose(compiled, reference, atol=1e-14)
    assert len(circuit.gates) == 0 and circuit.leaves.out_features == 1
    assert circuit.original_gates == 2


def test_unsupported_operator_is_not_silently_approximated():
    a = ValueNode(None, np.array([0.5]), "a")
    a.add_child(MeanNode([ValueNode(None, np.array([0.1]), "b")]))
    affines = {i: (np.ones(2), 0.0) for i in ("a", "b")}
    with pytest.raises(ValueError, match="MeanNode"):
        compile_binary_lattice(Tree.create_tree_from_root(a), affines)
