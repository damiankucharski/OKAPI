import numpy as np
import pytest

from okapi.backend.numpy_backend import NumpyBackend
from okapi.backend.pytorch import PyTorchBackend
from okapi import functions


@pytest.mark.parametrize("backend", [NumpyBackend, PyTorchBackend])
def test_zero_mass_after_multiclass_fusion(monkeypatch, backend):
    monkeypatch.setattr(functions, "B", backend)
    votes = backend.tensor(np.array([[0, 0, 0], [.2, .6, 0]], dtype=np.float32))
    actual = functions.scale_vector_to_sum_1(votes)
    np.testing.assert_allclose(backend.to_numpy(actual), [[1/3]*3, [.25, .75, 0]])
    assert actual.dtype == votes.dtype
