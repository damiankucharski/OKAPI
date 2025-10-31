import numpy as np
import pytest

from okapi.backend.numpy_backend import NumpyBackend
from okapi.backend.pytorch import PyTorchBackend

BACKENDS = [NumpyBackend, PyTorchBackend]


@pytest.mark.parametrize(
    "arrays, axis, expected_shape",
    [
        ([[1, 2], [3, 4]], 0, (4)),
        (
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ],
            0,
            (4, 2),
        ),
    ],
)
def test_concat(arrays, axis, expected_shape):
    for B in BACKENDS:
        tensors = [B.tensor(array) for array in arrays]
        result = B.to_numpy(B.concat(tensors, axis))
        np.testing.assert_array_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "array, expected_mean, axis, expected_shape",
    [
        ([1, 2, 3, 4], 2.5, None, ()),
        ([[1, 2], [3, 4]], [2.0, 3.0], 0, (2,)),
        ([[1, 2], [3, 4]], [1.5, 3.5], 1, (2,)),
    ],
)
def test_mean(array, expected_mean, axis, expected_shape):
    for B in BACKENDS:
        tensor = B.tensor(array)
        result = B.to_numpy(B.mean(tensor, axis=axis))
        np.testing.assert_array_equal(result, expected_mean)
        np.testing.assert_array_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "array, expected_sum, axis, expected_shape",
    [
        ([1, 2, 3, 4], 10, None, ()),
        ([[1, 2], [3, 4]], [4, 6], 0, (2,)),
        ([[1, 2], [3, 4]], [3, 7], 1, (2,)),
    ],
)
def test_sum(array, expected_sum, axis, expected_shape):
    for B in BACKENDS:
        tensor = B.tensor(array)
        result = B.to_numpy(B.sum(tensor, axis=axis))
        np.testing.assert_array_equal(result, expected_sum)
        np.testing.assert_array_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "array, expected_max, axis, expected_shape",
    [
        ([1, 2, 3, 4], 4, None, ()),
        ([[1, 2], [3, 4]], [3, 4], 0, (2,)),
        ([[1, 2], [3, 4]], [2, 4], 1, (2,)),
    ],
)
def test_max(array, expected_max, axis, expected_shape):
    for B in BACKENDS:
        tensor = B.tensor(array)
        result = B.to_numpy(B.max(tensor, axis=axis))
        np.testing.assert_array_equal(result, expected_max)
        np.testing.assert_array_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "array, expected_min, axis, expected_shape",
    [
        ([1, 2, 3, 4], 1, None, ()),
        ([[1, 2], [3, 4]], [1, 2], 0, (2,)),
        ([[1, 2], [3, 4]], [1, 3], 1, (2,)),
    ],
)
def test_min(array, expected_min, axis, expected_shape):
    for B in BACKENDS:
        tensor = B.tensor(array)
        result = B.to_numpy(B.min(tensor, axis=axis))
        np.testing.assert_array_equal(result, expected_min)
        np.testing.assert_array_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "array, min_val, max_val, expected_result",
    [
        ([1, 2, 3, 4], 2, 3, [2, 2, 3, 3]),
        ([[1, 2], [3, 4]], 2, 3, [[2, 2], [3, 3]]),
    ],
)
def test_clip(array, min_val, max_val, expected_result):
    for B in BACKENDS:
        tensor = B.tensor(array)
        result = B.to_numpy(B.clip(tensor, min_val, max_val))
        np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    "array, new_shape, expected_shape",
    [
        ([1, 2, 3, 4], (2, 2), (2, 2)),
        ([[1, 2], [3, 4]], (4,), (4,)),
    ],
)
def test_reshape(array, new_shape, expected_shape):
    for B in BACKENDS:
        tensor = B.tensor(array)
        result = B.to_numpy(B.reshape(tensor, new_shape))
        np.testing.assert_array_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "array, expected_result",
    [
        ([[1], [2], [3], [4]], [1, 2, 3, 4]),
        ([[[1, 2]], [[3, 4]]], [[1, 2], [3, 4]]),
    ],
)
def test_squeeze(array, expected_result):
    for B in BACKENDS:
        tensor = B.tensor(array)
        result = B.to_numpy(B.squeeze(tensor))
        np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    "array, axis, expected_shape",
    [
        ([1, 2, 3, 4], 0, (1, 4)),
        ([[1, 2], [3, 4]], 1, (2, 1, 2)),
    ],
)
def test_unsqueeze(array, axis, expected_shape):
    for B in BACKENDS:
        tensor = B.tensor(array)
        result = B.to_numpy(B.unsqueeze(tensor, axis))
        np.testing.assert_array_equal(result.shape, expected_shape)
