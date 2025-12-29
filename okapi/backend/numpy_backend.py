import numpy as np
from loguru import logger

from okapi.backend.backend_interface import BackendInterface


class NumpyBackend(BackendInterface):
    @staticmethod
    def tensor(x):
        return np.asarray(x)

    @staticmethod
    def concat(tensors, axis=0):
        return np.concatenate(tensors, axis)

    @staticmethod
    def mean(x, axis=None):
        return np.mean(x, axis)

    @staticmethod
    def max(x, axis=None):
        return np.max(x, axis)

    @staticmethod
    def min(x, axis=None):
        return np.min(x, axis)

    @staticmethod
    def sum(x, axis=None):
        return np.sum(x, axis)

    @staticmethod
    def argmax(x, axis=None):
        return np.argmax(x, axis=axis)

    @staticmethod
    def argmin(x, axis=None):
        return np.argmin(x, axis=axis)

    @staticmethod
    def to_numpy(x):
        return x

    @staticmethod
    def clip(x, min, max):
        return np.clip(x, min, max)

    @staticmethod
    def log(x):
        return np.log(x)

    @staticmethod
    def to_float(x):
        return x.astype(float)

    @staticmethod
    def shape(x):
        return x.shape

    @staticmethod
    def reshape(x, *args, **kwargs):
        return np.reshape(x, *args, **kwargs)

    @staticmethod
    def squeeze(x):
        return np.squeeze(x)

    @staticmethod
    def unsqueeze(x, axis):
        return np.expand_dims(x, axis)

    @staticmethod
    def load(path, device=None):
        if not any([str(path).endswith(suffix) for suffix in [".npy", ".npz"]]):
            logger.warning(f"file extension for {path} is different from common numpy extensions: .npy or .npz")
        loaded = np.load(path)
        if not isinstance(loaded, np.ndarray):
            raise ValueError(f"file {path} is not a numpy file")
        return loaded

    @staticmethod
    def clone(x):
        return x.copy()
