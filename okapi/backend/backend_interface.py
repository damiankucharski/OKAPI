import numpy as np


class BackendInterface:
    """
    Abstract interface for tensor backends used by OKAPI.

    This interface defines the tensor operations required by OKAPI, allowing
    for different backend implementations (e.g., NumPy, PyTorch) to be used
    interchangeably. Each backend must implement all these methods to provide
    a consistent interface for tensor operations.
    """

    @staticmethod
    def tensor(x):
        raise NotImplementedError()

    @staticmethod
    def concat(tensors, axis=0):
        raise NotImplementedError()

    @staticmethod
    def mean(x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def max(x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def min(x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def sum(x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def argmax(x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def argmin(x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def to_numpy(x) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def clip(x, min, max):
        raise NotImplementedError()

    @staticmethod
    def log(x):
        raise NotImplementedError()

    @staticmethod
    def to_float(x):
        raise NotImplementedError()

    @staticmethod
    def shape(x):
        raise NotImplementedError()

    @staticmethod
    def reshape(x, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def squeeze(x):
        raise NotImplementedError()

    @staticmethod
    def unsqueeze(x, axis):
        raise NotImplementedError()

    @staticmethod
    def load(path, device=None):
        raise NotImplementedError()
