import torch
from loguru import logger

from okapi.backend.backend_interface import BackendInterface


class PyTorchBackend(BackendInterface):
    # rewrite all not call the already defined functions

    @staticmethod
    def tensor(x):
        return torch.tensor(x)

    @staticmethod
    def concat(tensors, axis=0):
        return torch.cat(tensors, dim=axis)

    @staticmethod
    def mean(x, axis=None):
        return torch.mean(x.float(), dim=axis)

    @staticmethod
    def max(x, axis=None):
        if axis is None:
            return torch.max(x)
        return torch.max(x, dim=axis).values

    @staticmethod
    def min(x, axis=None):
        if axis is None:
            return torch.min(x)
        return torch.min(x, dim=axis).values

    @staticmethod
    def sum(x, axis=None):
        return torch.sum(x, dim=axis)

    @staticmethod
    def argmax(x, axis=None):
        return torch.argmax(x, dim=axis)

    @staticmethod
    def argmin(x, axis=None):
        return torch.argmin(x, dim=axis)

    @staticmethod
    def to_numpy(x):
        return x.detach().numpy()

    @staticmethod
    def clip(x, min, max):
        return torch.clamp(x, min, max)

    @staticmethod
    def log(x):
        return torch.log(x)

    @staticmethod
    def to_float(x):
        return x.float()

    @staticmethod
    def shape(x):
        return x.shape

    @staticmethod
    def reshape(x, *args, **kwargs):
        return x.reshape(*args, **kwargs)

    @staticmethod
    def squeeze(x):
        return x.squeeze()

    @staticmethod
    def unsqueeze(x, axis):
        return x.unsqueeze(axis)

    @staticmethod
    def load(path, device=None):
        if not any([str(path).endswith(suffix) for suffix in [".pt", ".pth"]]):
            logger.warning(f"file extension for {path} is different from common pytorch extensions: .pt or .pth")
        loaded = torch.load(path, map_location=device)
        if not isinstance(loaded, torch.Tensor):
            raise ValueError(f"file {path}  is not a torch.Tensor")
        return loaded
