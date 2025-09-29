from typing import Type

from loguru import logger

from giraffe.backend.backend_interface import BackendInterface
from giraffe.backend.numpy_backend import NumpyBackend


class Backend:
    """
    Factory class for managing tensor backends in GIRAFFE.

    This class provides a centralized way to set and retrieve the tensor backend
    implementation (NumPy or PyTorch) used throughout the GIRAFFE library.

    Important: The backend should only be set at the beginning of the program,
    before any GIRAFFE instances are initialized or predictions are loaded.
    """

    _current_backend: Type[BackendInterface] = NumpyBackend

    @classmethod
    def set_backend(cls, backend_name: str):  # TODO: Add option to set backend by providing class instead
        """
        Set the active tensor backend by name.

        Available backends:
        - 'numpy': Uses NumPyBackend for tensor operations
        - 'torch' or 'pytorch': Uses PyTorchBackend for tensor operations

        Args:
            backend_name: String identifier for the backend

        Raises:
            ValueError: If the backend name is not recognized
        """
        logger.info(f"Setting tensor backend to '{backend_name}'")
        if backend_name == "torch" or backend_name == "pytorch":
            from giraffe.backend.pytorch import PyTorchBackend

            cls._current_backend = PyTorchBackend
            logger.debug("PyTorch backend initialized successfully")
        elif backend_name == "numpy":
            cls._current_backend = NumpyBackend
            logger.debug("NumPy backend initialized successfully")
        else:
            logger.error(f"Invalid backend: {backend_name}")
            raise ValueError(f"Invalid backend: {backend_name}")

    @classmethod
    def get_backend(cls) -> Type[BackendInterface]:
        """
        Get the current tensor backend.

        Returns:
            The current backend implementation class (NumpyBackend or PyTorchBackend)
        """
        logger.trace(f"Getting current backend: {cls._current_backend.__name__}")
        return cls._current_backend

    def __init__(self):
        pass

    def __getattr__(self, name):
        return getattr(Backend._current_backend, name)
