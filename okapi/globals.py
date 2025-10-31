import os

from loguru import logger

from okapi.backend.backend import Backend

# Device to use for tensor operations, can be set via environment variable
DEVICE = os.environ.get("DEVICE", None)
logger.debug(f"Using device: {DEVICE if DEVICE else 'default'}")


# ---- Postprocessing functions ----
def _passthrough(x):
    """
    Default postprocessing function that simply returns the input unchanged.

    Args:
        x: Input to pass through

    Returns:
        The unchanged input
    """
    return x


# Global postprocessing function that will be applied to tree evaluations
class Postprocessor:
    def __init__(self):
        self._postprocessing_function = _passthrough

    def __call__(self, x):
        return self._postprocessing_function(x)

    def set_postprocessing_function(self, func):
        self._postprocessing_function = func


postprocessing_function = Postprocessor()


def set_postprocessing_function(func):
    """
    Set the global postprocessing function.

    Args:
        func: The function to use for postprocessing tree evaluations
    """
    logger.info(f"Setting global postprocessing function to: {func.__name__}")
    global postprocessing_function
    postprocessing_function.set_postprocessing_function(func)


# ---- Backend configuration ----
# Initialize the backend based on environment variable or default to numpy
backend_name = os.environ.get("BACKEND", "numpy")
logger.info(f"Initializing backend from environment: {backend_name}")
Backend.set_backend(backend_name)
BACKEND: Backend = Backend()


def set_backend(backend_name):
    """
    Set the tensor backend to use.

    Args:
        backend_name: Name of the backend to use ('numpy' or 'pytorch')
    """
    global Backend
    logger.info(f"Setting tensor backend to: {backend_name}")
    Backend.set_backend(backend_name)


def get_backend():
    """
    Get the current tensor backend.

    Returns:
        The current backend interface class
    """
    global Backend
    backend = Backend.get_backend()
    logger.debug(f"Retrieved current backend: {backend.__name__}")
    return backend


# ----
