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


# ---- Per-base-model metadata (e.g. trust scores) read by ValueNodes ----
class _ModelMetadataRegistry:
    """Global ``model_id -> metadata dict`` map consulted by ``ValueNode`` at
    creation, so meta signals computed once by the caller (e.g. trust from
    ``okapi.meta.model_trust``) reach every value node — initial population,
    mutation, crossover-copy and seeding — without threading them through the GP.
    """

    def __init__(self):
        self._d: dict = {}

    def set(self, mapping):
        self._d = dict(mapping) if mapping else {}

    def get(self, model_id):
        return self._d.get(model_id, {})

    def clear(self):
        self._d = {}


model_metadata = _ModelMetadataRegistry()


def set_model_metadata(mapping):
    """Set the global ``model_id -> metadata`` map, e.g. ``{id: {"trust": t}}``."""
    model_metadata.set(mapping)


def get_model_metadata(model_id):
    """Metadata dict for a base-model id (empty if none registered)."""
    return model_metadata.get(model_id)


def clear_model_metadata():
    model_metadata.clear()


# ---- Per-fit-evaluation context (ground truth) read by supervised operators ----
class _EvalContext:
    """Holds the fit-split ground truth ``y`` during fitness evaluation, so a
    *supervised* operator (e.g. ``IdealPointTrustNode``) can score each input's
    evaluation against the ideal point. The engine sets it around a fitness pass and
    clears it afterwards, so it is present **only** during fit and absent at prediction
    time — where supervised operators must fall back to weights cached during fit.
    """

    def __init__(self):
        self._y = None

    def set(self, y):
        self._y = y

    def get(self):
        return self._y

    def clear(self):
        self._y = None


eval_context = _EvalContext()


def set_eval_context(y):
    """Make the fit ground-truth ``y`` available to supervised operators during a fitness pass."""
    eval_context.set(y)


def get_eval_context():
    """Fit ground-truth ``y`` while a fitness pass is active, else ``None`` (prediction time)."""
    return eval_context.get()


def clear_eval_context():
    eval_context.clear()


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
