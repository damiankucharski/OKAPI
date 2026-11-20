"""Scope built-in GP operators to the active optimizer's random stream."""
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np


# Standalone operators historically follow np.random.seed().
_active_rng = ContextVar("okapi_rng", default=None)


def current_rng():
    stream = _active_rng.get()
    return np.random if stream is None else stream


@contextmanager
def use_rng(stream):
    token = _active_rng.set(stream)
    try:
        yield
    finally:
        _active_rng.reset(token)
