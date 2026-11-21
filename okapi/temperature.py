"""Fit the native multiclass log-probability pool's one global multiplier."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import brentq

from okapi.backend.numpy_backend import NumpyBackend
from okapi.functions import scale_vector_to_sum_1
from okapi.globals import BACKEND, use_postprocessing_function
from okapi.node import LogitMeanNode, ValueNode
from okapi.tree import Tree


def _probabilities(providers, *, copy=False):
    if not isinstance(providers, Mapping) or len(providers) < 2:
        raise ValueError('Supply a mapping of at least two provider IDs to [rows, classes] probabilities')
    arrays = {}
    shape = None
    for key, values in providers.items():
        if isinstance(key, bool) or not isinstance(key, (str, Integral)):
            raise ValueError('Provider IDs must be strings or integers')
        x = np.asarray(values)
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2 or x.dtype.kind not in 'fiu':
            raise ValueError('Each provider must supply a numeric [rows, classes] matrix with N >= 1 and C >= 2')
        if shape is None:
            shape = x.shape
        elif x.shape != shape:
            raise ValueError('All providers must have the same row and class axes')
        if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError('Every probability must be finite and in [0, 1]')
        tolerance = max(1e-12, 2 * np.finfo(x.dtype).eps) if x.dtype.kind == 'f' else 0
        x = x.astype(np.float64, copy=copy)
        if not np.allclose(x.sum(axis=1), 1, atol=tolerance, rtol=0):
            raise ValueError('Each provider probability row must sum to one')
        arrays[key] = x
    return arrays


def _active_tensor(value):
    """Reuse fit buffers already on the active backend; convert only on rebinding."""
    if BACKEND.get_backend() is NumpyBackend:
        return value if isinstance(value, np.ndarray) else value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        # NumPy views with negative strides cannot be converted directly by Torch.
        return BACKEND.tensor(np.ascontiguousarray(value))
    return value


@dataclass(frozen=True)
class TemperatureFit:
    """A one-operator native tree and its fit-only mean log losses and pass counts."""

    tree: Tree
    temperature: float
    lower: float
    upper: float
    reference_nll: float
    fit_nll: float
    gradient_evaluations: int
    nll_evaluations: int

    def predict(self, providers=None):
        """Return float64 NumPy probabilities with native clipping and row normalization.

        Supply fresh arrays under the fitted provider IDs and class order. Omitting
        them uses the fit rows. Both use the currently active NumPy or PyTorch
        backend; a previous fit on the other backend retains its original buffers.
        """
        bound = {node.id: node.value for node in self.tree.value_nodes}
        if providers is not None:
            arrays = _probabilities(providers)
            if arrays.keys() != bound.keys() or next(iter(arrays.values())).shape[1] != self.tree.root.value.shape[1]:
                raise ValueError('Supply the fitted provider IDs and number of classes')
            bound = arrays
        bound = {key: _active_tensor(value) for key, value in bound.items()}
        with use_postprocessing_function(scale_vector_to_sum_1):
            prediction = self.tree.do_pred_on_another_tensors(current_tensors=bound)
        return BACKEND.to_numpy(prediction)


def fit_multiclass_temperature_tree(providers, labels) -> TemperatureFit:
    """Fit `softmax(T * mean_provider(log(clip(p))))` to integer labels by mean NLL.

    Supply an ordered mapping of provider IDs to N-by-C arrays, C >= 2. Input values
    are promoted to float64 before clipping to [1e-6, 1-1e-6]; no input is mutated.
    Rows and physical class columns must be aligned across providers. The class index
    vector may omit classes. The fitted tree contains one existing LogitMeanNode.
    Its temperature multiplies logits (the inverse of the usual softmax divisor).

    The search interval starts at 0.05 and ends at the lesser of 50 and the maximum
    multiplier that keeps all fit log scores within the node's output clamp. On this
    interval the mean NLL is convex. Solve its derivative or choose the minimizing
    endpoint. A constant objective returns the feasible value nearest one. Report
    full-data derivative evaluations and separate full-data NLL evaluations at the
    solution and reference T=1 (reuse the value when they coincide). The frozen
    node can encounter its output clamp on new rows. `TemperatureFit.predict` scopes
    the standard OKAPI row normalization for both fit data and new predictions.
    """
    arrays = _probabilities(providers, copy=True)
    n, classes = next(iter(arrays.values())).shape
    y = np.asarray(labels)
    if y.shape == (n, 1):
        y = y[:, 0]
    if y.shape != (n,) or y.dtype.kind not in 'iu' or np.any((y < 0) | (y >= classes)):
        raise ValueError('Supply one integer class index in [0, C) per probability row')

    # Match LogitMeanNode.calculate's input order and streaming float64 mean.
    logs = np.zeros((n, classes), dtype=np.float64)
    for x in arrays.values():
        logs += np.log(np.clip(x, LogitMeanNode._EPS, 1 - LogitMeanNode._EPS))
    logs /= len(arrays)
    lower = 0.05
    largest = float(-logs.min())
    upper = min(50.0, LogitMeanNode._CLAMP / largest)
    if largest * upper > LogitMeanNode._CLAMP:
        upper = float(np.nextafter(upper, 0.0))
    centered = logs - logs.max(axis=1, keepdims=True)
    target_scores = centered[np.arange(n), y]
    gaps = centered - target_scores[:, None]
    gradient_cache = {}

    def gradient(temperature):
        if temperature not in gradient_cache:
            weights = np.exp(temperature * centered)
            row_gradient = (weights * gaps).sum(axis=1) / weights.sum(axis=1)
            gradient_cache[temperature] = float(row_gradient.mean())
        return gradient_cache[temperature]

    def nll(temperature):
        log_mass = np.log(np.exp(temperature * centered).sum(axis=1))
        return float(np.mean(log_mass - temperature * target_scores))

    if not centered.any():
        temperature = float(np.clip(1.0, lower, upper))
    elif gradient(lower) >= 0:
        temperature = lower
    elif gradient(upper) <= 0:
        temperature = upper
    else:
        temperature = float(brentq(gradient, lower, upper, xtol=1e-12, rtol=1e-14))
    reference = nll(1.0)
    fitted = reference if temperature == 1.0 else nll(temperature)
    root, *others = (ValueNode(None, _active_tensor(x), key) for key, x in arrays.items())
    root.add_child(LogitMeanNode(others, temperature=temperature, shift=0.0))
    return TemperatureFit(Tree(root), temperature, lower, upper, reference, fitted, len(gradient_cache), 1 + int(temperature != 1.0))
