"""Fit or transfer a binary probability or logit stacker onto OKAPI roots."""

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from okapi.node import BinaryAffineNode, ValueNode
from okapi.tree import Tree


def _validate_transform(transform, complementary):
    if transform not in ("probability", "logit"):
        raise ValueError("transform must be 'probability' or 'logit'")
    if not isinstance(complementary, (bool, np.bool_)):
        raise ValueError("complementary must be a boolean")


def stacking_tree(roots: Sequence[ValueNode], pipeline: Pipeline, *, transform: str, complementary: bool = True) -> Tree:
    """Represent a fitted StandardScaler + binary lbfgs LogisticRegression as a tree.

    The fitted feature order is negative, positive for each provider. Declare
    the original feature transform as 'probability' or 'logit' (log odds after
    clipping each channel to [1e-6, 1-1e-6]). Use identity node postprocessing.

    With complementary=True, supply one positive-class root per provider.
    This uses p_negative = 1 - p_positive to fold each coefficient pair into
    one weight. Independently rounded or non-complementary channels should
    instead be supplied as all 2M roots in original order with complementary=False.
    The logit reduction uses the mathematical antisymmetry of clipped log odds;
    finite-precision complements and clipping near 0 or 1 can affect parity.

    Roots may themselves be fitted programs. Each occurrence is copied with
    its subtree; the source programs and the sklearn estimators are untouched.
    The result predicts the positive probability with a 1e-12 binary
    probability floor and renormalization. This conversion performs no fitting.
    """
    _validate_transform(transform, complementary)
    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) != 2:
        raise ValueError("Expected a two-step StandardScaler and LogisticRegression pipeline")
    scaler, classifier = (step for _, step in pipeline.steps)
    if not isinstance(scaler, StandardScaler) or not isinstance(classifier, LogisticRegression):
        raise ValueError("Expected StandardScaler followed by LogisticRegression")
    check_is_fitted(scaler, "n_features_in_")
    check_is_fitted(classifier, ["coef_", "intercept_", "classes_"])
    if classifier.solver != "lbfgs" or getattr(classifier, "multi_class", None) == "multinomial":
        raise ValueError("Expected binary lbfgs logistic regression with a sigmoid decision function")
    if not np.array_equal(classifier.classes_, [0, 1]):
        raise ValueError("Expected ordered binary classes [0, 1]")

    n_features = scaler.n_features_in_
    if n_features % 2 or classifier.coef_.shape != (1, n_features):
        raise ValueError("Expected matching scaler and binary classifier with 2M feature columns")
    roots = tuple(roots)
    expected = n_features // 2 if complementary else n_features
    if len(roots) != expected or not all(isinstance(root, ValueNode) for root in roots):
        raise ValueError(f"Expected {expected} ValueNode roots in provider feature order")

    coefficients = np.asarray(classifier.coef_[0], dtype=float).copy()
    intercept = float(classifier.intercept_[0])
    if scaler.with_std:
        coefficients /= scaler.scale_
    if scaler.with_mean:
        intercept -= float(coefficients @ scaler.mean_)
    if complementary:
        if transform == "probability":
            intercept += float(coefficients[::2].sum())
        coefficients = coefficients[1::2] - coefficients[::2]

    parent, *children = (root.copy_subtree() for root in roots)
    parent.add_child(BinaryAffineNode(children, coefficients.tolist(), intercept=intercept, transform=transform))
    return Tree(parent)


def _fit_array(value):
    if hasattr(value, "detach"):
        from okapi.backend.pytorch import PyTorchBackend

        # PyTorch bfloat16 has no native NumPy representation. Match the
        # affine operator's promotion of half-precision probabilities.
        if value.is_floating_point() and value.element_size() < 4:
            value = PyTorchBackend.to_float(value)
        return PyTorchBackend.to_numpy(value)
    return np.asarray(value)


def fit_stacking_tree(
    roots: Sequence[ValueNode], labels, *, transform: str, complementary: bool = True,
    c_value: float = 1.0, max_iter: int = 2000,
) -> Tree:
    """Fit standardized binary logistic stacking to the current outputs of OKAPI roots.

    Supply fit-only scalar probability roots, each with shape (N,) or (N, 1),
    and matching labels containing both 0 and 1. Evaluation uses the active
    OKAPI backend on independent subtree copies. sklearn fits on the CPU.

    With complementary=True, provide M positive-class roots; both [1-p, p]
    channels for every provider enter the regularized fit. With False, supply
    all 2M scalar roots in negative/positive order, preserving the two columns
    as given. The logit transform clips probabilities to [1e-6, 1-1e-6].
    The returned tree uses the same feature conventions and probability floor
    as stacking_tree. Use identity node postprocessing for binary predictions.
    """
    _validate_transform(transform, complementary)
    roots = tuple(roots)
    if not roots or not all(isinstance(root, ValueNode) for root in roots):
        raise ValueError("Supply a nonempty sequence of ValueNode roots")
    if not complementary and len(roots) % 2:
        raise ValueError("Explicit channels require 2M roots in negative/positive order")

    y = _fit_array(labels)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or y.dtype.kind not in "biuf" or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("labels must be a vector with both binary classes 0 and 1")

    columns = []
    for root in roots:
        prediction = Tree(root.copy_subtree()).predict()
        column = _fit_array(prediction)
        if column.shape not in ((len(y),), (len(y), 1)):
            raise ValueError("Each root must return one probability per label, shaped (N,) or (N, 1)")
        if column.dtype.kind not in "biuf" or not np.all((column >= 0) & (column <= 1)):
            raise ValueError("Each root must return finite probabilities in [0, 1]")
        columns.append(column.ravel())
    x = np.column_stack(columns)
    # Keep float32 public feature rounding; promote half or integral inputs
    # so the logit clipping endpoints are representable.
    x = x.astype(np.result_type(x.dtype, np.float32), copy=False)
    if complementary:
        x = np.stack((1 - x, x), axis=2).reshape(len(y), -1)
    if transform == "logit":
        x = np.clip(x, 1e-6, 1 - 1e-6)
        x = np.log(x / (1 - x))
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, solver="lbfgs", max_iter=max_iter))
    pipeline.fit(x, y)
    return stacking_tree(roots, pipeline, transform=transform, complementary=complementary)
