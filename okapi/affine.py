"""Transfer a fitted binary probability or logit stacker onto OKAPI roots."""

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from okapi.node import BinaryAffineNode, ValueNode
from okapi.tree import Tree


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
    if transform not in ("probability", "logit"):
        raise ValueError("transform must be 'probability' or 'logit'")
    if not isinstance(complementary, (bool, np.bool_)):
        raise ValueError("complementary must be a boolean")
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
