# Fit signed native logit weights

Use `fit_pattern_tree` when a native fusion tree should learn how strongly to use each provider or evolved program. It searches signed weights and a binary intercept by evaluating a higher-is-better scalar objective. Supply provider probabilities on the fitting examples and a separate bank of predictions for deployment. A discontinuous objective is allowed. Its numeric scale determines the strength of any penalty.

## Fit and predict

```python
import numpy as np

from okapi.node import ValueNode
from okapi.pattern_logit import fit_pattern_tree

fit_inputs = {
    "model_a": np.array([[0.12], [0.80], [0.28], [0.86], [0.65], [0.15]]),
    "model_b": np.array([[0.70], [0.38], [0.50], [0.22], [0.42], [0.88]]),
}
labels = np.array([0, 1, 0, 1, 1, 0])
roots = [ValueNode(None, column, name) for name, column in fit_inputs.items()]


def score(probability, target):
    return -np.mean((probability[:, 0] - target) ** 2)


tree, fit = fit_pattern_tree(roots, labels, objective=score, l2_strength=0.25)
fresh = {
    "model_a": np.array([[0.78], [0.19]]),
    "model_b": np.array([[0.31], [0.72]]),
}
prediction = tree.do_pred_on_another_tensors(current_tensors=fresh)
assert prediction.shape == (2, 1)
```

The fitted tree contains copies of the supplied roots. Deployment needs probabilities from the providers referenced in that tree, identified by the same keys. `fit["score"]` is the raw fit score. `fit["evaluations"]` counts objective calls. Positive `l2_strength` adds `penalty` and `penalized_score` to the diagnostics. Its default is zero. The optional penalty scales each squared slope by the fit-set variance of that input's log evidence and divides the weighted sum by the number of fit examples. The binary intercept is unpenalized.

## Include evolved programs

`select_pattern_features` screens an ordered archive on the discovery sample. Supply pairs of native trees and their output matrices on those rows, the original provider predictions, the flat tree output and the discovery labels. It skips outputs equivalent to a provider or an earlier eligible program and retains up to three programs that improve the objective. Archive order resolves exact ties. Each selected program can be passed as another root to the same fitter.

```python
from okapi.node import MinNode
from okapi.pattern_features import select_pattern_features
from okapi.tree import Tree

left, right = (root.copy_subtree() for root in roots)
left.add_child(MinNode([right]))
candidate = Tree(left)
archive = [(candidate, candidate.predict())]
selected, search = select_pattern_features(
    archive, fit_inputs.values(), tree.predict(), labels,
    objective=score, report_indices=True,
)
augmented, fit_augmented = fit_pattern_tree(
    roots + [program.root for program in selected],
    labels, objective=score, l2_strength=0.25,
)
```

The screening weights are trial values. The second fit estimates the final joint weights. Screening on the discovery split and choosing between fitted candidate families on a separate validation split keep these two decisions distinct.

## Multiclass and backends

For binary data, use one positive-class probability per row shaped `(N, 1)` and identity postprocessing. For multiclass data, supply normalized `(N, C)` matrices, integer class labels and `multiclass=True`. The multiclass head uses one signed weight per input, shared across classes. The fitter accepts native NumPy or PyTorch tensors for the active backend. Half-precision log inputs are promoted before taking logarithms. The standard multiclass normalization assigns uniform probabilities to a zero-mass row, which can arise when taking minima of disjoint provider outputs. The screening function expects NumPy probability matrices. Use the standard multiclass normalization scope for subsequent predictions, including reloaded trees.

```python
from okapi.functions import scale_vector_to_sum_1
from okapi.globals import use_postprocessing_function

# Given a fitted multiclass_tree and the corresponding new provider matrices:
with use_postprocessing_function(scale_vector_to_sum_1):
    multiclass_prediction = multiclass_tree.do_pred_on_another_tensors(
        current_tensors=multiclass_inputs
    )
```

The functions are opt-in. The full experimental family selector, its splits and its comparisons to arithmetic mean, Top-N and logistic stacking are documented in the [research note](../research/ensemble-fusion/index.md).
