# Operators API

This section documents the operator aliases provided by OKAPI for easier usage.

Operator aliases provide convenient access to different operator node types.

```python
from okapi.operators import MEAN, MIN, MAX, WEIGHTED_MEAN
```

::: okapi.operators
    options:
      show_source: true

## Opt-in multiclass affine softmax

A multiclass affine node accepts one probability vector per provider or fitted subtree. It produces one signed affine score per output class and applies softmax. In `diagonal` mode, score (c) uses each input's probability for class (c). In `full` mode, the score for class (c) can use every input class. Both modes include a class-specific bias. Inputs and output use shape `(rows, classes)` with the same class order and at least two classes.

| Mode | Coefficient array | Score for output class c |
|---|---|---|
| `diagonal` | `W[provider, class]` | `bias[c] + sum_m p[m,c] * W[m,c]` |
| `full` | `W[provider, input_class, output_class]` | `bias[c] + sum_(m,i) p[m,i] * W[m,i,c]` |

### Fit on one split and predict on another

The example uses two raw provider matrices. To compose the head with a discovered program, include its fit-bound `evolved_tree.root` in the list passed to the fitter. Each root supplies its complete prediction, including any sequential operators. The fitter copies the roots and attaches the head to the first one. Its current prediction occupies slot zero and the remaining roots become the operator's children.

```python
import numpy as np

from okapi.functions import scale_vector_to_sum_1
from okapi.globals import use_postprocessing_function
from okapi.multiclass_affine import fit_multiclass_affine_tree
from okapi.node import ValueNode

# fit_a and fit_b are independently produced provider scores on fusion-fit rows.
roots = [ValueNode(None, fit_a, "a"), ValueNode(None, fit_b, "b")]

def fitness(probabilities, labels):
    target = np.eye(probabilities.shape[1])[labels]
    return 1.0 - 0.5 * np.square(probabilities - target).sum(axis=1).mean()

tree, info = fit_multiclass_affine_tree(
    roots, fit_labels, objective=fitness, mode="full",
    l2_strength=0.25, max_evaluations=2400,
)
with use_postprocessing_function(scale_vector_to_sum_1):
    probabilities = tree.do_pred_on_another_tensors(
        current_tensors={"a": test_a, "b": test_b}
    )
```

Replay requires every original provider ID used anywhere in the tree and the same class order. Supply scores for the new rows. The multiclass scope also normalizes the output of each nested native operator, as it did during fitting. The head itself applies a numerically stable float64 softmax. `Tree.save_tree_architecture` and `Tree.load_tree_architecture` persist the fitted structure and parameters for subsequent rebinding.

The fitter maximizes the supplied deterministic scalar score. It tests a uniform baseline and common positive slopes, then polls positive and negative coordinate changes on fit-centered, variance-scaled inputs. During search it subtracts `(l2_strength / n_fit) * sum(Var_fit(p[m,i]) * W[m,i,c]**2)` over the stored coefficients. Biases have no penalty. The deployed tree stores all coefficients and intercepts in the original probability coordinates. The default strength is zero. The default call cap is 2,400, including initialization and final native replay. The `info` dictionary returns the actual evaluation count, raw score, penalty and penalized score. High-dimensional full heads may exhaust the budget during a sweep. This bounded search can also optimize discontinuous scores, with no global optimality guarantee.

To replay prescribed coefficients directly, append `MulticlassAffineNode(children, W, bias, mode="full")` to the first `ValueNode` and construct a `Tree` from that root. As a standalone check, `MulticlassAffineNode(None, W, bias, mode="full").op(x)` accepts a tensor in `(providers, rows, classes)` order and returns the softmax. Manual parameters, copying, structural keys, and architecture persistence support both modes. The display label includes a short fingerprint of the weights and biases. Tree deduplication uses their exact bytes along with mode and shape.

Stock GP uses its existing operator palette. Including `okapi.operators.MULTICLASS_AFFINE` in a custom `allowed_ops` palette enables random diagonal initialization, which requires a bound child to infer the class count. A full head enters via explicit construction or the fitter. Parameter mutations preserve the chosen mode. Adding or removing a child adds or removes its coefficient block; random structural edits can change a fitted program's predictions. In a research comparison, record the method used to produce subtrees, the objective-call budget, the split used to fit the coefficients, and the held-out evaluation separately.
