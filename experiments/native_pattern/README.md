# Fit native logit weights with a scalar objective

The optional `okapi.pattern_logit.fit_pattern_tree` fits a signed logit combination of ordered provider values or complete native subprograms by evaluating a user-supplied score. Training tries coordinate moves and temperature changes and accepts strict score improvements. Prediction executes the returned native `Tree`. The routine fits no external classifier and requests only scalar objective values. A discontinuous objective is allowed; a long plateau can leave the initial solution unchanged.

For binary NumPy inputs, given an `(n_rows, n_providers)` array of positive-class probabilities and an integer label vector:

```python
import numpy as np
from okapi.globals import set_backend, use_postprocessing_function
from okapi.node import ValueNode
from okapi.pattern_logit import fit_pattern_tree

set_backend("numpy")
roots = [ValueNode(None, scores[:, i:i+1], f"m{i}") for i in range(scores.shape[1])]
def skill(pred, y):
    return 1.0 - np.mean((pred[:, 0] - y) ** 2)

tree, info = fit_pattern_tree(roots, labels, objective=skill)
new_inputs = {f"m{i}": held_out[:, i:i+1] for i in range(held_out.shape[1])}
with use_postprocessing_function(np.asarray):
    proba = tree.do_pred_on_another_tensors(current_tensors=new_inputs)
```

The fitted binary node uses a signed weight per incoming logit and an intercept. In multiclass mode pass `(n_rows, n_classes)` per provider, set `multiclass=True`, and use `okapi.functions.scale_vector_to_sum_1` as the tree's postprocessing function at inference. The multiclass node uses one weight per provider across all class log probabilities, followed by normalization. The optimizer caches the inputs to the local node, so each trial costs a scalar objective evaluation on all fit rows. Its `evaluations` counter includes starting configurations and native-order checkpoint scoring. Supply independent evaluation data to estimate generalization.

The [registered 19-family FusionBench experiment](https://github.com/damiankucharski/fusionbench/blob/41f73c6/docs/ensemble-study/native-pattern/protocol.md) evaluates this primitive alongside native GP discovery, with both public logistic stackers as external comparisons.

In the completed experiment one validation-selected native tree scored 0.18388 macro-average full-class Brier on fresh IID data, across 19 scenarios and 12 independent banks per scenario. A flat native node scored 0.21891, the logistic stacker on base probabilities 0.21698, and the logistic stacker on base logits 0.22017. On nine predefined distribution shifts the selected tree scored 0.32062 and probability stacking scored 0.30544. Lower is better. The native subtree-screening and parameter-search functions are part of this OKAPI branch. The full benchmark training recipe and independent replay are in [FusionBench](https://github.com/damiankucharski/fusionbench/blob/daf1455/docs/ensemble-study/native-pattern/README.md).

## Can validation perturbations choose a more robust tree?

`okapi.transfer_selection.select_transfer_candidate` accepts clean and perturbed validation fitness values and returns one candidate. `shrink_subtree_weights` copies a fitted logit head, scales only the appended subtree coefficients and preserves each base-provider coefficient. On a frozen FusionBench grid across 19 scenarios, eight banks per scenario, three development label-noise levels and six shared deployment disturbances, this variant scored 0.30487 full-class Brier. The previous clean-validation policy scored 0.30376. The paired difference was +0.00112 (95% interval −0.00026 to +0.00249). This experimental selector missed its improvement criterion. Use the previous policy as the reference when testing further changes. [Protocol, results and independent replay](https://github.com/damiankucharski/fusionbench/blob/05fa4ec/docs/ensemble-study/transfer-grid/README.md)
