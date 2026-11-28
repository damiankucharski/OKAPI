# Core API

This section documents the core components of OKAPI.

## Okapi Class

The `Okapi` class is the main entry point for evolutionary model ensemble optimization.

```python
from okapi import Okapi
```

::: okapi.okapi.Okapi
    options:
      show_root_heading: false
      show_source: true

## Tree Class

The `Tree` class represents a computational tree structure for model ensemble composition.

```python
from okapi.tree import Tree
```

::: okapi.tree.Tree
    options:
      show_root_heading: false
      show_source: true

## Penalize uncertain improvements during search

Some programs beat the provider mean on a few unusually favorable examples. OKAPI can charge each program for the variability of its improvement while it is evolving. The training labels already used to score the population supply this estimate.

On each row, compute the arithmetic mean's loss minus the program's loss. Positive values favor the program. Average these paired differences and subtract `paired_z` times their sample standard deviation divided by the square root of the number of independent units. The arithmetic mean scores zero. A program with a constant positive gain keeps its full gain. A program with a variable gain receives a larger deduction.

This example uses single-column positive-class probabilities and binary labels. The loss callback returns one value per row.

```python
import numpy as np
from okapi import Okapi

def brier_rows(prediction, labels):
    return (prediction[:, 0] - labels.reshape(-1)) ** 2

model = Okapi(
    preds_source="fit/providers",
    gt_path="fit/labels.npy",
    population_size=4,
    population_multiplier=2,
    tournament_size=2,
    backend="numpy",
    paired_row_loss=brier_rows,
    paired_z=1.0,
    selection_fitness_decimals=None,
)
model.train(iterations=20)
```

The installed objective, `model.paired_fitness_(prediction, labels)`, returns a raw score in the loss's units. The solver maps this scalar to `0.5 + arctan(score) / pi` for its unit-interval proximity calculation. The map is fixed across generations. Set `paired_z=0` for a matched empirical-gain control. Omitting `paired_row_loss` runs ordinary OKAPI.

When rows share a subject, flight, or acquisition session, pass the unit IDs as `paired_groups`. The objective first averages within each group and gives each group equal weight. With one independent unit and a positive penalty, the objective assigns every program zero. Its standard error characterizes sampling variability of each fixed program. Searching many programs can still produce optimistic scores. Estimate the selected tree's performance on unused data.

To use the same objective for every trial of a parameterized native head or feature screen, bind a separate instance to each data split. The baseline prediction and candidate predictions must use the same row order, labels and native probability convention. Construct the baseline under the same task postprocessing used to evaluate the trees. Valid probability means must be preserved by that postprocessing.

```python
from okapi.node import ValueNode
from okapi.paired_fitness import PairedGainFitness, arithmetic_mean_tree
from okapi.pattern_logit import fit_pattern_tree

inputs = model.train_tensors
labels = model.gt_tensor
baseline = arithmetic_mean_tree(inputs).predict()
objective = PairedGainFitness(baseline, labels, brier_rows, z=1.0)
roots = [ValueNode(None, tensor, key) for key, tensor in inputs.items()]
tree, info = fit_pattern_tree(roots, labels, objective=objective)
```

## Select a frozen tree on fresh paired validation rows

For a family fixed using separate fit data, `model.select_tree_against_mean(validation_tensors, labels, row_loss)` compares the learned Pareto frontier with the native arithmetic mean. The return value contains a single fit-bound tree and a decision record. For candidates from multiple searches, call `okapi.mean_guard.select_tree_against_mean(candidates, fit_tensors, validation_tensors, labels, row_loss)`. For precomputed row losses use `okapi.mean_guard.select_against_mean(mean_losses, candidate_losses)`, with one row of the candidate matrix per program. The decision's `index` is the challenger position or `None` for the mean.

The selector subtracts each challenger's mean from its paired gains. For each draw it multiplies the centered gains by a random sign per independent unit, shared across programs, sums them and divides by the square root of `U × (U − 1)` for `U` independent units. It then takes the maximum across all challengers and zero. Sharing signs retains correlations among programs, including exact copies. The challenger with the greatest mean gain must strictly exceed the upper empirical 95th percentile of those maxima. Ties among challengers follow the supplied order. The defaults use 4,095 draws and seed zero. With fewer than eight independent units the selector keeps the mean. Set `groups` to aggregate dependent rows before the calculation. Specify `alpha`, `draws`, `seed`, and `min_units` before inspecting the validation results.

This multiplier approximation assumes independent units and finite-variance losses. Few units or highly skewed contrasts can make its threshold unreliable. Generating additional programs from the same validation data introduces selection that the threshold does not account for. Score the chosen tree on a separate test set.
