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

## Separate breeding from incumbent replacement

A generation can propose children using its training population and then compare a fixed set of offers with the current deployment tree on fresh confirmation labels. This lets a child that fails to improve on the incumbent remain available as a parent when the breeding policy allows it.

`model.propose_offspring(count)` uses the current `model.population` and its aligned `model.fitnesses` from training. The method calls the native tournaments, crossover and structural and parameter mutation. It returns exactly one final product per requested slot, including duplicate programs and unchanged copies. Each entry contains the tree and the two indices drawn from the breeding population. A one-tree population can self. The recorded indices identify a mating. A crossover can copy a parent without introducing material from the other. The caller controls which parents and children survive to breed.

```python
from okapi.replacement import select_brier_replacement

# The training population already has its selection fitnesses. The incumbent
# is a frozen native Tree. Offer every slot, including duplicates.
offers = model.propose_offspring(12)
inc_prediction = incumbent.do_pred_on_another_tensors(current_tensors=block_inputs)
child_predictions = [
    offer.tree.do_pred_on_another_tensors(current_tensors=block_inputs)
    for offer in offers
]
decision = select_brier_replacement(
    inc_prediction, child_predictions, block_labels, alpha=0.05 / 6, rule="kl"
)
if decision.index is not None:
    incumbent = offers[decision.index].tree
```

In this example, prespecify six looks and use a fresh independent confirmation block for each look. Freeze the incumbent and all 12 offers before opening the block's labels. Each slot is charged one twelfth of that look's error allowance. The result's `eligible` tuple lists all candidates with strictly positive lower bounds. A strict breeding policy can admit those children. An open breeding policy can consider all children using training fitness. Keep evaluation of the deployed incumbent on a separate test set.

The selector accepts integer labels and one probability matrix per tree. For binary data a vector or single positive-class column is also supported. `gains`, `lower_bounds` and `margins` are in full Brier units, summing squared errors over all classes. The largest positive bound wins, with earlier slots breaking positive ties. A tie with the incumbent keeps it. Use `rule="raw"` for empirical gain alone and `rule="hoeffding"` for a bound based on the sum of squared row ranges.

The KL rule computes the smallest and largest possible gain on each row by considering every class label while holding both predictions fixed. Subtract each row's minimum. Rows with positive range are called active. Divide their offsets by the **largest** active-row range, giving independent bounded observations between zero and one. Their common bound permits different agreement probabilities across rows. At any positive exponent, convexity bounds each observation's exponential moment by the moment for a Bernoulli variable with the same mean. The arithmetic–geometric mean inequality then bounds the product by the moment of a binomial with the average mean. Inverting this one-sided KL-Chernoff bound at the allocated error level gives a lower bound on the mean normalized offset. Scaling back and restoring all row minima gives the lower bound on the challenger's fixed-design expected Brier gain.

For independent subjects with multiple dependent rows each, pass `groups=subject_ids`. The selector averages gains and endpoints within each subject and weights subjects equally. This coverage concerns new outcomes at the given provider inputs. An estimate of performance on new subjects or a shifted population requires separate data.
