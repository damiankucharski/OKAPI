# Fitness rounding and one-objective selection

A candidate with a lower binary Brier loss can lose its chance to reproduce because OKAPI rounds the objective before its population selection. This effect occurs with ordinary `1 − Brier` on a two-candidate fixture. A strictly increasing rescaling can make even a large difference disappear. The fixture freezes offspring, labels and predictions to isolate a single selection decision. A separate check calls the ordinary tournament with both candidates always present. [Executable examples](test_scale.py)

## Direct witnesses

Use balanced binary labels, with one short singleton assigning confidence 0.7 to the correct class on every row. The candidate averages the singleton and one other provider, so its tree has three nodes. The table shows positive-channel Brier. Survival keeps one program using ordinary Pareto-plus-proximity selection with node count. Two implementations of native fitness accounting, stock rescoring and unchanged-survivor reuse, give the following same answers.

| Other provider's confidence | Singleton loss | Candidate loss | Selection on raw `1 − loss` | Current native selection on `1 − loss` | Current native selection on `0.75 + 0.0001(1 − loss)` |
|---:|---:|---:|---|---|---|
| 0.9000 | 0.09000000 | 0.04000000 | mean | mean | singleton |
| 0.7012 | 0.09000000 | 0.08964036 | mean | singleton | singleton |

In the second row the raw fitnesses 0.91 and 0.91035964 both round to 0.910. The singleton has fewer nodes and dominates the mean in the rounded population. In the first row the compressed scores both round to 0.750. The same score table reaches native crossover tournaments. In 64 two-parent tournaments per row, every comparison on raw scores chose the mean. Rounded compressed scores admitted both candidates across the fixed seeds. [Initial, uncached and cached score paths](../../okapi/okapi.py) · [Survivor selection](../../okapi/population.py) · [Tournament selection](../../okapi/crossover.py)

## What follows from the selection rules

For one maximized objective in [0,1], distance to the ideal point is `1 − f`. For a minimized objective it is `f`. Any strictly increasing transformation preserving that range and all floating-point distinctions therefore preserves the one-dimensional order, equalities and Pareto comparisons with node count. With the same random draws and tie order this preserves raw-score selection. In contrast, the rounded form of `0.75 + 0.0001f` is 0.750 for every score in [0,1]. For multicriterion fitness, the Pareto domination relation remains invariant to coordinatewise increasing transformations, but distances in the final selection and tournaments can change even without rounding. The test includes a two-objective counterexample. These claims describe the current selection code and elementary algebra.

The two examples demonstrate a lost improvement in one decision. They provide no estimate of how often this happens in ordinary runs or how much it affects held-out performance. Removing ties may favor increments driven by noise in fit labels. The previous full-suite archive stores ordered trajectory digests and selected programs. It cannot reconstruct full counterfactual cohorts without replay. A prospective study at natural Brier scale needs equally funded native trajectories, raw fit and separately held-out scores, and the arithmetic mean, ranked-prefix Top-N and regularized logistic stack.
