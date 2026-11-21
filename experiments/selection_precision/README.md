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

For one maximized objective in [0,1], distance to the ideal point is `1 − f`. For a minimized objective it is `f`. In exact arithmetic, every strictly increasing transformation into that range preserves the one-dimensional order, equalities and Pareto comparisons with node count. With the same random draws and tie order it preserves selection. Floating-point replay additionally requires preserving the actual score and distance comparisons. For example, 0 and 10⁻²⁰ are distinguishable as float64 scores but their computed distances to 1 both equal 1. The fixture verifies that distinction. Meanwhile, the rounded form of `0.75 + 0.0001f` is 0.750 for every score in [0,1]. For multicriterion fitness, the Pareto domination relation remains invariant to coordinatewise increasing transformations, but distances in the final selection and tournaments can change even without rounding. The test includes a two-objective counterexample. These claims describe the current selection code and elementary algebra.

The two examples demonstrate a lost improvement in one decision. They provide no estimate of how often this happens in ordinary runs or how much it affects held-out performance. Removing ties may favor increments driven by noise in fit labels. The previous full-suite archive stores ordered trajectory digests and selected programs. It cannot reconstruct full counterfactual cohorts without replay. A prospective study at natural Brier scale needs equally funded native trajectories, raw fit and separately held-out scores, and the arithmetic mean, ranked-prefix Top-N and regularized logistic stack.

## Optional selection precision

`Okapi(..., selection_fitness_decimals=6)` keeps six decimal places for selection. `None` keeps the raw float64 objective values, and the default is the historical three places. Set the precision when constructing an engine. The initial population, the ordinary and reused-survivor update paths, correlation-based seeding and the receiving side of island migration use this policy. Native scoring stores objective values in a NumPy float64 matrix after tree inference. Tensor backends and user objectives can already have rounded earlier quantities. The raw scoring interface and the predictions returned by a particular frozen tree are independent of the selection setting. [OKAPI](../../okapi/okapi.py) · [Guided initialization](../../okapi/guided.py) · [Island migration](../../okapi/islands.py)

Three more fixtures qualify the interpretation.

* On equally weighted labels `(0,1,1)`, singletons `A=(0,½−2⁻¹²,1)` and `B=(0,1,½)` have raw fitness .91658527 and .91666667. A native tournament with A first picks A at three decimals and B at six or full precision. Extending either chosen parent with the same `MAX(parent,E)`, where `E=(0,0,1)`, yields two different outcomes. Starting from the same archive, the first choice leaves best Brier at 1/12, while the second finds a perfect fit in one extension. This example conditions on the same extension and proves reachability under a fixed proposal.
* On a single draw from a population with class probability ½, three equally sized possible parents predict `½−2⁻¹²`, `½` or `½+2⁻¹²`. Full precision always chooses the extreme matching the fit label, yielding conditional excess `2⁻²⁴`. With three-decimal selection they tie for either label. A full tournament with randomly ordered candidates yields lower expected excess, `(2/3)2⁻²⁴`. Thus exact optimization of the sampled objective has no general guarantee of better independent Brier.
* The native streaming mean of 0.1, 0.1 and 0.4 yields slightly different float64 values when the summands are reordered. Their `1 − Brier` on a single positive row differs by 2.22×10⁻¹⁶. Six decimal places preserve the .00035964 fitness improvement in the first section while treating these numerically distinct but mathematically identical means equally.

The three-row and single-label constructions were contributed by the FusionBench diagnostic team. We verified them against the native tournament or its uniform-tie distribution and verified the accumulation-order example using native trees. These are selection fixtures on deliberately small tasks. [Executable witnesses](test_scale.py)

## Historical replay

An independent verifier extracted the tree before the new option from Git at `630fd1f6a12d269c8bc8cd0ebbe0025e0421c302`. Separate processes ran that version, the current default, and explicit `selection_fitness_decimals=3`. On NumPy and CPU PyTorch, two seeds, with and without survivor score reuse, all 144 generation records matched in each version comparison. This includes 1,212 ordered raw assessments per version, the complete native RNG state, selected program structure, node ordering, mutation chances and selection fitness arrays. The common trace SHA-256 was `a58059775260641eaf48a8a72bcf558431a45aae4af20a5214f3353a9a1de8c3`. Reproduce in an OKAPI checkout with `python experiments/selection_precision/replay_default.py`. The fixture does not need FusionBench data. [Independent verifier](replay_default.py)
