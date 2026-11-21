# Where does ordinary OKAPI lose to stacking?

**Stacking was already ahead on the training rows.** With 384 labels, L2-regularized logistic stacking on logits had lower training Brier than a 48-generation fixed-seed OKAPI search in every one of 12 independent banks in each family. Most of the difference on new IID examples was already visible on fit. On TwoPathways we can also exhibit a much better tree using the same operators as the search.

## All four splits at 384 fit labels

Each cell is mean positive-class Brier over the 12 new banks. Lower is better. A single fixed GP start supplies both checkpoints. The two-start policy spends additional search and uses validation to choose a champion. Every fitted competitor uses the same 384 fit rows. The stacker uses C=1 and fits once per bank. The mean has no fitted parameters.

| Family | Method | Fit | Validation | Fresh IID | Shift |
|---|---|---:|---:|---:|---:|
| TwoPathways | Arithmetic mean | .15456 | .15742 | .15451 | .19238 |
| | Logistic stack | .12487 | .13398 | .13275 | .17543 |
| | GP, 8 generations | .17961 | .19478 | .19658 | .21960 |
| | GP, 48 generations | .15121 | .15988 | .16071 | .20009 |
| | GP, selected from two starts at 48 | .14383 | .15399 | .15482 | .19830 |
| | Source-informed feasible tree | .06816 | .06912 | .06744 | .15495 |
| CorrelatedRedundancy | Arithmetic mean | .14358 | .14520 | .14379 | .12148 |
| | Logistic stack | .07288 | .09221 | .08751 | .07934 |
| | GP, 8 generations | .13145 | .14652 | .14203 | .13826 |
| | GP, 48 generations | .10349 | .12406 | .11992 | .11449 |
| | GP, selected from two starts at 48 | .09214 | .11199 | .10908 | .10323 |
| | Source-informed feasible tree | .12824 | .13214 | .13019 | .12541 |

The held-out loss gap decomposes exactly into the gap on fit and the difference between the two methods' IID-minus-fit changes. On TwoPathways, GP minus stack was **+.02634** on fit (paired 95% interval +.01975 to +.03293) and **+.02795** on IID (+.02184 to +.03407). The differential change was **+.00161** (−.00369 to +.00692). On CorrelatedRedundancy the gaps were **+.03061** on fit (+.01752 to +.04370) and **+.03241** on IID (+.01952 to +.04531), with a differential change of **+.00180** (−.00452 to +.00813). Both models showed some training optimism. The data support a fit deficit as the main contributor to their difference at this budget.

The arithmetic mean tells a different story in the two families. On TwoPathways the 48-generation GP slightly beats mean on fit (.15121 vs .15456) and loses on IID (.16071 vs .15451). On CorrelatedRedundancy GP beats the mean on both splits. Conclusions about GP depend on which comparator is being explained.

## The same contrast across fit sizes

| Family | Fit rows | GP48 − stack on fit | GP48 − stack on fresh IID |
|---|---:|---:|---:|
| TwoPathways | 96 | +.02708 | +.02659 |
| | 384 | +.02634 | +.02795 |
| | 1,536 | +.03031 | +.03151 |
| CorrelatedRedundancy | 96 | +.03506 | +.03523 |
| | 384 | +.03061 | +.03241 |
| | 1,536 | +.03546 | +.03664 |

Training size increased sixteenfold across the sweep and the loss difference persisted. Fit sizes use nested examples and therefore are dependent within each bank. [All four split means for every size and method](results/means.csv) · [Paired intervals](results/paired.csv)

## Search versus the operator set

* **More search helped.** Going from 8 to 48 generations lowered IID Brier at N=384 in all 12 banks in each family. The mean reductions were .03587 (95% interval .02578–.04596) and .02211 (.01277–.03146) on TwoPathways and CorrelatedRedundancy respectively. The mean also improved on fit by .02840 and .02796. Choosing a GP48 champion across two starts required two validation scores and approximately doubled fitness calls. Its average IID improvement over the first start alone was .00588 and .01085, with both intervals including zero.
* **TwoPathways has a reachable solution that search missed.** The source-informed 15-node tree averages each pair of detectors, takes the minimum within each of the two pathways, then takes the maximum across pathways. Its IID Brier was .06744, a reduction of .09326 relative to ordinary GP48 (95% interval .08708–.09945). It also beat stacking under the frozen shift. This tree uses the simulator's provider grouping and task structure. It demonstrates search headroom in the stock grammar. Discovering that grouping from labeled rows is part of the problem faced by ordinary OKAPI.
* **The dependency-informed arithmetic tree did not repeat that success.** Its source weights account for correlated noise but average probabilities. It gave .13019 on IID CorrelatedRedundancy, compared with .11992 for GP48 and .08751 for stacking. This generator's posterior combines logit evidence using its covariance. The stacker can learn coefficients and extrapolate beyond all provider probabilities on a row.
* **One representation limit is measurable.** With a single probability channel, identity postprocessing, and this stock grammar, every tree output lies between the smallest and largest input probabilities on every row. The min, max and threshold operators select in-range values, and both mean operators use convex combinations. Induction gives the bound for any depth. With true posterior q and interval [a,b], the lowest possible conditional excess Brier over all interval-valued functions is the squared distance from q to [a,b]. On IID CorrelatedRedundancy its measured mean was .00257, against .03741 excess for GP48; the stacker went outside the interval on 67.9% of rows. On IID TwoPathways the floor was .000015, against .12328 excess for GP48. The floor is a relaxation over every interval-valued function. The remaining loss can reflect the smaller tree grammar, finite-sample estimation and optimization. Optional logit operators or other postprocessors would require a separate bound.

## Accounting and evaluation boundaries

Ordinary GP uses the six stock operators, one initial singleton per provider, native crossover, mutation and rounded fitness, and size as a Pareto objective. It uses one Brier objective, offspring multiplier two and tournament size two. Search has no tree cap and no warm start. On TwoPathways the average number of full fit evaluations at 8 and 48 generations was 116 and 886; on CorrelatedRedundancy it was 142 and 1,178. Two-start selection used 1,750 and 2,370 fit calls respectively, plus two full 256-row validation scores. At N=384 the one-start fit times were 0.044 and 0.487 s on TwoPathways (8/48), or 0.050 and 0.693 s on CorrelatedRedundancy. The matched stacker fit took 0.003 and 0.005 s. Every GP fitness call scored all fit rows. The stacker instead optimized regularized log loss and was evaluated on Brier. Diagnostic scoring costs for each split are recorded separately. CPU timings are descriptive.

The study used independent RNG streams for fit, validation, IID and shift. Model choices at all fit sizes were frozen before opening test inputs and their generator posteriors. Validation loss for the two-start policy is a selection score. The other GP policies and stacker do not select on validation. Both GP starts are fixed functions of each fresh bank seed, and the first start is the prespecified comparison. Intervals are paired t intervals over 12 independent data/optimizer banks. The four primary fit/IID contrasts have no multiplicity correction. The detector accuracy shift is .90→.78, the correlated-block shift .85→.10. These two synthetic families do not establish general behavior across other tasks. A fresh process reproduced all 336 non-timing measurement rows for banks 24100 and 24111 exactly.

## Reproduce

Use FusionBench `4cf4810e9876f51837b6eba1c4d7aa4bf1c4dac7` and this OKAPI branch. Set `FUSIONBENCH` to the pinned checkout and run from the OKAPI repository.

```sh
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python experiments/fit_gap/run.py --out /tmp/fit-gap
```

[Preregistered protocol](protocol.md) · [Driver](run.py) · [Per-bank measurements](results/measurements.csv) · [All means](results/means.csv) · [Paired contrasts](results/paired.csv)
