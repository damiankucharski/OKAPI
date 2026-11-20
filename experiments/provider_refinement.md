# Local provider reassignment after GP search

A single local move improved on the frozen TwoPathways tree. Under the same additional fit and validation budget, the comparison with continued GP was inconclusive. On the prespecified IID TwoPathways endpoint, local reassignment reduced Brier by 0.00237 relative to ordinary continued GP, with a paired 95% interval from −0.00492 to +0.00018 for local minus continued. Allowing continued GP to reuse survivor scores reduced the difference to −0.00023 (−0.00372 to +0.00326).

## What the move changes

A value node identifies a provider and may itself contain further operations. Replace the provider at exactly one occurrence, including the root if sampled. All operators, numerical weights, sibling order and tree size stay fixed. This can introduce a previously unused provider. Score one private copy repeatedly on fit data and retain the best strict improvement. A capped neighborhood samples provider/occurrence pairs without replacement. For a full neighborhood the canonical order is breadth-first node order followed by input mapping order.

```python
from okapi.refinement import refine_provider_assignments

step = refine_provider_assignments(winner, fit_predictions_by_id, fit_labels,
                                   loss, max_evaluations=32, seed=17)
# Both trees can now be scored on the same independent validation set.
score = step.tree.do_pred_on_another_tensors(current_tensors=validation_predictions_by_id)
```

`loss(predictions, labels)` returns one finite scalar to minimize. Every occurrence is first rebound to the supplied fit mapping. The result carries the new tree, initial and final fit losses, and actual number of full fit evaluations, including the incumbent.

## Held-out results

Positive-class Brier, averaged across 12 independent data draws and two GP starts per draw. Lower values are better. Each row uses independent fit, validation and test samples. IID and shift share the selected trees. The logit stacker uses the full fit and validation label budget.

| Family and domain | Frozen tree | Local reassignment | Continued GP | Continued GP with reuse | Mean | Logit stack |
|---|---:|---:|---:|---:|---:|---:|
| TwoPathways, IID | .17236 | .16815 | .17052 | .16838 | .15558 | .13182 |
| TwoPathways, detector accuracy .90 → .78 | .20590 | .20393 | .20484 | .20414 | .19297 | .17413 |
| CorrelatedRedundancy, IID | .13688 | .13481 | .13642 | .13598 | .14418 | .08550 |
| CorrelatedRedundancy, block correlation .85 → .10 | .13442 | .13407 | .13418 | .13365 | .12263 | .07282 |

Local reassignment beat the frozen tree by .00420 on IID TwoPathways (paired 95% interval for the improvement .00153–.00688) and by .00197 on its shift (.00074–.00321). Its comparisons with either continued-GP policy included zero in every domain. Under the correlation shift, the arithmetic mean overtook all four GP policies. The logit stacker achieved the best Brier in each of the four comparisons. The local policy retained a fit-selected change after validation in 13/24 TwoPathways runs and 11/24 CorrelatedRedundancy runs.

## Costs and evaluation boundaries

Eight initial generations averaged 220.42 and 211.13 full fit evaluations on TwoPathways and CorrelatedRedundancy. The three challenger policies used exactly the same allowance within every paired run, averaging 28.75 and 29.46 additional evaluations, including a fresh incumbent score. Each fit evaluation scores all 256 examples. Each policy then scores two programs on the same 256 validation examples. The controls may use the best program from any evaluated prefix of a partially affordable generation. The frozen comparator spends zero additional evaluations. The mean uses no labels. The stacker uses all 512 labels and no validation selection. Model inputs are simulated provider predictions. The cost of training upstream providers is outside this comparison.

| Family | Local extra search | Ordinary GP extra search | Reuse GP extra search | Fit + validation for local / ordinary / reuse |
|---|---:|---:|---:|---:|
| TwoPathways | 1.09 ms | 17.44 ms | 15.68 ms | 49.12 / 65.43 / 63.69 ms |
| CorrelatedRedundancy | 1.73 ms | 13.79 ms | 22.13 ms | 59.69 / 71.75 / 80.13 ms |

Durations are in-process elapsed times on shared CPU. They include materializing the fit files, initial GP search, additional search, and final validation. Generating simulated data, importing libraries and forking a checkpoint for the paired comparison are outside the timer. Local refinement ran first in each cell, so these timings are descriptive. The per-run table includes inference latency on 4,096 examples separately. These shallow synthetic trees and small fit sets limit conclusions about production latency.

The two initial GP seeds share data within each bank. The paired intervals average both runs before using the 12 independent data draws as units. Only the IID TwoPathways ordinary-GP contrast was designated primary. All other intervals are descriptive and have no multiplicity correction. The driver draws the two test domains only after all policies have frozen their choices. It uses separate streams for fit, validation, IID, and shift. The two search starts and the local neighborhood seed are independent of the held-out labels. Fresh-process replays of data seeds 20200 and 20211 reproduced all 96 non-timing result rows exactly.

## Reproduction

Use this branch and FusionBench revision `4cf4810e9876f51837b6eba1c4d7aa4bf1c4dac7`. The simulation and adapter files at that revision match those used here. The recorded environment used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0 and CPU PyTorch 2.15.0. Set the path to the FusionBench checkout in `FUSIONBENCH` and run these commands from the OKAPI worktree.

```sh
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python experiments/provider_refinement.py --out /tmp/provider_refinement.csv
```

[Preregistered protocol](algorithm-lab.md) · [Driver](provider_refinement.py) · [Per-seed outcomes, latency and cost](results/provider_refinement.csv) · [Paired differences](results/provider_refinement_contrasts.csv)
