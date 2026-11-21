# Does training optimism explain the loss to stacking?

This retrospective analysis replays the published single-population arm on the original 12 confirmation seeds in each of two synthetic scenarios. The original logistic stack fitted its coefficients on 640 labels. OKAPI searched on 384 and used the other 256 to select one of four finalists. Here both logistic stackers fit on exactly the 384 search rows with fixed regularization (C = 1). The baseline arithmetic mean has no fitted parameters. All methods see identical expert predictions in each split. The binary Brier scores use the positive-class probability, and lower values indicate better predictions.

| Scenario | Method | Fit, 384 | Selection, 256 | Fresh IID, 4,096 | Shifted, 4,096 |
|---|---|---:|---:|---:|---:|
| TwoPathways | OKAPI | .14234 | .14809 | .14692 | .19160 |
| | Arithmetic mean | .15557 | .15651 | .15519 | .19272 |
| | Logistic on probabilities, 384 | .11702 | .12846 | .12302 | .16833 |
| | Logistic on logits, 384 | .12698 | .13807 | .13337 | .17556 |
| | Logistic on logits, 640 | .12881 | .13198 | .13233 | .17406 |
| CorrelatedRedundancy | OKAPI | .12295 | .12232 | .12388 | .12258 |
| | Arithmetic mean | .14907 | .14213 | .14326 | .12163 |
| | Logistic on probabilities, 384 | .08278 | .08668 | .08804 | .08059 |
| | Logistic on logits, 384 | .08078 | .08471 | .08629 | .07663 |
| | Logistic on logits, 640 | .08283 | .07857 | .08485 | .07154 |

The selection column is diagnostic. OKAPI used it to choose a finalist, while both the fit and selection columns include training examples for the 640-label stack. Only the final two columns are unseen by every method. The shift weakens the detectors in TwoPathways and lowers error correlation in CorrelatedRedundancy. Expert outputs are simulated directly.

### Which part of the deficit appears on new data?

The 384-row logit stack beat OKAPI on fit and on fresh IID test in every paired seed in both scenarios. OKAPI's mean excess over the stack on TwoPathways was .01536 on fit and .01356 on IID. On CorrelatedRedundancy it was .04217 on fit and .03759 on IID. In both cases most of the observed deficit was already present before testing.

For each seed we subtracted the fit deficit from the fresh-IID deficit. The mean differences and paired 95% t intervals across 12 seeds were −.00180 [−.00535, +.00174] and −.00457 [−.00917, +.00002], respectively. The apparent training optimism of OKAPI itself (IID minus fit) was +.00458 [+.00156, +.00761] and +.00093 [−.00293, +.00479]. Some optimism is compatible with these data, particularly on TwoPathways. There is no observed increase in OKAPI's average disadvantage relative to the stack when moving from fit to IID. These intervals are descriptive, with scenarios and contrasts chosen after seeing the original results.

The replay cannot attribute the fit deficit to a particular cause. OKAPI searched trees of at most 15 nodes for 480 objective calls and selected on Brier. Logistic regression minimized regularized log loss. Operators, optimization and the relationship between objectives could all affect fit. A search-budget sweep and a positive control with a known, representable fusion rule would help separate these explanations.

### Reproduction

Use an environment with both research repositories on the import path. From the OKAPI checkout run `python experiments/islands/replay_baselines.py`. The script checks all 144 overlapping test scores against the published mean, single-population and 640-label stack columns to absolute tolerance 1e−12 before writing the output. A second full execution (11.17 seconds elapsed on CPU, including startup) reproduced the output byte for byte. [Original protocol](protocol.md) · [Per-seed observations](results/fit-test-replay.csv)
