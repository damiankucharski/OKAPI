# Correcting false duplicates in OKAPI tree search

On fresh standard GP runs, the previous printable key suppressed 918 distinct candidates among 45,959 joined parent/offspring proposals. Of these, 884 changed at least one prediction on 64 fit examples. The fixed search retained those programs. At 640 scored candidates, the primary structural-minus-legacy difference in TwoPathways IID Brier was −.00051 (paired 95% interval −.00686 to +.00583). These data confirm a correctness bug and leave its average effect on predictive quality unresolved. [Registered protocol](protocol.md) · [Production fix](https://github.com/damiankucharski/OKAPI/commit/86f21cdd580564f7e84bed9b326caad977ee2325)

## What the search actually discarded

The trees in the regression test start from input A and apply two operators in sequence. The first applies MIN(B,C) then MAX(D); the second applies MIN(B) then MAX(C,D). Both display as `VN[A]_MIN_MAX_VN[B]_VN[C]_VN[D]`. For A=B=D=0 and C=1 they output 0 and 1. The former production generation discarded whichever came second before fitness evaluation. Thresholds, weighted means and logit parameters can also collide under display rounding. The new key records ordered arities, exact built-in node types, typed provider IDs and full prediction parameters. The search and island receiver both use it. Unknown node types are conservatively retained. This identity assumes a shared mapping from provider IDs to input tensors. [Regression tests and actual generation and migration witnesses](../../test/test_structural_key.py)

The audit compares first occurrence under each key on each actual cohort. Every region of the final cohort is included, even when the fitness budget stops scoring part-way through it. Counts are candidate occurrences and can include a program proposed more than once across generations. The probe is label-free and uses only fit examples. Equal probe outputs alone cannot establish that two programs agree everywhere.

| Trajectory | Family | Proposals | Distinct candidates the display mask removes | Different topology or provider layout | Any prediction difference | Largest absolute difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Legacy | TwoPathways | 22,163 | 366 | 2 | 358 | .97016 |
| Legacy | CorrelatedRedundancy | 23,796 | 552 | 7 | 526 | .87280 |
| Structural | TwoPathways | 21,890 | 1,041 | 0 | 1,012 | .95815 |
| Structural | CorrelatedRedundancy | 24,366 | 2,565 | 17 | 2,329 | .87280 |

On legacy trajectories, 909 of 918 cases had the same ordered layout and different exact parameters. Seven of the nine cross-layout cases changed probe outputs. All 48 legacy trajectories contained at least one changed probability. The structural trajectory counts are counterfactual applications of the display mask to that policy's own cohorts. Its larger count reflects the changed population and repeated near-duplicates. Neither policy produced a reversal in which the display mask retained a program that the structural key would suppress. [All candidate batches](results/batches.csv) · [Every collision and its maximum, average and number of changed probe predictions](results/collisions.csv)

## Quality at matched budgets

Each family has 12 independent banks (29200–29211) and two fixed search starts per bank. Each method sees the same anonymous provider columns. GP, the public ranked-prefix Top-N (`metric="brier"`) and C=1 logistic regression on provider logits share exactly 384 labeled fit rows. Top-N ranks individual providers, scans every equal-weight prefix, then freezes its selection. There are 8 providers on TwoPathways and 10 on CorrelatedRedundancy, hence 16 and 20 fit-score scans per bank respectively. The mean uses no labels. Search freezes the best full-precision fit Brier seen at 160 and 640 actual objective calls, counting parent rescoring. There is no validation selection or refinement. Each policy is then scored once on 256 validation, 4,096 independent IID and 4,096 shifted rows. Lower positive-class Brier is better.

| Family | Method | Fit | Validation | IID | Shift |
| --- | --- | ---: | ---: | ---: | ---: |
| TwoPathways | Mean | .15575 | .15465 | .15516 | .19321 |
| | Top-N | .15520 | .15730 | .15757 | .19519 |
| | Logit stacking | .12535 | .13219 | .13262 | .17517 |
| | Legacy 160 | .17755 | .18948 | .18776 | .21399 |
| | Structural 160 | .17696 | .18689 | .18685 | .21371 |
| | Legacy 640 | .15697 | .16528 | .16646 | .20340 |
| | Structural 640 | .15656 | .16536 | .16595 | .20306 |
| CorrelatedRedundancy | Mean | .14321 | .14033 | .14563 | .12200 |
| | Top-N | .13189 | .13730 | .13964 | .13180 |
| | Logit stacking | .07588 | .08605 | .08692 | .07417 |
| | Legacy 160 | .12840 | .14014 | .14093 | .13489 |
| | Structural 160 | .12956 | .13916 | .14176 | .13563 |
| | Legacy 640 | .11341 | .12525 | .12723 | .12270 |
| | Structural 640 | .11787 | .12645 | .13004 | .12622 |

At 640 evaluations the CorrelatedRedundancy IID difference was +.00281 [−.00256, +.00817]. The shift differences were −.00034 [−.00313, +.00246] for TwoPathways and +.00352 [−.00216, +.00919] for CorrelatedRedundancy. Intervals are unadjusted Student intervals over independent bank differences after averaging starts. The mean within-bank IID standard deviation across the two optimizer starts at 640 calls was .00871 legacy vs .00656 structural on TwoPathways and .01178 vs .00826 on CorrelatedRedundancy. Mean winning node counts were 12.88 vs 9.92 and 12.83 vs 8.83 respectively. [All scores, complexity, starts, selected N and timings](results/scores.csv) · [All paired Brier comparisons](results/paired.csv)

At 640 calls each GP used 245,760 fit row-scores. Mean fit/search time was .589 s legacy vs .550 s structural on TwoPathways and .518 s vs .461 s on CorrelatedRedundancy. These timings include input file preparation and normal deduplication. Paired diagnostic overhead was timed separately and averaged .027 s vs .051 s and .031 s vs .084 s. Top-N fit averaged .010 s on both families; stacking averaged .003 s. The timing is descriptive of a single shared CPU environment and changing trajectories. The post-fit inference and diagnostic times for every model are in the score file.

## Reproduction and independent integration study

Run with OKAPI production fix `86f21cd` and public FusionBench `f074e85281c056e45585073073c93a0a921e88ff`. The driver installs either the original display mask or the production key at the same native entry point. A second process on banks 29200 and 29211 reproduced all 176 score rows (excluding timings), 509 joined batches and 666 collision rows exactly. A separate unpatched process imported every OKAPI module from the pre-fix archive at `7da9e4b27164c9002033b26c98aab65e99790b72` and called the original native search on those two banks. Both metrics for the 16 frozen models on four splits matched the legacy-policy CSV exactly (128 of 128 values); incumbent call, node count and generation also agreed in all 64 rows. Use the lab checkout containing the [driver](run.py) and set FUSIONBENCH to the pinned FusionBench checkout.

```sh
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.structural_key.run --out /tmp/key-audit
```

The independent FusionBench integration test compares both immutable production versions under a restricted MIN/MAX search. Its exact min-cut oracle measures the optimum attainable in that specified family on an eight-provider panel. The general-purpose palette in this audit has no such certificate. [Independent preregistered FusionBench comparison](https://github.com/damiankucharski/FusionBench/blob/346c2df/docs/ensemble-study/search-regret/structural-key-protocol.md)
