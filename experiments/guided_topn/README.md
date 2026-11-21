# Retrospective ranked-prefix averaging baseline

This audit adds the public FusionBench Top-N baseline to the original correlation-guided initialization study and its frozen transfer to two other tasks. Its settings were registered after the search results were available. It reuses the original four splits and the published native, guided and shuffled GP winners at 160 and 640 full fit calls. GP was not rerun. [Review plan](protocol.md) · [Original guided study](../guided/README.md) · [Transfer study](../guided_transfer/README.md)

On TwoPathways, guided GP at 160 fit calls reached IID Brier .06715 versus .15720 for Top-N, .15504 for the equal mean and .13398 for stacking. On CorrelatedRedundancy, Top-N (.14021) and guided GP (.14123) were close at that budget, while stacking (.08835) was substantially better. On WeakestLink the stacker also beat the early guided method, both IID and after shift. On Parity the guided rules outperformed Top-N but their Brier remained close to the class-prevalence reference (.25063 IID). The added comparison does not change the difference between detecting useful structure and obtaining calibrated probabilities.

## Method and data matching

Each bank provided 384 labeled fit rows for Top-N, logistic stacking and the GP arms. The equal mean used no labels. The public Top-N implementation stably sorts individual providers by increasing fit Brier, then scores cumulative equal-weight averages of prefixes 1 through M. It chooses the shortest best-scoring prefix, with the original anonymous column order breaking individual ranking ties. It sees both classes. Binary class-summed Brier in its stored prefix scores is twice the positive-class Brier below. Scaling by two preserves the ranking. Across all 48 fits, the public selected length matched the first minimum of its stored prefix scores and that minimum reproduced the reported fit loss after conversion. Ranking and prefix length are learned on the same fit rows; validation selects zero settings. The C=1 stacker is trained on all providers' logits. The original shuffled and guided arms use the same shape library and differ in their provider assignments.

Every baseline was fit for every bank before any validation or test draw. All 384 replayed arithmetic-mean and stacker records agreed exactly with their published Brier **and** accuracy (4 families × 12 banks × 4 splits × 2 methods). Each GP cell uses the mean of the two published optimizer starts; each baseline fits once per bank. The 256 validation and separate 4,096-row IID and shifted tests use their original streams and anonymized provider ordering. [Replay checks](results/archive_checks.csv) · [Baseline scores and timings](results/controls.csv)

## Matched scores

Brier entries are positive-class losses, with smaller values preferred. Accuracy uses the original threshold strictly greater than .5. The final two columns report test accuracy. The 160 and 640 labels refer to full 384-row tree evaluations.

| TwoPathways | Fit Brier | Val Brier | IID Brier | Shift Brier | IID acc. | Shift acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-provider mean | 0.15263 | 0.15437 | 0.15504 | 0.19250 | 0.80334 | 0.74864 |
| Top-N | 0.15160 | 0.15687 | 0.15720 | 0.19391 | 0.79604 | 0.73820 |
| Logit stacker | 0.12120 | 0.13259 | 0.13398 | 0.17598 | 0.80528 | 0.73645 |
| Native 160 | 0.17348 | 0.18365 | 0.18521 | 0.21306 | 0.71373 | 0.65981 |
| Shuffled 160 | 0.14037 | 0.14516 | 0.14700 | 0.18963 | 0.79095 | 0.72501 |
| Guided 160 | 0.06950 | 0.06854 | 0.06715 | 0.15423 | 0.94956 | 0.81775 |
| Native 640 | 0.15331 | 0.16205 | 0.16449 | 0.20204 | 0.76395 | 0.69025 |
| Shuffled 640 | 0.11191 | 0.12369 | 0.12124 | 0.18260 | 0.83345 | 0.73469 |
| Guided 640 | 0.06891 | 0.07020 | 0.06865 | 0.15532 | 0.94302 | 0.81267 |

| CorrelatedRedundancy | Fit Brier | Val Brier | IID Brier | Shift Brier | IID acc. | Shift acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-provider mean | 0.14773 | 0.14443 | 0.14460 | 0.12167 | 0.80349 | 0.95461 |
| Top-N | 0.13785 | 0.13818 | 0.14021 | 0.13209 | 0.83128 | 0.87773 |
| Logit stacker | 0.08275 | 0.08555 | 0.08835 | 0.07554 | 0.87773 | 0.89699 |
| Native 160 | 0.13986 | 0.14143 | 0.14294 | 0.13831 | 0.81229 | 0.82531 |
| Shuffled 160 | 0.14318 | 0.14231 | 0.14435 | 0.12829 | 0.80812 | 0.89299 |
| Guided 160 | 0.13884 | 0.13754 | 0.14123 | 0.12656 | 0.81194 | 0.87825 |
| Native 640 | 0.12849 | 0.12957 | 0.13207 | 0.12651 | 0.84290 | 0.85729 |
| Shuffled 640 | 0.12950 | 0.12971 | 0.13279 | 0.12872 | 0.85062 | 0.86928 |
| Guided 640 | 0.12946 | 0.13062 | 0.13273 | 0.12855 | 0.84212 | 0.86400 |

| WeakestLink | Fit Brier | Val Brier | IID Brier | Shift Brier | IID acc. | Shift acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-provider mean | 0.24392 | 0.24379 | 0.24185 | 0.24332 | 0.56429 | 0.55859 |
| Top-N | 0.24166 | 0.24805 | 0.24581 | 0.24616 | 0.56278 | 0.55595 |
| Logit stacker | 0.03007 | 0.04178 | 0.03968 | 0.05482 | 0.94600 | 0.93933 |
| Native 160 | 0.11994 | 0.13213 | 0.12829 | 0.14085 | 0.80109 | 0.79231 |
| Shuffled 160 | 0.06754 | 0.07173 | 0.06982 | 0.09485 | 0.92887 | 0.91425 |
| Guided 160 | 0.03946 | 0.04368 | 0.04141 | 0.07801 | 0.96425 | 0.93659 |
| Native 640 | 0.04725 | 0.05330 | 0.05152 | 0.07229 | 0.92888 | 0.91381 |
| Shuffled 640 | 0.02963 | 0.03685 | 0.03585 | 0.05509 | 0.95322 | 0.94042 |
| Guided 640 | 0.02909 | 0.03573 | 0.03549 | 0.05617 | 0.95513 | 0.94148 |

| Parity | Fit Brier | Val Brier | IID Brier | Shift Brier | IID acc. | Shift acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-provider mean | 0.32555 | 0.34140 | 0.32980 | 0.27981 | 0.50269 | 0.50035 |
| Top-N | 0.32496 | 0.34357 | 0.33126 | 0.28175 | 0.50216 | 0.49923 |
| Logit stacker | 0.24610 | 0.25399 | 0.25407 | 0.25136 | 0.52407 | 0.50623 |
| Native 160 | 0.25930 | 0.28006 | 0.27463 | 0.28617 | 0.67455 | 0.57821 |
| Shuffled 160 | 0.26783 | 0.29051 | 0.27714 | 0.28227 | 0.69868 | 0.60343 |
| Guided 160 | 0.24672 | 0.26461 | 0.25185 | 0.25015 | 0.73598 | 0.65786 |
| Native 640 | 0.24138 | 0.26492 | 0.25840 | 0.27250 | 0.68219 | 0.58514 |
| Shuffled 640 | 0.25523 | 0.27803 | 0.26563 | 0.27045 | 0.70985 | 0.61662 |
| Guided 640 | 0.24406 | 0.26228 | 0.25028 | 0.25159 | 0.73016 | 0.64846 |

The paired differences below subtract Top-N Brier from guided Brier. Each unadjusted 95% Student interval uses 12 independent bank differences within its family. These comparisons are retrospective. The full table includes every control, both metrics and all four splits. [Paired estimates](results/paired.csv)

| Family | Calls | IID difference (95% interval) | Shift difference (95% interval) |
| --- | ---: | --- | --- |
| TwoPathways | 160 | −.09005 (−.09175, −.08834) | −.03968 (−.04142, −.03795) |
| TwoPathways | 640 | −.08854 (−.09038, −.08670) | −.03859 (−.04063, −.03656) |
| CorrelatedRedundancy | 160 | +.00102 (−.00383, +.00587) | −.00553 (−.01232, +.00126) |
| CorrelatedRedundancy | 640 | −.00748 (−.01314, −.00182) | −.00354 (−.00982, +.00274) |
| WeakestLink | 160 | −.20440 (−.20650, −.20230) | −.16815 (−.17007, −.16623) |
| WeakestLink | 640 | −.21031 (−.21261, −.20802) | −.18999 (−.19303, −.18694) |
| Parity | 160 | −.07941 (−.08219, −.07663) | −.03160 (−.03354, −.02966) |
| Parity | 640 | −.08098 (−.08549, −.07647) | −.03016 (−.03194, −.02837) |

## Selected prefix and cost

Top-N scores M individual providers and M cumulative prefixes per bank, each on 384 rows. A prefix scan incrementally adds one provider. Using the same labels for these 2M assessments costs 384 unique labels. No candidate uses held-out labels. Inference averages the selected N providers. The durations below are means from one run on a shared CPU pinned to logical core 0, with numerical-library threads limited to one. They exclude source generation. Inference timing covers the 4,096 IID examples.

| Family | M | Selected size × number of banks | Fit scans, rows scored | Mean fit ms, Top-N / stack | Mean inference ms, mean / Top-N / stack |
| --- | ---: | --- | --- | --- | --- |
| TwoPathways | 8 | 5×1, 7×6, 8×5 | 16, 6,144 | 17.58 / 5.26 | .06 / .18 / 2.45 |
| CorrelatedRedundancy | 10 | 2×4, 3×4, 4×1, 5×1, 10×2 | 20, 7,680 | 17.52 / 6.18 | .39 / .24 / 2.28 |
| WeakestLink | 8 | 4×1, 5×1, 6×4, 7×4, 8×2 | 16, 6,144 | 11.72 / 3.46 | .05 / .18 / 1.31 |
| Parity | 6 | 4×3, 5×2, 6×7 | 12, 4,608 | 7.81 / 2.60 | .04 / .22 / 1.18 |

Top-N improved on the mean on CorrelatedRedundancy IID (.14021 versus .14460), but its shift loss was higher (.13209 versus .12167) when the correlation among redundant providers decreased. On the other three families, prefix selection slightly lowered fit Brier and raised both IID and shift Brier relative to retaining everyone. [Per-bank provider rankings, prefix scores and selections](results/selection.csv)

## Reproduction

Use the OKAPI algorithm-lab checkout containing this driver, the two archived studies, and FusionBench commit `f074e85281c056e45585073073c93a0a921e88ff`. The run imported the public baseline from an isolated archive of this exact release. A second process reproduced every non-timing field in the 576 baseline score records, 48 fit records, 384 archive checks and 288 paired estimates. Its inputs came from the original experiment's data function and its dependency versions are listed in the [first study](../guided/README.md). From the OKAPI checkout, with `FUSIONBENCH` pointing at that pinned release, run the following command.

```sh
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m experiments.guided_topn.run --out /tmp/guided-topn
```

[Audit driver](run.py) · [Public Top-N implementation](https://github.com/damiankucharski/FusionBench/blob/f074e85281c056e45585073073c93a0a921e88ff/fusionbench/methods/averaging.py)
