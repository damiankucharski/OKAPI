# Transferring signed stackers onto the calibrated suppressor panel

Both native OKAPI tree forms reproduced their fitted stackers across all
four splits of the published FusionBench panel. Among **4,902,912 row-level
comparisons**, the largest difference in positive-class probability was
**4.55 × 10⁻¹⁵**, below the registered 10⁻⁹ threshold. The largest tree Brier
discrepancy against a published split loss was 5.55 × 10⁻¹⁷. This reuses the
original samples and estimates the numerical error of transferring fitted
parameters. [Source study](https://github.com/damiankucharski/FusionBench/tree/62afd107e9ba0bfa0052186acb0e9f687d574822/docs/ensemble-study/calibrated-suppressor) · [Transfer protocol](protocol.md)

| Source stacker | Native input slots | Maximum probability error against refit | Maximum Brier error against archive |
| --- | ---: | ---: | ---: |
| Probability | 4 explicit | 5.55 × 10⁻¹⁶ | 2.78 × 10⁻¹⁷ |
| Probability | 2 folded | 9.99 × 10⁻¹⁶ | 5.55 × 10⁻¹⁷ |
| Logit | 4 explicit | 5.55 × 10⁻¹⁶ | 5.55 × 10⁻¹⁷ |
| Logit | 2 folded | 4.55 × 10⁻¹⁵ | 5.55 × 10⁻¹⁷ |

The explicit form uses both original class columns for both providers. The
folded form uses their two positive-class columns. This generator evaluates
both channel sigmoids separately in float64, so complement and logit
antisymmetry hold to floating-point precision. Both forms call `Tree.predict()`
on copies of the same frozen tree rebound to each split. An independent call
to the original `run_bank` reproduced **all 2,016 archived Brier losses
exactly**, including both privileged references. All Top-N choices and
archived effective logit slopes also matched. The archive supplies split-level losses.
Row-level comparisons use public stackers refitted on the original rows.

## The negative coefficient survives transfer and changes the shift trade-off

The experiment fits anonymous columns. After freezing the trees, undoing
the source permutation recovers the effective coefficient for the weaker
provider. In the explicit form this is the positive-channel coefficient
minus the negative-channel coefficient. Both native forms returned the
source coefficient exactly in every bank.

| Fit correlation | Mean weak logit slope | Negative in banks | Mean weak probability slope | Negative in banks |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.00423 | 0/24 | 4.86966 | 0/24 |
| 0.50 | 0.04915 | 9/24 | 0.50764 | 7/24 |
| 0.85 | −1.91965 | 24/24 | −7.97801 | 24/24 |

At fit correlation 0.85, the native signed logit tree beat the nonnegative
logit fit in every IID bank. The mean difference in Brier was **−0.02823**
(paired 95% t interval −0.02897 to −0.02749). When correlation fell to zero,
the signed tree scored worse in every bank, with a difference of **+0.09209**
(0.08662 to 0.09756). Both forms returned the same reported estimates.

| Method trained at ρ = 0.85 | Fit Brier | Validation Brier | IID Brier | Shift Brier |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic mean | 0.14210 | 0.14113 | 0.14241 | 0.12275 |
| Ranked-prefix Top-N | 0.11176 | 0.11051 | 0.11233 | 0.11254 |
| Logit stacker and its native tree | 0.08349 | 0.08373 | 0.08481 | 0.20536 |
| Probability stacker and its native tree | 0.09094 | 0.09155 | 0.09234 | 0.17673 |
| Nonnegative logit fit | 0.11113 | 0.11148 | 0.11304 | 0.11327 |

Top-N ranked and selected its prefix on fit Brier and chose one provider
in all 24 banks at ρ = 0.85. The logit stacker uses C=1 and the source
nonnegative fit constrains the effective signs using the same class-channel
design and ridge penalty. The source study's mechanism, an advantage at
high correlation that reverses under this shift, is inherited by the tree.
The transfer procedure alone provides no improvement over its source fit.

## Cost and reproducibility

The 24 seeds 41000–41023 each supply three source correlations. For each
condition the two public stackers see 384 fit labels, then their frozen
models are converted into four trees. Conversion adds zero fitting calls
and zero validation choices. The 256 validation rows and two independent
8,192-row test sets are used for scoring. The `source_fit_seconds` column
is repeated for both forms of the same fit. The separate archive check
reruns the five original fits. Its costs are recorded in `source_fits.csv`.

| Source and form | Fit (ms) | Transfer (ms) | Rebind 8,192 rows (ms) | Tree prediction (ms) | Source prediction (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Probability, explicit | 3.43 | 0.16 | 0.12 | 0.29 | 1.45 |
| Probability, folded | 3.43 | 0.10 | 0.09 | 0.21 | 1.45 |
| Logit, explicit | 3.40 | 0.15 | 0.13 | 0.48 | 1.62 |
| Logit, folded | 3.40 | 0.10 | 0.09 | 0.34 | 1.62 |

Times are means from a single-thread CPU run. Rebinding includes the tree
copy. Native predictions call the NumPy OKAPI backend. The source emits both
class probabilities via its scikit-learn pipeline. Source prediction time is
repeated across its two transfer forms.

From the OKAPI checkout, set `FUSIONBENCH` to the pinned source checkout,
including its archived results.

```bash
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m experiments.affine_suppressor.run --out /tmp/suppressor-transfer
python -m experiments.affine_suppressor.run --out /tmp/suppressor-replay --seeds 41023
```

A fresh-process replay of seed 41023 at all three correlations reproduced
all non-timing fields in 48 transfer scores, 12 exported trees, 84 source
scores and 15 source fits.

[Driver](run.py) · [Native scores and timing](results/scores.csv) · [Exported coefficients and costs](results/heads.csv) · [Replayed source scores](results/source_scores.csv) · [Source fits vs archive](results/source_fits.csv) · [Paired intervals](results/paired.csv)
