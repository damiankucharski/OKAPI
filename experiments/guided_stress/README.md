# Which providers share a concept, and which just share a measurement error?

Pearson clustering found the four true concept pairs in every clean-noise bank, then missed all four pairs in every bank with strong correlated measurement error. A fourth-order statistic on the same anonymous fit columns recovered the correct partition in **12/12** banks in both conditions. At 1,536 fit rows and error correlation .90, IID Brier was **.15143** for the fourth-order initializer and **.19770** for the Pearson initializer. The paired difference was −.04628 (95% interval −.04795 to −.04460). Both searched the same 135 tree structures and used the same labels to select one tree.

## Mechanism and simulation

Each provider observes one of four independent balanced binary concepts. There are two providers per concept. TwoPathways labels are positive if both concepts in either path are positive. Each provider's standardized measurement takes the form

```text
Tᵢ = s Cg(i) + √ρ Hh(i) + √(1−ρ) εᵢ
```

The balanced sign C is the concept, H is shared Gaussian noise in a separate randomly assigned pair, and ε is independent Gaussian noise. The nuisance matching never pairs sensors of the same concept. The providers report sigmoid(2sTᵢ) with s equal to the normal quantile at .78. Their marginal sign accuracy and calibration for their respective concepts therefore stay the same as ρ changes. Each bank independently scrambles the columns. Only the simulator and the diagnostic oracle receive the group assignments.

When ρ=.90, it exceeds the variance of the repeated signal (s²≈.60), making the wrong pair more correlated than the true pair on the logit scale. The fourth joint cumulant provides a different measurement. For standardized centered logits X and Y the score is 1 + 2 E[XY]² − E[X²Y²], the negative of their mixed fourth cumulant. A balanced sign has fourth cumulant −2. Joint Gaussian noise has zero fourth cumulants, even when correlated. Independence makes mixed fourth cumulants additive across the independent signal and noise sources. The population score is therefore positive for two sensors of one concept and zero for sensors observing distinct concepts. Empirical fourth moments still carry sampling error.

All learned methods ran average linkage and searched the same bounded library of mean/min/max trees. The oracle separately supplied the true grouping and selected among 120 four-group trees using fit labels. Learners received neither source grouping nor labels during clustering. The fit, validation and test rows are independent. All model and template choices were frozen before evaluating the test domains.

## Results at N=1,536

Positive-class Brier averaged over 12 fresh banks. The shifted evaluation switches ρ between 0 and .90 without refitting the tree. Topology recovery refers to the four-group cut in the source data.

| Source | Grouping or comparator | Fit | Validation | IID | Switched ρ | Exact group recovery |
|---|---|---:|---:|---:|---:|---:|
| Independent noise | Pearson, probabilities | .15323 | .15398 | .15516 | .15296 | 12/12 |
| | Fourth cumulant, logits | .15323 | .15398 | .15516 | .15296 | 12/12 |
| | Pearson, logits | .15323 | .15398 | .15516 | .15296 | 12/12 |
| | Random assignment | .18792 | .18934 | .19024 | .19390 | 0/12 |
| | True-group library | .15323 | .15398 | .15516 | .15296 | 12/12 supplied |
| | Arithmetic mean | .19281 | .19341 | .19370 | .19989 | — |
| | Logit stacker | .16383 | .16815 | .16847 | .19957 | — |
| Shared noise .90 | Pearson, probabilities | .19619 | .19805 | .19770 | .19043 | 0/12 |
| | Fourth cumulant, logits | .15171 | .15069 | .15143 | .15404 | 12/12 |
| | Pearson, logits | .19619 | .19805 | .19770 | .19043 | 0/12 |
| | Random assignment | .18929 | .19400 | .19209 | .19066 | 1/12 |
| | True-group library | .15171 | .15069 | .15143 | .15404 | 12/12 supplied |
| | Arithmetic mean | .19865 | .20007 | .19864 | .19298 | — |
| | Logit stacker | .18755 | .19277 | .19075 | .17546 | — |

In the high-noise setting at N=384, the fourth-order score again recovered all 12 partitions. Its IID advantage over Pearson was .04958 (95% interval .04498–.05418). In the clean-noise source the two learned rules selected equivalent functions and their paired loss difference was zero at both fit sizes. Under high shared noise, switching from probabilities to logits while retaining Pearson correlation did not restore the correct partition. Source Pearson's mean adjusted Rand index was −.167 at ρ=.90; the fourth-order score's was 1.0.

Each learned library spent 135 assessments on all fit rows, or 51,840 and 207,360 candidate-example evaluations at the two sizes. Each retained one fit-selected winner, with zero validation choices and no GP generations. The oracle control spent 120 assessments. The arithmetic mean used zero fit labels, and logistic stacking (C=1) used only the fit labels. At N=1,536 and ρ=.90, mean grouping plus construction, fit scoring and selection times were about 16 ms for Pearson and 17 ms for the fourth-order method on the shared CPU. The raw records give grouping time and inference time separately. Both fit sizes use nested rows, so they are separate conditions rather than independent replications. The 12 data banks are the units for paired intervals.

This mechanism relies on non-Gaussian latent concepts and Gaussian measurement noise on the logit scale. A shared non-Gaussian nuisance can itself contribute a fourth cumulant; a Gaussian latent factor supplies no such signature. The result concerns initialization by a fixed library on this simulator. It is insufficient to replace the ordinary correlation metric generally.

## Reproduce

Use the same pinned FusionBench checkout and numerical environment as in the [first guided-search study](../guided/README.md).

```sh
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python experiments/guided_stress/run.py --out /tmp/guided-stress
```

A separate process reproduced all 224 non-timing result rows for banks 25300 and 25311. [Registered protocol](protocol.md) · [Driver](run.py) · [Per-bank losses, recovery and costs](results/runs.csv) · [Paired differences](results/paired.csv)
