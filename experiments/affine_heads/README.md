# A learned signed head for native OKAPI trees

An optional OKAPI tree reproduced both public logistic stackers on the same floating-point inputs. With a separate native fitting call that saw only roots and labels, maximum absolute prediction error on float64 was 5.56 × 10⁻¹⁶ for explicit class channels and 6.75 × 10⁻¹³ after folding binary complements. Adding three prespecified nonlinear subtrees to the fitted logit head improved TwoPathways IID Brier from .13364 to .04872 (paired difference −.08492, 95% CI −.08726 to −.08257). The same extension lowered fit Brier on CorrelatedRedundancy and worsened IID Brier in all 12 banks. [Primary preregistration](protocol.md) · [Separate native fitting check](independent-fit-protocol.md)

## Representation, fitting and input precision

LOGIT_MEAN can learn a common temperature and intercept. WEIGHTED_LOGIT_MEAN exposes weights but no intercept, initializes nonnegative weights and clips them to nonnegative values on mutation; manually supplying a negative coefficient is possible. The new BINARY_AFFINE operator has a separate signed coefficient for each ordered scalar input, an intercept, and a probability or clipped-logit feature transform. Its built-in structural key includes the complete settings. The default evolutionary palette is unchanged. Use identity node postprocessing when reproducing the binary stacker. [Node implementation](../../okapi/node.py) · [Transfer and native fit APIs](../../okapi/affine.py)

The public FusionBench fit standardizes *both* class channels for each provider in order negative, positive. With logits, each scalar input is individually clipped at 1e-6 before the transform. The C=1 logistic regression uses L2 on all standardized coefficients, unpenalized intercept, lbfgs and max_iter=2000. Given standardized weights w, center μ and scale s, the native affine weights are a=w/s and its intercept is b−a·μ. With complementary probability channels the compressed slope for each provider is a_positive−a_negative and the intercept additionally absorbs the sum of the negative-channel coefficients. With symmetric logits only the slope changes. Keeping both columns in the supervised fit preserves the original effective regularization. Fitting M positive channels at the same C would generally change the estimator.

Two public entry points separate conversion and supervised learning. The first uses a fitted public or equivalent sklearn pipeline. The second evaluates the copied roots, builds the original two-channel training design and fits a fresh scaler and logistic classifier with no reference parameters. Both return a native Tree which can be inferred and nested in OKAPI. A root can itself be a program that returns a binary probability.

```python
from okapi.affine import fit_stacking_tree, stacking_tree

copied = stacking_tree(positive_roots, public_stacker.model, transform="logit")
learned = fit_stacking_tree(positive_roots, fit_y, transform="logit", c_value=1.0)
explicit = fit_stacking_tree(class_channel_roots, fit_y, transform="logit", complementary=False)
```

Only the probabilities at prediction time are compared. The objective used while fitting is penalized logistic loss; held-out Brier measures a different quantity. The table gives the **largest row-level error** relative to independent predictions from the public stacker, over all four splits in both families after independent training.

| Source array and native representation | Probability features | Logit features |
| --- | ---: | ---: |
| float64, both original class channels | 5.55e−16 | 4.44e−16 |
| float64, one positive root per provider | 1.33e−15 | 6.74e−13 |
| float64, positive roots including nonlinear subtrees | 1.83e−15 | 5.38e−13 |
| float32, both original class channels | 1.19e−7 | 9.07e−8 |
| float32, one positive root per provider | 2.00e−7 | 4.79e−4 |
| float32, positive roots including nonlinear subtrees | 2.26e−7 | 1.72e−4 |

All preregistered float64 checks passed, including the full two-channel and compact tree on every split. Parameter transfer alone gave the same errors to the displayed precision. On float32, scaling the original design and folding its affine coefficients can round differently. Additionally, casting the two physical channels independently can break the assumed complement after taking logit near 0 or 1. Use the explicit channel representation when both original channels matter. Numerical agreement on float64 does not imply bitwise portability to float32. [Public fit transfer checks](results/transfer.csv) · [Independent native fits, errors and full-precision Brier](results/independent_fit.csv)

## What the nonlinear terms add

Four groups were specified in advance because TwoPathways has four latent factors. The root shape was also informed by this generator. Group membership was estimated solely from anonymous fit predictions. For every partition of the four group means into two pairs, form max(min(i,j),min(k,l)). All three possible pairings enter the regression beside all original providers. No pairing was chosen from the answers. The shape-matched control permutes assignments while preserving group sizes. Both feature sets receive the same number of coefficients, rows, regularization and fits.

Adding the three subtrees with zero weights reproduces the flat stack. Fitting the enlarged head can change all coefficients. On TwoPathways, its fit penalized log-loss objective fell from .3847 to .1576 on average. On CorrelatedRedundancy it fell from .2732 to .2706, yet IID logit-head Brier worsened by +.00076 [+.00036, +.00115]. The measured gain in TwoPathways concerns the supplied nonlinear basis and its fitted head. The inner structures were fixed. The ordinary evolutionary search had a different compute budget and no opportunity to learn this new node in its default palette. The fit does not certify an optimum for Brier, tree topology or nested parameters. [Independent discussion of constant fitting in GP](https://doi.org/10.1007/s10710-019-09371-3)

Mean positive-class Brier across 12 independent fresh banks per family is shown below. Public flat and native flat rows agree at the displayed precision for each transform. Validation is scored after freezing, with no model selection. IID and shift each contain 4,096 independent rows per bank.

| Family | Method | Fit, 384 | Validation, 256 | IID | Shift |
| --- | --- | ---: | ---: | ---: | ---: |
| TwoPathways | Mean | .15383 | .15166 | .15433 | .19305 |
| | Ranked-prefix Top-N | .15300 | .15366 | .15740 | .19502 |
| | Ordinary GP, 160 calls | .17541 | .18611 | .18875 | .21526 |
| | Public / native flat, probability | .11411 | .12077 | .12338 | .16757 |
| | Public / native flat, logit | .12363 | .13268 | .13364 | .17342 |
| | Shuffled nonlinear, probability | .10163 | .11057 | .11239 | .16658 |
| | Shuffled nonlinear, logit | .10657 | .11738 | .11756 | .16662 |
| | Grouped nonlinear, probability | .03860 | .04435 | .04745 | .13561 |
| | Grouped nonlinear, logit | .03939 | .04692 | .04872 | .13115 |
| CorrelatedRedundancy | Mean | .14460 | .14558 | .14528 | .12191 |
| | Ranked-prefix Top-N | .13629 | .14011 | .14034 | .12931 |
| | Ordinary GP, 160 calls | .13536 | .14632 | .14310 | .13598 |
| | Public / native flat, probability | .08395 | .08787 | .08939 | .07706 |
| | Public / native flat, logit | .08281 | .08675 | .08737 | .07336 |
| | Shuffled nonlinear, probability | .08240 | .08873 | .09049 | .08105 |
| | Shuffled nonlinear, logit | .08137 | .08735 | .08830 | .07807 |
| | Grouped nonlinear, probability | .08298 | .08879 | .09011 | .07768 |
| | Grouped nonlinear, logit | .08203 | .08748 | .08812 | .07502 |

On TwoPathways the grouped logit model also beat the equally large shuffled model on IID (−.06884 [−.07816, −.05952]) and its flat stack under detector weakening (−.04227 [−.04535, −.03919]). The raw-probability variant improved IID vs its own stack by −.07592 [−.07781, −.07404]. In the control family the grouped probability version instead increased IID loss by +.00072 [+.00015, +.00129]. All intervals use 12 independent bank differences and unadjusted Student intervals. The native search and shuffled models average starts 311/751 before analysis. Mean within-bank IID SD between the two starts for shuffled logit was .01035 on TwoPathways and .00058 on CorrelatedRedundancy, versus .00738 and .00912 for native GP. The inferred feature set and supervised solver are deterministic given inputs. [Scores, accuracy and log loss](results/scores.csv) · [Paired estimates](results/paired.csv)

## Budget and reproducibility

Each learned method saw the same 384 unique labeled fit rows. Top-N ranked all M providers by fit Brier and scored each equal-weight prefix on those same rows, with 16/20 total scans (6,144/7,680 row-scores) for TwoPathways/CorrelatedRedundancy. Mean selected N was 7.00/6.17, frozen on all four splits. The mean uses no labels. The grouped search computed three additional label-free fit feature columns (1,152 feature-row evaluations), then fitted one regression, with 22 or 26 standardized design columns vs 16 or 20 baseline columns. These feature computations can be shared between the probability and logit models; per-arm timings show the standalone cost. No additional validation, refinement or restart selection is used. The positive-only grouped trees contain on average 54.00/54.25 nodes, versus 9/11 for the compact flat programs. The more complex trees repeat input providers across their nested programs.

Grouped logit models averaged 14.33/16.92 solver iterations, versus 9.08/13.92 for flat logit, with construction of the three feature columns taking 1.2/1.5 ms and supervised fitting 2.5/3.4 ms. The independent end-to-end native float64 fitting call averaged 3.5 ms for a grouped logit head across both families. Ordinary GP made 160 full fitness evaluations (61,440 fit row-scores) per start, averaging 97/116 ms. The native learned grouped logit trees evaluated a 4,096-row IID batch in 1.26/.87 ms. These shared-CPU timings are descriptive; solver iterations and tree assessments are different operations. [Per-fit grouping, costs, complexity, coefficients and objectives](results/cost.csv)

A fresh process replayed both families on banks 31700 and 31711. All 224 score rows, 256 transfer checks and 48 fit records matched exactly outside timing columns. Use FusionBench `f074e85281c056e45585073073c93a0a921e88ff` on PYTHONPATH beside this checkout.

```sh
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.affine_heads.run --out /tmp/affine-heads
python -m experiments.affine_heads.independent_fit --archive /tmp/affine-heads/scores.csv --out /tmp/affine-heads/independent_fit.csv
```

[Primary benchmark](run.py) · [Independent fitting and precision check](independent_fit.py) · [FusionBench reference stackers](https://github.com/damiankucharski/FusionBench/blob/f074e85281c056e45585073073c93a0a921e88ff/fusionbench/methods/stacking.py)

Independent FusionBench validation also transferred fitted stacks on real WDBC out-of-fold provider predictions. Its maximum float64 discrepancy was 8.36e−13; the held-out Brier improvement from a single forest/LDA maximum feature over flat stacking was unresolved. This uses a different feature and data source. [Independent WDBC report](https://github.com/damiankucharski/FusionBench/blob/ee9b789/docs/ensemble-study/real-affine-head/README.md)

On the independently published calibrated-suppressor panel, both native tree forms preserved the negative weak-source coefficient in all 24 high-correlation fits. They also reproduced its deterioration when that correlation disappeared. [Matched transfer replay and source anchors](../affine_suppressor/README.md)
