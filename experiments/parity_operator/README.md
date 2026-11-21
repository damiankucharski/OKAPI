# An opt-in probabilistic XOR recovers Parity on anonymous providers

At eight fit-scored trees, substituting probabilistic XOR for a logit-mean candidate reduced IID Brier from .25085 to .01135 on FusionBench Parity (paired difference −.23950, 95% interval −.24291 to −.23608). All 12 independent banks recovered the true detector groups from anonymous fit outputs and selected the nine-node tree with within-group logit sums and outer XOR. Its predictions agreed with the generator’s analytic posterior to within 9 × 10⁻¹⁴ across all four splits. The same search on CorrelatedRedundancy never selected XOR. [Preregistered protocol](protocol.md) · [Posterior replay using only frozen choices](results/posterior.csv)

## Search and interpretation

The optional operator combines two positive-class probabilities as p(1−q)+(1−p)q. For larger arities it tracks odd and even probability mass over the inputs. The probabilistic interpretation requires independent Bernoulli events. A repeated observation of the same realized event violates that assumption. Opt in using `from okapi.operators import PROBABILISTIC_XOR` in an explicit operator list or tree. It supports single positive-probability channels and preserves the native default palette. [Operator implementation](../../okapi/node.py) · [NumPy and Torch regression tests](../../test/test_probabilistic_xor.py)

Each eight-tree library crosses two within-group pools (arithmetic mean or clipped logit sum) with four root operators. Both libraries have MIN, MAX and mean. Their fourth root is XOR or the logit mean at unit temperature. One arm estimates two groups by average linkage on fit-prediction correlations. A separate arm permutes anonymous provider assignments while preserving group sizes; both libraries see identical permutations on a given start. All 432 matched fit assessments for the three shared roots agree exactly between the two libraries. A library selects the first candidate with lowest fit Brier. Labels never guide grouping. [All 1,152 assessed candidates](results/candidates.csv)

The generator’s known two-bit XOR structure motivated the operator and the decision to cut at exactly two groups. In particular, multiplying evidence from conditionally independent, calibrated detectors within each group matches how this family was generated. Finding the grouping from predictions and choosing the candidate on labels are parts of the experiment. This is a positive result under those conditions; it leaves open behavior when the dependency structure or number of factors differs. It measures a restricted template search, while the ordinary evolutionary search below uses its existing palette and a larger budget.

## Paired results

There are 12 fresh banks (30400–30411) per family, with 384 fit rows, 256 validation, and 4,096 independent IID and shifted examples in each bank. Candidate selection, learned references and the native GP use the same 384 fit labels; the arithmetic mean uses none. The deterministic grouped policies fit once per bank. Shuffled libraries and native GP use starts 311 and 751. All policies are frozen before generating any held-out splits. The primary metric is positive-class Brier (lower is better). Accuracy uses a strict .5 threshold. [Per-bank scores and accuracy on all splits](results/runs.csv)

| Family | Method | Fit | Validation | IID | Shift |
| --- | --- | ---: | ---: | ---: | ---: |
| Parity | All-provider mean | .33435 | .33715 | .32968 | .27972 |
| | Ranked-prefix Top-N | .33308 | .34025 | .33191 | .28239 |
| | Logit stacker | .24534 | .25266 | .25435 | .25172 |
| | Native GP, 160 calls | .27684 | .28752 | .27866 | .28629 |
| | Shuffled, substitute | .30561 | .31333 | .30065 | .29183 |
| | Shuffled, XOR available | .10919 | .10792 | .11023 | .18403 |
| | Grouped, substitute | .25620 | .26327 | .25085 | .25810 |
| | Grouped, XOR available | .01189 | .01388 | .01135 | .11569 |
| CorrelatedRedundancy | All-provider mean | .14592 | .14661 | .14468 | .12150 |
| | Ranked-prefix Top-N | .13577 | .13902 | .13912 | .12996 |
| | Logit stacker | .08309 | .08733 | .08704 | .07792 |
| | Native GP, 160 calls | .13774 | .14215 | .14173 | .13750 |
| | Shuffled, substitute | .14107 | .14212 | .14208 | .08513 |
| | Shuffled, XOR available | .14164 | .14224 | .14133 | .07929 |
| | Grouped, substitute | .12393 | .12329 | .12194 | .06676 |
| | Grouped, XOR available | .12436 | .12341 | .12149 | .06169 |

On Parity, retaining XOR and randomizing the groups raised IID loss by .09888 [+.07969, +.11806]. Reducing detector accuracy from .92 to .78 raised the selected tree’s Brier to .11569, versus .25810 for the matched substitute (difference −.14240 [−.15553, −.12928]). Its IID and shift accuracies were .9849 and .8325. The source-fit prevalence constant scored .25037 IID. The frozen trees and oracle agree on individual predictions, including after this shift.

On CorrelatedRedundancy, the IID loss difference between grouped libraries was −.00044 [−.00141, +.00053]. Eleven banks selected identical trees across the two libraries. In the remaining bank the substitute library selected its extra logit-mean gate. The XOR library always selected the shared arithmetic-mean root over within-group logit sums. The stacker had the lowest IID loss (.08704). Under the published shift in redundant-provider dependence, the grouped arithmetic-root tree reached .06169, compared with stacking at .07792. The grouped-library shift difference was −.00507 [−.01622, +.00609]. Those numbers cannot be attributed to XOR when no winning tree used it.

Intervals use paired bank differences, averaging starts before inference; secondary intervals are unadjusted. Mean within-bank standard deviations across shuffled XOR starts were .03046 on Parity and .00553 on CorrelatedRedundancy, versus .02964 and .00837 for native GP. The inferred-group libraries are deterministic for fixed fit rows and anonymous column order. [All 336 paired contrasts](results/paired.csv)

## Cost and reproduction

Each library arm scores 8 × 384 = 3,072 fit rows in aggregate, including its substitute or XOR trials. There is no extra refinement or validation selection. The ordinary GP spends 160 × 384 = 61,440 fit row-scores including parents. The public Top-N ranks M individual providers, scans all M unweighted prefixes on fit, then freezes N for both test domains. With M=6/10 it costs 4,608/7,680 fit row-scores on Parity/CorrelatedRedundancy. Selected N averaged 5.17 and 5.00 respectively. The C=1 logit stacker trains one regularized model per bank. [All frozen choices, selected groups, prefix curves, tree sizes and timings](results/choices.csv)

On one shared CPU core, mean fit times on Parity were 1.44 ms for the grouped XOR library, 1.66 ms for the grouped substitute, 7.92 ms for Top-N, 2.83 ms for stacking and 51.64 ms for native GP. The corresponding CorrelatedRedundancy times were 1.91, 1.96, 12.77, 3.60 and 103.00 ms. A frozen grouped-XOR-library winner needed an average of .40 ms and .51 ms respectively to predict a 4,096-row IID set; only the first family actually selected the XOR node. Fit times include group discovery and construction for each library, and input file staging for native GP. They exclude synthetic data generation.

The benchmark used the NumPy backend and FusionBench commit `f074e85281c056e45585073073c93a0a921e88ff`. A fresh process on banks 30400 and 30411 recovered every non-timing field in 176 scores, 44 choices and 192 individual template assessments. In a separate scoring-only process all 48 split losses from the 12 frozen Parity winners matched the archived scores and each selected grouping matched the generator assignment. Reproduce from the OKAPI lab checkout with FUSIONBENCH set to the pinned FusionBench source.

```sh
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.parity_operator.run --out /tmp/parity-operator
python -m experiments.parity_operator.verify_posterior --results /tmp/parity-operator
```

[Search driver](run.py) · [Independent posterior and source-prevalence check](verify_posterior.py)
