# Trimming one extreme per tail in a fixed OKAPI tree

With one of seven providers intermittently stuck at a random extreme, trimming
reduced positive-class Brier from **0.13104 to 0.11606**. The paired difference
was **−0.01498** (95% t interval −0.01516 to −0.01481, 16 independent banks).
Part of the difference was present before any faults. On the very same cases
the fault increased mean-pool Brier by 0.01009 and trim-one Brier by 0.00524.
Trimming therefore avoided **0.00485** of the extra loss (paired 95% interval
0.00473 to 0.00497). Both quantities use all rows, including unaffected ones.

| Test distribution | Mean | Trim one | Median | Logistic stacking |
| --- | ---: | ---: | ---: | ---: |
| Clean IID | 0.12109 | 0.11110 | 0.10620 | **0.02936** |
| Clean counterpart of shifted rows | 0.12095 | 0.11082 | 0.10556 | **0.03005** |
| One provider stuck on affected rows | 0.13104 | 0.11606 | **0.10937** | 0.12710 |
| Four providers stuck at the same value | **0.19521** | 0.20807 | 0.30276 | 0.26488 |
| One informative provider, six report 0.5 | 0.22845 | 0.25000 | 0.25000 | **0.17350** |

Lower Brier is better. Under the minority fault, the median had the lowest
Brier and trim-one had the highest accuracy (93.72%, versus 92.88% for mean,
91.37% for median and 84.35% for stacking). For the common majority fault,
trimming increased Brier relative to the mean in all 16 banks, by 0.01286 on
average (95% interval 0.01239 to 0.01333). In the singleton control both
trimming and the median discarded all informative deviations from 0.5.

## Experiment and budget

FusionBench's calibrated Committee supplies seven equally accurate independent
providers conditional on the label. Each bank has 512 clean fit examples, 256
separate clean validation examples, 4,096 IID test examples and 4,096 fresh
examples shared by four shift conditions. The fault mechanism uses predictions
and independent randomness only. It selected 50.07% of shifted rows.
The stacker was frozen before either test split was drawn. The three fixed
rules performed zero fits, objective evaluations and validation passes.
Each rule consumed seven predictions per test item. The fixed tree experiment
used zero GP generations and zero local refinement steps.

Each bank fit six logistic stackers using the same 512 unique labels and made
six Brier passes over the same 256 validation labels (3,072 training and 1,536
validation example visits across configurations, excluding repeated solver
iterations). Training took 14.49 ms and validation 2.76 ms per bank on
average. The six optimizations used 50.81 solver iterations in total per bank
on average. Clean validation chose logit features in 12 banks and raw
probabilities in four. Across all 80 batches of 4,096 test items, average
inference time per batch was 0.18 ms for mean, 0.17 ms for trimming, 0.53 ms
for median and 1.11 ms for stacking. These shared-CPU timings describe one
run. Tree timings include construction. The raw measurements retain time
per method, seed and condition.

## Using and reproducing the operator

Add `okapi.operators.TRIMMED_MEAN` to the operators supplied in `allowed_ops`
to make it available to search. A fixed tree passes its parent value and all
child values to the operator. It drops one minimum and one maximum per
coordinate for three or more inputs and uses the arithmetic mean otherwise.
Streaming retains the running extremes and the sum of the inner values.
Multiclass callers can use OKAPI's probability-normalizing postprocessor.
The reported scores concern a fixed seven-provider binary tree.

Install the companion FusionBench research branch at commit
`329c565959d7fde4010e491629fbd928626db258`, or add its checkout to
`PYTHONPATH`. Run from the OKAPI checkout:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m experiments.trimmed_pool --out /tmp/trimmed_pool
```

The recorded environment used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and
scikit-learn 1.8.0. The predeclared seed banks are 2300–2315.

[Protocol recorded before measurement](trimmed_pool_protocol.md) ·
[Measurements by bank](results/trimmed_pool_measurements.csv) ·
[Validation and compute by stacker configuration](results/trimmed_pool_stacker.csv) ·
[Paired summary](results/trimmed_pool_summary.csv)
