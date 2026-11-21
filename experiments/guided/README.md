# Initializing OKAPI search from correlations between anonymous providers

On TwoPathways, selecting from 135 correlation-guided programs reached an IID Brier loss of **0.06715**. At 160 tree evaluations, shuffling provider assignments in the same templates gave **0.14700** and singleton-initialized GP gave **0.18521**. Regularized logit stacking on the same fit labels gave **0.13398**. All 24 guided incumbents at 160 evaluations already came from the template library. The paired guided-minus-shuffled difference was −0.07985 (95% interval −0.08485 to −0.07484), with a lower loss in all 12 data banks.

On CorrelatedRedundancy the corresponding advantage was much smaller. By 640 evaluations, guided and shuffled GP had similar IID Brier, 0.13273 and 0.13279, and stacking achieved 0.08835. A correlation-based grouping can identify mutually redundant providers in either family. The value of composing those groups with extrema depends on how the label is generated. [Protocol committed before scoring](protocol.md)

## What the search knew

A previous study measured IID Brier 0.06744 for an achievable TwoPathways tree built with the true grouping of detectors. This motivated a specific search prior. Find clusters of providers that vary together, average within clusters, and try combinations of minima and maxima over the averages. The proposal uses the providers’ predictions on the 384 fit rows to form clusters. It sees anonymous column IDs. In this experiment, the grammar and its four-group maximum were chosen with prior knowledge of the problem family. Its utility across other families is an empirical question. The earlier achievable-tree estimate used a different sample and provides a reference level. These experiments do not establish a global optimum. [Earlier achievable-tree study](https://github.com/damiankucharski/OKAPI/blob/639c5ff/experiments/fit_gap/README.md)

TwoPathways has eight providers, two per latent condition in a disjunction of two conjunctions. The shift reduces detector accuracy from .90 to .78. CorrelatedRedundancy has ten providers, including a block of eight whose pairwise noise correlation changes from .85 to .10 under shift. Each bank shuffled provider columns before writing them to anonymous filenames. It preserved that ordering across splits. The program sets used only fit probabilities and labels. All incumbents and baselines for all banks froze before the first validation or test split was generated.

## Fit, validation and test results

The entries are mean binary positive-class Brier loss, with smaller values preferred. Each of the 12 banks contributed the mean of its two GP starts. The fixed mean and logit stacker were evaluated once per bank. “Library” freezes the best fit template immediately after scoring the template set. It includes the shared cost of scoring the 8 or 10 singletons first. The arithmetic mean reads no labels. The stacker uses the same 384 fit labels as the GP methods, with C fixed to 1.

| TwoPathways | Fit calls | Fit | Validation | IID | Shift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arithmetic mean | 0 | .15263 | .15437 | .15504 | .19250 |
| Logit stacking | 0 GP calls | .12120 | .13259 | .13398 | .17598 |
| Guided library | 143 | .06950 | .06854 | .06715 | .15423 |
| Shuffled library | 143 | .14290 | .14728 | .14866 | .19007 |
| Native GP | 160 | .17348 | .18365 | .18521 | .21306 |
| Shuffled GP | 160 | .14037 | .14516 | .14700 | .18963 |
| Guided GP | 160 | .06950 | .06854 | .06715 | .15423 |
| Native GP | 640 | .15331 | .16205 | .16449 | .20204 |
| Shuffled GP | 640 | .11191 | .12369 | .12124 | .18260 |
| Guided GP | 640 | .06891 | .07020 | .06865 | .15532 |

All 24 early guided winners have 15 nodes. At 640 assessments, 14 of 24 guided winners first appeared during evolution, with an average of 22.92 nodes across all 24. Further optimizing fit Brier lowered it from .06950 to .06891, while losses on validation and both test distributions increased. IID accuracy also fell from .94956 to .94302.

| CorrelatedRedundancy | Fit calls | Fit | Validation | IID | Shift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arithmetic mean | 0 | .14773 | .14443 | .14460 | .12167 |
| Logit stacking | 0 GP calls | .08275 | .08555 | .08835 | .07554 |
| Guided library | 145 | .13911 | .13762 | .14130 | .12623 |
| Shuffled library | 145 | .14460 | .14300 | .14512 | .12687 |
| Native GP | 160 | .13986 | .14143 | .14294 | .13831 |
| Shuffled GP | 160 | .14318 | .14231 | .14435 | .12829 |
| Guided GP | 160 | .13884 | .13754 | .14123 | .12656 |
| Native GP | 640 | .12849 | .12957 | .13207 | .12651 |
| Shuffled GP | 640 | .12950 | .12971 | .13279 | .12872 |
| Guided GP | 640 | .12946 | .13062 | .13273 | .12855 |

Stacking had the lowest loss on every split in this family. At 640 calls guided GP had mean tree size 25.79 nodes, compared with 11.38 for native GP. The arithmetic mean had lower loss on the reduced-correlation shift than each of the three GP arms at that budget.

## Paired uncertainty

Each interval is an unadjusted paired Student 95% interval across the 12 banks. Lower differences favor the guided assignment. The TwoPathways IID comparison at 160 calls was the preregistered primary contrast. The two SD columns describe the guided method's IID loss. Between-bank SD uses the 12 bank averages. Within-bank SD averages the 12 sample SDs between two optimizer starts.

| Family | Calls | Guided − shuffled IID (95% interval) | Lower IID banks | Guided − shuffled shift (95% interval) | Between-bank SD | Within-bank SD |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| TwoPathways | 160 | −.07985 (−.08485, −.07484) | 12/12 | −.03540 (−.03768, −.03312) | .00155 | .00000 |
| TwoPathways | 640 | −.05259 (−.05876, −.04642) | 12/12 | −.02728 (−.02961, −.02496) | .00159 | .00078 |
| CorrelatedRedundancy | 160 | −.00313 (−.00607, −.00019) | 10/12 | −.00172 (−.00698, +.00354) | .00314 | .00033 |
| CorrelatedRedundancy | 640 | −.00006 (−.00504, +.00492) | 6/12 | −.00017 (−.00562, +.00528) | .00577 | .00112 |

## Budget and implementation

1. Each arm begins with the same K stock singletons. Guided initialization performs average-linkage clustering on one minus signed Pearson correlation. Cuts at one through four groups yield 1, 2, 12 and 120 trees, including functionally equivalent programs. The shuffled control changes the provider assignments with one permutation per start across all four cuts, preserving group sizes, shapes and scoring cost.
2. The two template arms score every proposal, then apply stock Pareto/proximity survival to K trees. All arms subsequently use stock crossover, the six default operators, offspring multiplier 2, tournament size 2, ordinary survivor rescoring and unrestricted tree size. All deploy the best full-precision training score ever observed, with first observation breaking ties. This is a shared incumbent selection policy, including for native GP. The budget includes rescored survivors.
3. Every arm makes exactly 160 or 640 fit calls, equal to 61,440 or 245,760 tree-row scores. The checkpoint can fall partway through a generation. One frozen tree is scored once on 256 validation rows and once on each independent 4,096-row test set. The fit diagnostic independently recomputes the winner's original score, agreeing to within 2 × 10⁻¹⁵. Validation selects zero candidates.

Elapsed fit seconds include fit-file I/O, model initialization, template construction and scoring when applicable, survival and evolution up to that checkpoint. Mean fit seconds for native / shuffled / guided on TwoPathways were .101 / .034 / .024 at 160 calls and .501 / .145 / .127 at 640. On CorrelatedRedundancy the corresponding times were .102 / .026 / .026 and .531 / .157 / .153. For example, guided construction, scoring and initial survival averaged 7.7, 5.6 and 4.4 ms on TwoPathways. At 640 calls, native started 34.08 generations on average and guided started 22.21. Equal assessment counts can therefore incur different costs for variation, deduplication and survival. Mean single-tree scoring times across GP checkpoints and arms were .46 ms on TwoPathways validation and .60 ms on its IID test. Times are descriptive for a shared CPU. [Per-run score, timing, cost and tree records](results/runs.csv) · [Paired contrasts for all splits and controls](results/paired.csv)

## Reproduction

Use this OKAPI branch and FusionBench commit `4cf4810e9876f51837b6eba1c4d7aa4bf1c4dac7`. The run used CPU PyTorch, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0 and loguru 0.8.0. Install both projects’ dependencies and set `FUSIONBENCH` to the pinned checkout. Run from the OKAPI checkout.

```sh
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python experiments/guided/run.py --out /tmp/guided-reproduction
```

The default banks are 25100–25111, crossed with GP seeds 311 and 751. A fresh-process replay of banks 25102 and 25110 with both starts reproduced all 288 records in all 20 non-timing fields exactly, including program, incumbent assessment number, size, Brier and accuracy. [Experiment driver](run.py) · [Opt-in template constructor](../../okapi/guided.py)
