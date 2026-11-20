# Reusing fitness for unchanged GP survivors

On two synthetic FusionBench families, reusing a surviving parent's fitness removed 46–49% of tree fitness evaluations at 12 generations. All 576 paired population fingerprints and all selected IID and shifted predictions agreed exactly. The wall-time intervals include zero. Reusing scores still requires crossover, mutation, deduplication and selection in every generation.

Keeping the **fitness evaluation budget** fixed let OKAPI complete about 22 generations. The extra search lowered held-out Brier loss on both families and shifts. It roughly doubled elapsed fit and validation time. The supervised logit stacker had the lowest Brier throughout.

## Protocol

The comparison uses the actual NumPy OKAPI population, crossover, mutation, deduplication and survival selection. Each run starts with one tree per expert and six seeded shallow compositions from the FusionBench adapter. Trees have at most 15 nodes. The offspring multiplier is 2 and tournament size is 3. The deterministic objective is one minus binary positive-class Brier loss. Both fixed-generation arms use the same seed and data. The stock arm uses the default setting. The reuse arm sets `reuse_survivor_fitness=True`. Keep the switch opt-in for deterministic scoring with unchanged parent inputs.

TwoPathways has eight providers observing four latent conditions. The shift reduces detector accuracy from 0.90 to 0.78. CorrelatedRedundancy uses eight correlated providers and two independent providers. The shift reduces the block correlation from 0.85 to 0.10. Data seeds 8200–8211 cross GP seeds 53 and 97 in each family. Each data seed draws 1,024 fit examples, 256 validation examples, 4,096 IID test examples and 4,096 shifted test examples using distinct random streams. The two search seeds share examples. Every method sees identical evaluation examples within a paired cell.

The stock and reuse arms run for 12 generations. At each generation boundary a SHA-256 digest captures the ordered survivor trees’ exact node parameters and child counts, model IDs, mutation probabilities, fitness, fit predictions and RNG state. The script also checks that temporary offspring lists and value-node evaluation caches are empty. It checks the selected stock and reuse trees’ output arrays again on both test sets. Extra audit predictions on copied trees are excluded from application costs. The two arms alternate execution order across seeds. A fresh-process replay at data seed 8200 and search seed 53 reproduced all 24 generation records and every non-timing value in its 20 method/domain records.

For reinvestment the cap equals the actual number of stock tree evaluations in that paired cell. Each fitness evaluation scores one tree on all 1,024 fit rows. Any generation that would exceed the remaining budget is stopped before evaluating its fitness batch. The last completed population goes to validation. Time spent proposing the final rejected batch counts. All three arms score the top six candidates by training fitness on the same 256 held-out rows, choosing the one with lowest validation Brier. Validation adds 1,536 row scores per arm. The test sets are drawn after all three choices freeze. There is no further refinement. The arithmetic mean needs no fitting. The logit stacker uses the combined 1,280 labeled fit and validation rows.

## Held-out probability quality

Entries are binary Brier loss, lower is better. The final column subtracts stock from reinvestment. Parentheses give an unadjusted paired 95% Student interval over 12 independent data seeds after averaging the two GP seeds within each.

| Family | Domain | Stock / reuse | Reinvest | Mean | Logit stack | Reinvest − stock (95% interval) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| TwoPathways | IID | .16612 | .15900 | .15498 | .13029 | −.00711 (−.01035, −.00387) |
| TwoPathways | Weaker detectors | .20110 | .19752 | .19249 | .17330 | −.00358 (−.00538, −.00178) |
| CorrelatedRedundancy | IID | .13184 | .12808 | .14409 | .08396 | −.00376 (−.00632, −.00120) |
| CorrelatedRedundancy | Lower correlation | .12872 | .12593 | .12145 | .06753 | −.00279 (−.00508, −.00050) |

## CPU and fitness budget

Times include materializing fit files, initializing and evolving the population, and selecting a winner on validation. Data generation and process startup are excluded. Separate inference and end-to-end times for each test domain are reported in the per-run CSV. The benchmark uses in-process elapsed time on a shared Intel Xeon Platinum 8573C host with numerical library threads limited to one. Wall-time uncertainty includes host load.

| Family | Mean fit calls, stock / reuse / reinvest | Reinvest generations | Mean fit + selection ms, stock / reuse / reinvest | Reuse − stock ms (95% interval) |
| --- | --- | ---: | --- | --- |
| TwoPathways | 363.25 / 195.25 / 354.67 | 21.79 | 87.48 / 87.25 / 186.17 | −0.23 (−8.76, 8.30) |
| CorrelatedRedundancy | 391.92 / 199.92 / 384.58 | 22.38 | 100.27 / 97.24 / 205.87 | −3.03 (−14.33, 8.27) |

Each saved fitness call represents an entire 1,024-row tree prediction and objective pass. The whole-generation stopping rule used an average of 97.6% and 98.1% of available calls on TwoPathways and CorrelatedRedundancy respectively. Every run stayed within its own cap. The paired time intervals for reinvestment minus stock were +82.58 to +114.80 ms on TwoPathways and +90.43 to +120.78 ms on CorrelatedRedundancy. These estimates cover inexpensive binary CPU fitness evaluation on two simulation families.

## Reproduction

Use this OKAPI branch and FusionBench revision `4cf4810e9876f51837b6eba1c4d7aa4bf1c4dac7`. The recorded run used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0, loguru 0.8.0 and CPU PyTorch 2.15.0. Set `FUSIONBENCH` to the companion checkout and run from this OKAPI checkout in an environment containing both projects' dependencies.

```sh
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python experiments/survivor_reuse.py --out /tmp/survivor-reuse
```

[Driver](survivor_reuse.py) · [Run configuration](results/survivor_reuse/config.json) · [Generation fingerprints](results/survivor_reuse/trajectories.csv) · [Per-run results and times](results/survivor_reuse/metrics.csv) · [Paired estimates](results/survivor_reuse/paired.csv)
