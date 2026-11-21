# Scoring 63 distinct min/max functions from 135 guided templates

The opt-in `deduplicate=True` option reduces the guided initializer's library from **135 to 63** candidates. Scoring the library on 4,096 labeled examples saves 72 full tree evaluations, or 294,912 tree-row scores per objective. On both TwoPathways and CorrelatedRedundancy, mean construction and average-precision scoring time fell by about half. In all 20 paired data banks both versions produced exactly the same first fit winner and the same complete set of probability functions on the four measured splits. [Registered protocol](protocol.md)

## Certificate and use

At a fixed hierarchy cut, each group mean is one real-valued operand. For any threshold, a minimum exceeds it exactly when both operands exceed it, and a maximum exceeds it when either operand does. Each min/max expression can therefore be evaluated symbolically by a Boolean truth table with at most 16 rows. If two expressions agree on all Boolean assignments, then their real outputs exceed exactly the same thresholds for every vector of finite inputs. Their outputs are equal. The implementation uses bitwise AND and OR, preserves the first eligible representative, and restarts its equivalence check at each cut.

| Number of groups | Original programs | Distinct truth tables |
| ---: | ---: | ---: |
| 1 | 1 | 1 |
| 2 | 2 | 2 |
| 3 | 12 | 8 |
| 4 | 120 | 52 |
| **Total** | **135** | **63** |

Use the flag with binary probability scores and identity node postprocessing. A nonlinear postprocessor applied inside every node can break equivalence between parenthesizations. Both entry points retain their original enumeration by default. For the engine initializer, the usual depth, size and operator restrictions apply before accepting the first representative. It scores existing residents as well as the surviving templates. Equivalent syntactic programs can mutate or cross over differently, and discarding copies changes the population given to survival selection. This study measures the initial candidate functions and fit-selected winner. Later GP trajectories can differ.

```python
from okapi.guided import correlation_templates, seed_from_correlations

templates = correlation_templates(fit_provider_probabilities, deduplicate=True)
# Alternatively, on an initialized NumPy engine before evolving its population
initial_fit_calls = seed_from_correlations(engine, deduplicate=True)
```

## Measured cost

Ten independent banks (27100–27109) per family used anonymous providers, 4,096 fit rows, 256 validation rows and 4,096 rows each for IID and shifted test. Within each bank both arms saw identical inputs. The split construction and shifts follow the [guided search experiment](../guided/README.md). Each objective independently paid to construct and score its entire library and selected the first best fit candidate. The timing endpoint covers construction, fit predictions, metric computation and selection. Order was counterbalanced and numerical libraries used one thread on logical CPU 0. Intervals are paired, unadjusted 95% Student intervals across the ten banks.

| Family | Objective | Full mean (ms) | Compressed mean (ms) | Compressed − full, ms (95% interval) |
| --- | --- | ---: | ---: | --- |
| TwoPathways | Average precision | 203.0 | 100.8 | −102.2 (−113.1, −91.3) |
| CorrelatedRedundancy | Average precision | 210.2 | 99.0 | −111.2 (−123.0, −99.4) |
| TwoPathways | Brier loss | 40.2 | 12.3 | −27.9 (−67.4, +11.6) |
| CorrelatedRedundancy | Brier loss | 20.7 | 10.8 | −9.9 (−11.9, −8.0) |

Fit-row scores fell from 552,960 to 258,048 under both objectives. Each frozen winner received a single post-selection scoring pass per split. One 4,096-row IID inference and metric pass took between 1.43 and 1.65 ms on average across the AP conditions and between 0.15 and 0.26 ms in the Brier conditions. Deployment still evaluates the same winning tree. The timing gains apply to library search. The TwoPathways cheap-loss interval includes zero on the shared host. [Raw measurements](results/runs.csv) · [Paired timing and call estimates](results/paired.csv)

## Quality and functional coverage

| Family | Fit objective | Fit | Validation | IID | Shift |
| --- | --- | ---: | ---: | ---: | ---: |
| TwoPathways | Brier (lower is better) | .06657 | .06577 | .06658 | .15516 |
| CorrelatedRedundancy | Brier | .13809 | .14151 | .14022 | .12278 |
| TwoPathways | Average precision (higher is better) | .98309 | .98391 | .98351 | .86299 |
| CorrelatedRedundancy | Average precision | .92003 | .91379 | .91611 | .95806 |

The two arms have identical entries. Across both objectives and all banks, the first winning tree representation, its fit objective and all four output arrays agreed exactly. An independent diagnostic hashed each candidate's joint row-aligned outputs across fit, validation, IID and shift. All 20 pairs returned the same 63 distinct SHA-256 signatures. This diagnostic incurred another 540 versus 252 tree evaluations per bank, covering 1,693,440 versus 790,272 tree-row scores, in addition to the paid optimization runs. It does not enter selection or search timing. A fresh-process rerun of banks 27100 and 27109 reproduced every non-timing field in 64 score records and eight function-set records. Tests also cover all Boolean assignments on four providers, continuous held-out scores, shuffled assignments and engine depth restrictions. A two-bank replay of the previously published ambiguous-group experiment reproduced all non-timing fields of its 24 fit, 12 recovery and 56 test records using the original helper interface. [Full-function audit](results/functions.csv)

## Reproduction

Use the environment described in the [guided search experiment](../guided/README.md) with the pinned FusionBench commit `4cf4810e9876f51837b6eba1c4d7aa4bf1c4dac7`. From the OKAPI checkout run the following commands.

```sh
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m pytest -q test/test_guided.py test/test_guided_equivalence.py
python experiments/guided_equivalence/run.py --out /tmp/guided-equivalence
```

[Benchmark driver](run.py) · [Public constructor and initializer](../../okapi/guided.py) · [Equivalence tests](../../test/test_guided_equivalence.py)
