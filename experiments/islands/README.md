# Do four OKAPI islands benefit from migration?

On TwoPathways, transferring one fit elite every second round improved held-out Brier over four isolated islands in 10 of 12 independent data/optimizer banks. At identical fit and validation budgets the mean difference was **−0.00670** (paired 95% interval −0.01266 to −0.00073). A single population of the same total size did better again. Its loss was **0.14692**, compared with **0.15238** for the ring. The ring-minus-single difference was +0.00546 (+0.00067 to +0.01025). The prespecified criterion for preferring migration over both controls was unmet.

## Equal evaluation budget

| Family and test condition | Four isolated | Ring | One population | Mean | Logit stack |
|---|---:|---:|---:|---:|---:|
| TwoPathways IID | .15908 | .15238 | .14692 | .15519 | .13233 |
| TwoPathways detector accuracy .90 → .78 | .19786 | .19381 | .19160 | .19272 | .17406 |
| CorrelatedRedundancy IID | .12891 | .12290 | .12388 | .14326 | .08485 |
| CorrelatedRedundancy block correlation .85 → .10 | .12742 | .12334 | .12258 | .12163 | .07154 |

All GP arms started with the same 24 programs and spent **480 assessments × 384 fit rows**, including assessed immigrants, followed by **four evaluations × 256 validation rows**. An unfinished terminal generation contributed only its evaluated offspring. Each island had its own 120-call cap. A ring stopped exchanging migrants when its first island reached that cap. A logit stacker fitted to all 640 labeled rows achieved the lowest Brier in each cell.

In CorrelatedRedundancy, intervals for the ring against either GP comparator covered zero in both test conditions. The ring admitted an average 13.83 of 24.67 offers (17.92 assessed). On TwoPathways the corresponding figures were 13.08 admissions, 23.00 offers and 16.58 assessments. Across final populations of 24 trees, the average number of distinct prediction vectors on 32 fit examples went from 18.92 for isolated islands to 15.58 for the ring on TwoPathways, and from 19.92 to 15.42 on CorrelatedRedundancy. This records an association between migration and lower diversity. Only the migration policy itself was experimentally varied.

## Time, splits and reproducibility

Equal assessments took different elapsed times. Mean search plus validation on TwoPathways used about 254 ms for the four isolated islands, 226 ms for the ring and 229 ms for the single population. On CorrelatedRedundancy these means were 302, 220 and 211 ms. The measured process ran sequentially on one logical CPU with BLAS threads limited to one. This measures search efficiency on that hardware. The study did not measure parallel scaling or network communication latency.

For the time-matched comparison each paired bank used the fastest arm's terminal search time as its common ceiling, then chose the latest completed snapshot within this ceiling for each arm. Mean ceilings were 196 ms (TwoPathways) and 190 ms (CorrelatedRedundancy). At that endpoint the TwoPathways IID Briers were .15899, .15396 and .14712 for isolated, ring and single. Ring minus isolated was −.00503 (−.01023 to +.00018). For CorrelatedRedundancy IID the corresponding Briers were .13167, .12327 and .12444. Ring minus isolated was −.00840 (−.01383 to −.00297). Per-run data include the achieved time, next snapshot gap, validation time, inference time, fitness calls and functional diversity. The common ceiling uses timing information from the paired run and is a retrospective comparison. Snapshot copies, diversity measurements, test inference and simulation are outside the search timer. Hardware time endpoints vary across replays.

The selector averaged validation scores over four development banks and two families. It chose period 2 (mean Brier .12846) over period 5 (.13140), before drawing confirmation tests. Each confirmation bank used a fresh dataset and fresh optimizer tuple, separate fit, validation, IID and shifted streams, and arm order balanced over all six permutations. Held-out IID and shifted inputs were generated after all selections. Every test domain had 4,096 new rows per bank. The intervals summarize 12 independent banks, with one primary contrast and unadjusted secondary comparisons. A fresh process replayed both families for seeds 21200 and 21211 and reproduced all 40 non-timing rows at the evaluation budget and fixed baselines exactly.

## API and reproduction

Call the migration helper between complete generations on two or more independently constructed OKAPI engines. It picks the closest elite to each source's existing objective ideal, copies and rebinds every provider to receiver tensors, checks caps, allowed operators and provider IDs, scores each admissible arrival once, and applies the receiver's usual survival rule. The returned immutable outcomes include sender, receiver, reason, assessment and admission. Duplicate detection follows the existing printable tree codes, including their rounding of numeric parameters. The engines share OKAPI's process-level backend and postprocessor. The ordinary GP loop changes only when the caller invokes this helper.

```python
from okapi.islands import migrate_ring

for island in islands:
    island.run_iteration()
events = migrate_ring(islands)
extra_fit_calls = sum(event.evaluated for event in events)
```

Use FusionBench commit `4cf4810e9876f51837b6eba1c4d7aa4bf1c4dac7` and this OKAPI branch with both projects' dependencies. Set `FUSIONBENCH` to that checkout.

```sh
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python experiments/islands/run.py development --out /tmp/islands
python experiments/islands/run.py confirmation --out /tmp/islands
```

[Protocol](protocol.md) · [Development runs](results/development.csv) · [Locked interval](results/chosen.json) · [Confirmatory outcomes and costs](results/confirmation.csv) · [Paired differences](results/contrasts.csv) · [Driver](run.py)
