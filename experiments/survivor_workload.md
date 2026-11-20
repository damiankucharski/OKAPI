# Does survivor reuse save time on a ranking objective?

## Protocol specified before measurement

The first survivor experiment found fewer evaluations and similar wall time on a cheap Brier objective. This follow-up measures the same opt-in flag with ordinary binary average precision on one preselected FusionBench family, TwoPathways with two detectors per condition. Both arms use the stock GP operators and the initialization from the survivor experiment, a 15-node cap, six generations and a separate 256-row validation set. Select the best validation average precision from the top six fit candidates. Evaluate its average precision on 4,096 held-out IID examples and on 4,096 examples with detector accuracy reduced from 0.90 to 0.78.

Two fit sizes are fixed at 1,024 and 32,768. Each size has eight paired runs from independent data seeds 9200–9207 with paired optimizer seeds 153–160. Each seed supplies the same source examples to both arms, and the small fit set is a prefix of the large fit set. Stock runs first for half the seeds in each size and reuse runs first for the other half. Size order alternates across seeds. Each arm runs once per seed and size. Pin the process to the first permitted logical CPU when Linux affinity is available and limit numerical library threads to one.

The primary endpoint is the paired difference in elapsed time for search and selection at 32,768 rows. Include per-arm fit-file writes, model initialization, all GP evaluations and the six validation scores. Exclude shared generation of predictions, process startup, test inference and diagnostic assertions. Record search, selection and held-out inference separately. The 1,024-row condition provides a scale control. Report the mean paired time difference and its unadjusted 95% Student interval over the eight independent seeds; the ratio of the average times summarizes relative change. Record objective calls, validation calls and output parity on both test sets. Report both time estimates regardless of sign. An untimed, independent tiny warm-up exercises both code paths before timing. The scope is a single shared-host benchmark with a deterministic ranking objective.

## Results

At 32,768 fit examples, survivor reuse reduced mean search and validation time from 1.243 to 0.751 seconds, a 39.6% reduction. The paired 95% interval for time saved was 0.462 to 0.522 s. The smaller workload also benefited. Binary average precision requires ranking the predictions at each fitness call. Skipping a repeated ranking measurably reduced elapsed time in this workload.

| Fit rows | Mean stock time (s) | Mean reuse time (s) | Reuse − stock (s), 95% interval | Mean fit calls, stock / reuse | Mean IID / shift AP, both arms |
| ---: | ---: | ---: | --- | ---: | --- |
| 1,024 | .1911 | .1266 | −.0645 (−.1017, −.0273) | 199.13 / 115.13 | .84403 / .72910 |
| 32,768 | 1.2430 | .7512 | −.4918 (−.5217, −.4618) | 202.75 / 118.75 | .86722 / .74229 |

One fitness call covers all fit rows. The six additional validation calls cover 256 rows each. All 32 paired held-out output arrays matched exactly. Logical CPU 0 was used throughout. Each arm ran first four times per size. The eight seeds quantify between-run uncertainty on this host. A different CPU, scoring function, population size or callback can change the timing result.

[Raw per-run measurements](results/survivor_workload.csv) · [Benchmark driver](survivor_workload.py)

## Reproduction

Use the environment and pinned FusionBench commit specified in the [survivor experiment](survivor_reuse.md). In this OKAPI checkout with the companion checkout on `PYTHONPATH`, run the following command. The driver prints paired estimates and records elapsed time and score for every run.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python experiments/survivor_workload.py --out /tmp/survivor_workload.csv
```
