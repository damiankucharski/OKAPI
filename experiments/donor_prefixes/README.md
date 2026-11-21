# Did partially evaluated trees contain the missing donors?

At generation 15, **0 of 32 live populations** could construct the matched two-pathway rule by joining two pieces from different target-free programs. This remained true after exposing the partial results of every operation inside every source tree. Sixteen populations held one of the two planted conjunctions and sixteen held neither. The two conjunctions never coexisted in any of these 32 live populations. The empirical average across 16 independent banks was zero. The one-sided 95% upper bound for a new bank to have at least one positive start out of two is **17.1%**. [Registered retrospective analysis](protocol.md) · [Registered supplementary count](exposure.md) · [Independent source experiment](https://github.com/damiankucharski/FusionBench/blob/31466b16b341f9058a2c7db222dbde73b9e29448/docs/ensemble-study/semantic-donors/README.md)

## What the archiving test revealed

The target was MAX(MIN(A,B),MIN(C,D)) with anonymous input roles permuted independently for each bank. We allowed every possible MIN and MAX of two distinct donors, also counting decompositions other than the planted pair. Each donor came from a different source genotype. We separately excluded donors equal to the target and programs whose full output already equaled the target. We evaluated all joins on the 16 binary corners. MIN/MAX commute with thresholding, so this fixes the full functions on real inputs.

| Matched task, G15, 32 starts per row | Neither conjunction | Exactly one | Both from distinct programs | Any strict join giving target |
|---|---:|---:|---:|---:|
| Live population, complete ValueNode outputs | 16 | 16 | 0 | 0 |
| Live population, all intermediate outputs | 16 | 16 | 0 | 0 |
| All programs scored before G15, complete ValueNode outputs | 14 | 17 | 1 | 1 |
| All programs scored before G15, all intermediate outputs | 13 | 18 | 1 | 1 |

The historical pair in bank 6614, GP seed 809, was already available using the complete outputs of two source programs. The additional solitary conjunction in bank 6608, seed 809, only appeared as an intermediate output. Across both laws at G1, 5, 10 and 15, all four extraction policies agreed on join availability in **512 of 512** live or historical states. At matched G15, three live populations contained any program with sequential updates and one gained a globally new donor function from a prefix. The corresponding historical counts were 17 and two. On the additive control those counts were one and zero live, 17 and one in the archive. [All availability results](results/availability.csv) · [Exposure by state](results/exposure.csv) · [Counts by law and generation](results/exposure_summary.csv)

A counterexample establishes that the extraction policies can differ. P = MAX(MIN(A,B),A) returns A, and Q = MAX(MIN(C,D),C) returns C. All their complete ValueNode outputs are single providers. The first update in each source exposes one needed conjunction, and joining those two with MAX gives the target on every input. The independent native checker physically truncates the two programs and verifies this positive control. [Native checker](verify_native.py)

## Search and predictive quality

The true matched function has a seven-node native program. On the source panel, the exact empirical MIN/MAX winner attained IID positive-channel Brier .18746. GP attained .24719 at 400 fit evaluations, the ordinary mean .21341, ranked-prefix Top-N .21610 and regularized stacking .21401 on the same rows. The prefix audit reads already fixed search trajectories, makes zero new fitness calls and consumes no additional labels. These opportunity counts describe the states reached by this search. They cannot tell us how a new donor-construction or retention policy would perform. [Fit, validation, IID, shifted losses and comparison costs](https://github.com/damiankucharski/FusionBench/blob/31466b16b341f9058a2c7db222dbde73b9e29448/docs/ensemble-study/semantic-donors/README.md)

## Functional use and export

For functional-use questions, measure interventions on the intact frozen prediction. In an older Compound experiment an equivalent tree duplicated an idempotent update. Bypassing either of the two copies individually changed zero decisions. Bypassing both reproduced all 819 repairs and 156 harms of the single update in the original serialization on seed 0 IID. In a future joint benchmark with a specified repeated subprogram, use the existing native grouped bypass on the set of occurrence paths and report the paired change on independent rows alongside the individual bypasses. Our current donor sets retain which programs supply each function. They support a count of available pieces, whereas measuring their effect on the source program would require intervening on their occurrences. [Prior empirical comparison](https://github.com/damiankucharski/OKAPI/blob/86e84ec24974de3ead4e8dba5077d4080b161587/experiments/results/operator_equivalence.csv) · [Grouped intervention implementation](https://github.com/damiankucharski/OKAPI/blob/86e84ec24974de3ead4e8dba5077d4080b161587/okapi/interventions.py)

The new FusionBench topology experiment measures diversity of whole-program outputs. Its full benchmark has no common specification of affine logit experts on shared raw inputs. For a future genuinely recovered binary MIN/MAX tree with those experts and identity postprocessing, use the existing `compile_binary_lattice(tree, affines)` and compare its sigmoid output with the frozen tree on identical held-out inputs. The earlier eight-instance test preserved the selected trees to maximum probability error 5.56e−16. It also showed why exact compilation and successful discovery require separate checks. [Existing compiler and input contract](https://github.com/damiankucharski/OKAPI/blob/35ca8e6b22522c64bb8829b451c385caffd1516d/okapi/lattice.py) · [Earlier empirical compilation test](https://github.com/damiankucharski/FusionBench/blob/d10b51e7e279597405563dbdddc667cd0cda9260/docs/ensemble-study/lattice-compile/README.md)

## Reproduction

Run the commands from the OKAPI repository, with `TRACE_DIR` pointing to the directory containing both archived files. The independent verifier physically rebuilt 1,706 distinct truncated programs, including 139 distinct proper intermediate expressions, and checked native predictions on 16 corners and all 256 four-level inputs per program. It reproduced all 2,048 join-availability flags across 64 traces with 25,600 chronological scored entries. A separately computed comparison also reproduced all 512 supplementary states and all 1,024 matched source-policy component records. [Source archive and file checksums](../donor_opportunity/README.md) · [Recorded native replay](results/verified.json)

```sh
python -m experiments.donor_prefixes.analyze "$TRACE_DIR" /tmp/okapi-prefix-results
python -m experiments.donor_prefixes.verify_native "$TRACE_DIR/traces.jsonl.gz" \
  --reference /tmp/okapi-prefix-results/availability.csv
```
