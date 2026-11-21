# Two relevant groupings can explain identical provider predictions

The two best correlation-ranked matchings covered the correct grouping in
**24 of 24 labeled worlds**. The first matching alone was correct in 12.
Scoring both sets of OKAPI trees with fit labels and selecting on separate
validation rows gave IID Brier **0.15023**, compared with **0.19202** for
two random matchings at the same 240 fit and four validation assessments.
The preregistered difference was **−0.04180** (paired 95% interval −0.04490
to −0.03869, wins on all 12 independent seed banks).

| Method | Fit Brier | Validation Brier | IID Brier | Shift Brier | IID accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two highest matchings | 0.14947 | 0.14796 | 0.15023 | 0.17991 | 81.61% |
| Highest matching only | 0.17665 | 0.17620 | 0.18114 | 0.20036 | 73.01% |
| Two random matchings | 0.18700 | 0.18656 | 0.19202 | 0.20795 | 70.73% |
| Stock-operator GP | 0.19975 | 0.21319 | 0.21389 | 0.23455 | 66.39% |
| Mean | 0.19650 | 0.19567 | 0.19659 | 0.20556 | 71.43% |
| Logistic stacker | 0.17577 | 0.18416 | 0.18406 | 0.20449 | 71.77% |
| Known matching and paths | — | — | 0.15023 | 0.17991 | 81.61% |

## Which signal identifies the grouping?

Each bank has two worlds with **the same eight columns on every example**.
Any sensor logit is proportional to the sum of the two independent
Rademacher signs on its incident edges plus independent Gaussian noise.
Sensors sharing an edge have population correlation 1/(2 + σ²) on logits,
and all other sensor pairs have zero. The eight correlated pairs form an
even cycle. Selecting the even or the odd edges around it covers every
sensor exactly once, so two of the 105 perfect matchings have full support.
Swapping which matching determines the labels leaves every prediction
unchanged. No statistic of these observations can resolve that swap.

The driver ranks the 105 matchings by the sum of four observed correlations.
Each offered matching contributes 120 read-once min/max trees over the four
pair means. The two highest proposals included both cycle matchings on
every bank. The selected tree used the correct one in both worlds every
time. Two random proposals included the correct matching in 0 of 24 worlds.
With the single highest proposal, mean IID Brier was 0.15022 when it belonged
to the label-generating world and 0.21206 when it belonged to its twin.
The two-proposal method matched the known structural reference's Brier on
all 48 world-by-test-distribution evaluations. This reference applies the
hidden pathway structure to observed probabilities. The Bayes posterior
given the noisy sensors is outside the scope of this comparison.

The world labels disagreed on 48.96% of IID rows and 49.12% of shifted rows.
On such a row any **common** predicted probability p has pair-averaged
Brier `(p−0.5)² + 0.25`. This gives a lower bound of 0.12241 on overall
empirical IID Brier for a single predictor shared across worlds, even if
its scores on agreement rows are perfect. Supervised fitting produces a
different predictor per world. Top-two reached 0.20861 conditional IID
Brier on disagreement rows and 0.09422 on agreement rows. For random-two
the respective losses were 0.26085 and 0.12595. Stacking gave 0.29643 and
0.07626. Conditional scores and the floor are diagnostics using
both labels after evaluation.

## Budget and interpretation

Per world, every supervised search used 384 fit labels. The three 240-call
searches each scored 92,160 fit example–candidate pairs and four finalists
on 256 independent validation rows (1,024 validation predictions). Top-one
used 120 fit assessments. The stacker used logit inputs, fixed C=1 and
averaged 10.2 solver iterations. Mean and stacking used zero validation
labels for selection. Their reported fit/validation losses were measured
separately. GP started from eight single sensors and charged all objective
calls, including reevaluated survivors. Its four finalists were distinct
programs from the fit archive. Each matching-based winner used 15 nodes and
the GP winner averaged 8.04.

Average fit-search and selection times per world were 38.26 and 0.42 ms
for top-two, 24.36 and 0.42 ms for random-two, 23.86 and 0.40 ms for
top-one, and 84.53 and 1.12 ms for GP. Top-two's search time includes 0.47
ms for enumerating and ranking matchings and 27.04 ms for constructing the
240 trees. The stacker fit took 2.68 ms. These shared-CPU measurements
include interpreter overhead. Full timing fields are in the CSVs.

The shift increased independent sensor noise from σ=0.60 to 1.25 and kept
the same latent graph. The top-two advantage over random-two was 0.02804
Brier (95% interval 0.02527 to 0.03082). The paired intervals average
the two related worlds first, then use 12 independent bank means. Evidence
for transfer is confined to this additive alternating-cycle construction,
whose two surviving matchings fit the explicit grammar. Other nuisance
graphs, group counts or target functions may require more proposals.

## Reproduction

Use the OKAPI lab checkout and FusionBench commit `4cf4810` on PYTHONPATH.
The two worlds share fit, validation and test sensor matrices. Only each
world's own labels enter its model fitting. Both worlds' predictors are
frozen before the separate IID and shifted test draws. Four tests cover
the graph, bit-identical swapped-world inputs, ranking and fitting. The
driver test checks which label arrays reach each fit and when tests are drawn.
A fresh-process replay of banks 41200 and 41211 exactly reproduced all
non-timing fields in 24 fit, 12 recovery and 56 test records.

```bash
export PYTHONPATH="$PWD:$FUSIONBENCH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python -m pytest -q experiments/ambiguous_groups
python -m experiments.ambiguous_groups.run --out /tmp/ambiguous-groups
```

[Registered protocol](protocol.md) · [Worlds](worlds.py) · [Methods](methods.py) ·
[Driver](run.py) · [Training and selection](results/fits.csv) ·
[Grouping recovery](results/recovery.csv) · [Held-out measurements](results/test.csv) ·
[Paired contrasts](results/contrasts.csv)
