# Propose a new threshold in an existing GP tree

In the preceding switch benchmark, 13 of 24 fit-winning trees contained no threshold to tune. A new optional proposal adds one real CLOSE or FAR operation and an additional provider at a terminal value in the current program. The previously available random append mutation also can create these operators. This proposal keeps one attachment site fixed while comparing incoming providers and scalar settings on fit data. It returns an ordinary OKAPI tree that can enter normal fitness selection and subsequent evolution. [Previous fit and test results](../threshold_refinement/README.md)

## Controlled structural change

Pick one feasible terminal occurrence uniformly using the specified seed; only fit provider identities and tree geometry are needed. Preserve the existing provider at the chosen terminal as the gate's first operand. Shuffle the remaining provider IDs in the supplied mapping's stable order using the same random stream. The caller can search the first new provider or all candidates in that order. Both policies try CLOSE and FAR for each candidate, and either use the fixed threshold .5 or fit it with the existing downstream-Brier coordinate operation. The same seed fixes site and ranking across the four arms. Repeated occurrences of a provider elsewhere in the winning tree stay in place. The proposal changes exactly two nodes in size when accepted. It preserves all existing branches and accepts only strict improvement in full-precision training Brier.

For s incoming providers, score one incumbent plus 2s whole trees with fixed thresholds. When fitting thresholds, each of 2s inserted gates requires four full evaluations: the proposed structure at .5, two forced operand predictions, and a normal reevaluation of the fitted threshold. Thus the fit route uses 1+8s full evaluations. Charge the event solver's examined scalar settings and diagnostic event counts separately. On six anonymous providers, searching all five alternatives costs 11 or 41 full forwards and involves 10 structures. A structurally infeasible edit costs one incumbent evaluation. The proposal uses neither validation nor test inputs. [Fixed-genotype coordinate fitting](../threshold_refinement/README.md)

The shared FusionBench experiment uses the first complete ordinary generation at or beyond 1,200 scored trees. Each trial tests four insertion arms, normal GP continuations at two cost levels, and the effect of offering the fitted proposal to the actual breeding population. The archive-only continuation offers the same frozen proposal for final selection while keeping it out of breeding. Both continue for 300 evaluations with matched starts. The public arithmetic mean, ranked-prefix Top-N and regularized logit stack use matched source data; held-out tests span true switching, attenuated switching and the linear control. FusionBench owns the independent draws and scoring.

The frozen cross-repository protocol defines population admission and the archive-only comparison, paired inference, subgroups with and without a starting gate, and directional robustness. [Independent FusionBench protocol](https://github.com/damiankucharski/FusionBench/blob/bd1ec0b/docs/ensemble-study/search-regret/native-graft-protocol.md)

```python
from fusionbench.eval.gate_events import fit_threshold
from okapi.threshold_graft import propose_threshold_graft

result = propose_threshold_graft(
    fit_champion, fit_providers, fit_labels,
    seed=293, search_operands=True, fit_threshold=fit_threshold,
)
candidate = result.tree
```

For a fixed-threshold control, omit the solver. Set search_operands to False to offer only the first new provider. The result records the terminal’s ordered child-index path, its original ID, the seeded provider order, accepted direction/child/threshold, number of structures, full forward evaluations, threshold settings and events. Ties preserve the incumbent then the first strict winner. It uses a private fit-bound copy and never mutates the input tree. Require a pure, deterministic, rowwise binary predictor and postprocessor, as in the coordinate solver. [Production operator](../../okapi/threshold_graft.py) · [Native mutation, population, replay and cost tests](../../test/test_threshold_graft.py)

## Research provenance

Kommenda, Burlacu, Kronberger and Affenzeller’s *Parameter identification for symbolic regression using nonlinear least squares* motivated comparing fixed-structure tuning against a structural search. They studied local fitting of numerical constants with Levenberg–Marquardt in two GP variants over 94 symbolic-regression tasks. Their sinusoidal example also shows sensitivity to the optimizer’s initial value. The published evidence concerns continuous parameters in symbolic regression. [Original publication](https://doi.org/10.1007/s10710-019-09371-3) · [FusionBench review of its scope](https://github.com/damiankucharski/FusionBench/blob/72fc12a/docs/ensemble-study/literature/2026-11-21.md)

The threshold graft, its four ablation arms, and the decision to test archive-only against breeding are OKAPI/FusionBench constructions. The observed absence of a threshold in 13/24 earlier champions comes from our previous synthetic benchmark. The new panel uses independently generated data and measures the consequences of this specific insertion policy. [Original benchmark evidence](https://github.com/damiankucharski/FusionBench/blob/f10d150/docs/ensemble-study/search-regret/native-refinement/README.md) · [New joint protocol](https://github.com/damiankucharski/FusionBench/blob/bd1ec0b/docs/ensemble-study/search-regret/native-graft-protocol.md)

## Independent experiment

On the stationary switch task (12 independent banks × two GP starts), the original champion had conditional excess Brier .02050. Immediate insertion with the first shuffled incoming provider and a fixed .5 threshold gave .02059, versus .01965 when searching all five candidate providers. Fitting the threshold gave .02073 for the first provider and .01791 for all five. The fully fitted edit on its own did not resolve an advantage over plain GP at the same candidate-credit budget.

After 300 ordinary evaluations, admitting the fitted graft through native survivor selection yielded **.00539** conditional excess Brier against **.01791** with the identical proposal retained only in the selection archive. Plain GP given the complete candidate-credit budget reached **.01545**, with a preregistered active-minus-plain difference of −.01006 [−.01772, −.00241]. The fully searched one-gate control reached .00159. All-provider mean, ranked-prefix Top-N and regularized logistic stacking reached .09053, .04029 and .02085 respectively. Mean observed test Brier was .13886 for the active proposal, .15158 for archive-only and .14911 for candidate-matched GP. These measures summarize independent synthetic FAR-switch data.

Fifteen of 24 fit-champion proposals entered the breeding population. Eight initial champions had no threshold; seven of those eight proposals entered. In that subgroup active search reached .00678 against .02070 for candidate-matched GP. A post hoc check found the proposal's direction and exact threshold in 14 of the 15 final winners following admission. The saved programs record parameter retention. Attribution to individual mutations or crossovers would require full ancestry records.

At an average generation-boundary budget of 1,207.71 ordinary fit assessments, the full edit took 41 full-tree assessments and 2,216.17 internal threshold candidates. After continuation this amounts to 1,548.71 actual full forwards or 3,764.88 evaluations if each internal candidate is charged a full forward. Forty-one forward evaluations here use the same 512 fit rows. When the true switch was removed at test time, the difference versus candidate-matched GP crossed zero (+.00078 [−.00137, +.00291]). On the independently trained logistic control, stacking attained .00199 excess against .00967 for active GP. Population admission helped on the stationary-switch task. The linear control and the abolished switch limit extrapolation to other label laws.

Independent reconstruction recovered 18,480 held-out and development Brier values and 192 graft neighborhoods. A separate pass replayed 253,152 native objective evaluations with their original batching. [Full paired FusionBench results, ablations and reproduction commands](https://github.com/damiankucharski/FusionBench/blob/9de06f04a07e4cdeb10d07af42b0f2fb7b12931d/docs/ensemble-study/search-regret/native-graft/README.md) · [Central literature record and its relationship to this experiment](https://github.com/damiankucharski/FusionBench/tree/ff9e87a352eaa28ae18921635abc14eca35af4b5/docs/ensemble-study/literature)
