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
