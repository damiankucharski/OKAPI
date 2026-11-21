# Fit one threshold inside a native tree

The first native-switch benchmark found a gap between the best Brier obtained by ordinary GP and a feasible, fitted one-gate tree on a planted switch. A fixed genotype may already contain the right gate with an unsuitable threshold; it may also read the wrong operands. The joint follow-up assigns the production wrapper to OKAPI and the independent seeded GP, ablations and equal-cost continuations to FusionBench. [Original switch study](https://github.com/damiankucharski/FusionBench/blob/cf2a31ef1e103b76d7950b8ba9341a8a5181efc1/docs/ensemble-study/search-regret/native-switch/README.md)

The opt-in coordinate operation binds a private copy of the tree to the provided fit tensors. For each operand of the chosen native CLOSE or FAR gate, force that gate to return that operand on all rows, then score the final tree probability on each row. The gate's first operand is its parent's value immediately before the chosen operator, including any preceding operations. The resulting N-by-M table contains the final squared losses for choosing each operand. The gate's own prediction can have the opposite preferred value from the whole program, as when a downstream operation complements its input. This method assumes independent rowwise evaluation and deterministic binary probability operations. A caller supplies the threshold solver; FusionBench implements an event sweep and a native candidate scan for finer inputs. [Solver and exact scope](https://github.com/damiankucharski/FusionBench/blob/208f34169b168967605f3ec31b9bc72f9d9c20ae/fusionbench/eval/gate_events.py)

One coordinate fit charges M+2 full tree evaluations: the original Brier, M forced predictions, and ordinary reevaluation at the nominated threshold. Count solver thresholds, sorted row events and elapsed time separately for fair comparisons. Ties retain the incumbent. Selecting a threshold does not tune operand identities, the CLOSE/FAR direction or additional coordinates. The separate existing single-occurrence provider move preserves all thresholds. [Operand refinement](../../okapi/refinement.py)

The joint protocol freezes twelve fresh data banks, two GP starts, threshold-only, operand-only and combined steps. It charges original full fitness calls separately from solver candidates, including matched ordinary-GP continuations. All three inexpensive references use matched development labels. [Independent joint preregistration](https://github.com/damiankucharski/FusionBench/blob/9415861/docs/ensemble-study/search-regret/native-refinement-protocol.md)

```python
from fusionbench.eval.gate_events import fit_threshold
from okapi.threshold_refinement import refine_threshold_coordinate

result = refine_threshold_coordinate(
    tree, fit_tensors, fit_labels, operator_index=i, fit_threshold=fit_threshold
)
fitted_tree = result.tree
```

The integration regression checks CLOSE and FAR inside a native tree with an upstream arithmetic mean, three selectable operands and downstream XOR with one (which reverses the gate's prediction). Against an exhaustive 129-position grid covering all attainable decisions on this dyadic fixture, both fitted native trees achieved the same minimum training Brier. Each check used five ordinary or forced full tree forwards. [Full-tree regression using the actual FusionBench event solver](../../test/test_threshold_fusionbench.py)
