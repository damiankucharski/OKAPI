# Separate an inherited gate from population perturbation

A previous joint experiment found that placing a fit-selected graft in the population helped stationary FAR-switch predictions after 300 more native evaluations. Holding the exact same program in a selection archive gave a smaller gain. A parent introduction can also displace one resident, change survivor ordering, or change later random choices. Exact cutoff equality in saved trees alone could not distinguish a copied gate from independent recurrence. [Previous OKAPI/FusionBench evidence](https://github.com/damiankucharski/FusionBench/blob/9de06f04a07e4cdeb10d07af42b0f2fb7b12931d/docs/ensemble-study/search-regret/native-graft/README.md)

## Native interventions

At the originally grafted terminal there is an anchor provider, followed by a single two-operand CLOSE/FAR gate with a new terminal input. Control copies can independently reset that threshold to .5 or replace the new input with a duplicate of the anchor. With identity node postprocessing, selecting either copy of the anchor produces its original probability exactly, even if the fitted cutoff remains in the neutral tree. The fitted scalar can matter later if genetic operations separate the operands. The fixed-cutoff duplicate therefore provides a separate control. Actual fit scores are recalculated after the intervention.

The passive tag is assigned to the gate occurrence in a private copy of the admitted parent. It propagates with native subtree copying and crossover, follows the gate through scalar mutations, and disappears when the occurrence is removed. Report the marker separately from exact direction/cutoff matches and the direct ordered operands. Equal provider IDs cannot establish that preceding updates or operand subtrees still produce the same probabilities. An independently constructed threshold with the identical value has no marker. Shared archive entries remain unmarked before admission. The selection engine and structural duplicate key ignore the marker.

The independent FusionBench design fixes the actual admission event across arms. In the four content arms the selected pair or duplicate is crossed with the fitted cutoff or .5 and placed in the same slot, with the other survivors in the same order. An order-only arm puts the displaced original parent in that slot, and a pure archive control preserves the original population. The ordinary GP comparison receives the designated proposal budget. All arms have the same proposal choices available for archive selection. They start from the same RNG state at the intervention boundary when the comparison calls for it. Distinct parent programs can change future operations and random-number consumption. Paired inference includes trials in which no native admission occurred.

## Research provenance

Kommenda, Burlacu, Kronberger and Affenzeller compared local numerical-parameter fitting in symbolic regression. They used Levenberg–Marquardt and reported dependence on starting parameters in a nonlinear example. This motivated separating fitted constants from a change in structure. The gate-marking mechanism, same-slot controls, artificial switch laws and results in this series are constructions and measurements of OKAPI and FusionBench. [Parameter identification for symbolic regression using nonlinear least squares](https://doi.org/10.1007/s10710-019-09371-3) · [Central project evidence index](https://github.com/damiankucharski/FusionBench/tree/ff9e87a352eaa28ae18921635abc14eca35af4b5/docs/ensemble-study/literature)

[Prospective six-arm protocol, conditional and all-run analyses](https://github.com/damiankucharski/FusionBench/blob/8df5db69e8b92964fbf69daa1eeee21dc48d3c7a/docs/ensemble-study/search-regret/matched-graft-protocol.md)

```python
from okapi.gate_lineage import gate_descendants, mark_gate_origin
from okapi.threshold_graft import threshold_graft_variant

fixed = threshold_graft_variant(proposal.tree, proposal.site_path, reset_threshold=True)
neutral = threshold_graft_variant(proposal.tree, proposal.site_path, duplicate_anchor=True)
marked = mark_gate_origin(proposal.tree, (*proposal.site_path, 0), token="offered")
hits = gate_descendants(marked, token="offered")
```

The construction functions return private copies. A marked parent can be used in the ordinary population. Marking consumes no RNG draws; on an independent nine-generation replay both marked and unmarked versions of the same parent produced all 75 scored programs in the same order with identical scores, survivors and RNG states. This is a transparency check for identical parent programs. [Lineage operations](../../okapi/gate_lineage.py) · [Control copies](../../okapi/threshold_graft.py) · [Lineage tests](../../test/test_gate_lineage.py) · [Control tests](../../test/test_threshold_graft_variants.py)
