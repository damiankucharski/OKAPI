# Trim-one pooling under intermittent provider faults

## Frozen design

Compare fixed aggregators on seven independent binary providers from FusionBench
`Committee(k=7, accuracy=.75)`. The proposed OKAPI node removes the smallest
and largest positive-class score in each row, then averages the other five.
Its controls are an actual OKAPI mean tree, the coordinate median, and a
supervised logistic stacker. The stacker selects probability or logit features
and C in {0.1, 1, 10} by clean validation Brier, resolving ties in listed order.
No other settings will be selected. The experiment measures a fixed tree.

Use 16 independent seed banks (2300–2315). Each bank draws independent
training (512), validation (256), IID test (4096) and shifted-test (4096)
examples. Generate independent streams for the corruptions. Train the stacker
only on clean training examples and freeze it after clean validation.
The fixed methods use zero training and validation labels. Use identical
provider outputs for all methods within each bank and condition.

Half the shifted-test rows experience a stuck-output fault, selected by
independent fair coins. On each affected row, rank the seven providers in
random order and choose a fresh common saturated bit (.001 or .999 with
equal probability, independent of the label). The minority condition replaces
the first provider in that order. The majority condition replaces the first
four with the same bit. Keep the same rows, provider order and bits in both
conditions. Also retain their clean counterfactual. As a separate boundary
control, retain only the score of the first randomly ranked provider on each
shifted-test row and replace the six others by .5. This tests informative
extremes and gives no method an indicator of the active provider.

Primary contrast is trim-one minus equal mean in full minority-shift test
Brier, with a two-sided paired 95% t interval across seed banks. Report
secondary accuracy, all Brier scores, per-bank degradation from each shifted
test's clean counterfactual, and the observed fraction affected. Inspect the
common-majority and singleton controls even if the primary contrast fails.
The paired contrasts reuse cases within each bank. Test labels enter only
scoring. No deployment labels, validation on faults or GP generations.

Record training, validation and inference wall times on single-thread CPU.
Log unique labels, number of fits, validation scoring passes and solver
iterations per bank. Include the cost of comparing all six stackers.
Inference timings include construction and evaluation of the fixed trees,
and exclude data simulation and metric calculation. They are descriptive
on the shared machine. Each method consumes seven provider outputs per row.
