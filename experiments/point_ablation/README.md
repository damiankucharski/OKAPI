# Control the opportunity to change an operator

After the canonical regression screen, an unanswered question is whether a fourth category of structural mutation helps by preserving a candidate that a more disruptive edit would have changed. The elected source tree and native mutation dispatcher can remain the same while changing what is returned at that fourth opportunity. The stock arm has three structural categories, depending on eligibility. The three point-enabled arms add the stock arity-preserving point replacement to the eligible categories with its normal sampling probability. [Previous multi-scenario comparison](../standard_point_control/README.md)

When the dispatcher selects the added operation, all three arms first materialize the ordinary real type replacement, including selection of the slot, choice of a different family and every native parameter-initialization draw. The full arm returns that program. The identity arm returns a private search-order-preserving copy of the input tree. The refresh arm retains the selected slot's old family and draws its initialization parameters anew with the native factory. It uses a clone of the ordinary RNG state captured immediately after the full proposal, which isolates those additional draws. Other native mutations, subsequent parameter mutation, crossover, scoring and survivor selection continue as usual. A parameter-free family in the refresh arm yields an identical structural program. Materializing a shadow program costs construction time but no additional fitness assessment. The policies can diverge as soon as different programs enter a generation. This four-arm comparison cannot hold every subsequent proposal or random draw fixed.

Langdon, Barrett and Buxton used same-arity exchange in neural ensemble GP alongside other operators. The same-opportunity identity and parameter-refresh controls described here are this project's design to evaluate the specific alternatives in OKAPI. [Original fusion paper, section 5.5](https://doi.org/10.1007/978-1-4471-0123-9_51) · [Research controls](moves.py) · [Native targeted initialization](../../okapi/mutation.py) · [RNG and complete native-generation replay checks](test_moves.py)

```python
from experiments.point_ablation.moves import MatchedPointControls
from okapi import Okapi

class Observed(MatchedPointControls, Okapi):
    pass

# Supply real tensors, labels, and the usual engine options.
engine = Observed(**gp_options, operator_point_mutation=True)
engine.point_policy = "refresh"  # Other matched policies are "retype" and "identity".
events = []
engine.point_event_callback = events.append
engine.run_iteration()
```

For a stock comparison set operator_point_mutation=False. The callback receives the exact site, original family, materialized proposal family, returned family and whether the complete returned program differs from the original. It receives no labels. A single-draw regression checks that all three policies consume the same ordinary stream on the same parent. Two independent 40-generation replays check that adding the observation mixin to stock-disabled and stock-enabled runs preserves all selected populations, fitness arrays and native RNG states.

The independently registered follow-up uses fresh cohorts for the full 19-scenario regression panel, checkpoints at 1,200 true fitness assessments and ten completed native generations, and a separate Wine multiclass task with out-of-fold base predictions. Fit, validation, IID and predefined synthetic shifts are distinct; real Wine has its own independent outer splits. The primary contrast is real retyping versus identity at the same opportunity. [Frozen synthetic and Wine evaluation protocol](https://github.com/damiankucharski/FusionBench/blob/f7638e4a907ac5d0fa1ff6feaa3b287c0692b0a9/docs/ensemble-study/mutation-opportunity/protocol.md)
