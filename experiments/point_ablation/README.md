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

## Separating the refresh draw from the next native decision

The original `refresh` implementation exactly copied the post-proposal MT19937 state. We detected a consequence by inspecting the native variation loop while the registered panel was running. A CLOSE/FAR factory draws `t` uniformly. The very next native draw tests whether to mutate all parameters. The two draws are numerically identical when the shadow is a direct copy, so after a refresh the native parameter mutation occurs exactly when `t < tree.mutation_chance`. The first sampled weight of a weighted mean has the same issue. No result from an arm involving refresh had reached the authors of this sensitivity check when it was designed. The original policy and four-arm estimates retain their original definitions. [Factory and native draw](../../okapi/node.py) · [Variation loop](../../okapi/okapi.py)

`refresh_independent` is a separately identified sensitivity control. It initializes the original family with a private MT19937 seeded from all 256 bits of SHA-256. The hash covers a fixed purpose string, the algorithm name, the full little-endian uint32 MT state, the state index, Gaussian-cache flag and cached float64. This removes reuse of the next native uniform while preserving reproducibility and the ordinary post-proposal stream. Subsequent genotypes may diverge. In a one-gate test with 256 fixed seeds, the original cutoff equaled the next native draw 256 times. With the separate stream it equaled it zero times. Splitting each number at 0.5 gave counts 139/0/0/117 in the four cells for the direct copy, and 72/71/67/46 for the separate stream. The next native draw matched across policies at every seed. These are mechanical unit checks. The changes in predictive risk require the independently registered sensitivity study. [Replay test](test_moves.py) · [Implementation](moves.py)

## Independent results

On all 19 registered scenarios, the actual family replacement reduced mean IID full-class Brier from **.26753** with slot-matched identity to **.25551** after 1,200 fitness assessments. The paired difference across six fresh data replicates was **−.01202 [−.01756, −.00648]**. Seventeen of the 19 individual point estimates favored replacement. At ten complete generations the comparison was −.01219 [−.01658, −.00780]. On the nine predefined shifts the mean difference was −.00100 [−.00479, +.00279]. The shift result leaves the overall direction uncertain. No global tree optimum is known for this registry. [Independent results, all scenario effects, uncertainty and costs](https://github.com/damiankucharski/FusionBench/blob/b93281d79c48b0190d8591b865f31595c94e0828/docs/ensemble-study/mutation-opportunity/README.md)

| Fit-selected method | Mean IID full Brier, 19 synthetic scenarios | Pooled full Brier, 178 held-out Wine specimens |
|---|---:|---:|
| All-provider arithmetic mean | .37857 | .04971 |
| Ranked-prefix Top-N | .33434 | .04633 |
| Regularized logistic stack | .22610 | .04276 |
| Original stock GP | .26485 | .05104 |
| Same opportunity, preserve original tree | .26753 | .05254 |
| Redraw original family, independently seeded | .26684 | .05254 |
| Replace operator family | .25551 | .05096 |

The true retype versus independently seeded refresh gave −.01133 [−.01671, −.00595] across the synthetic panel. Only 579 of 6,629 refresh proposals changed parameters. The opportunity to do so was absent in 134 of 228 identity trajectories and all five Wine folds. Resampling the old family therefore addresses a limited set of cases in this mixture. The Wine retype-versus-identity difference was −.00157, favorable in five of five folds. All three inexpensive baselines scored better on Wine. The fixed 19-scenario improvement measures the value of allowing the different family into ordinary evolution, including later changes in its population. It does not identify which operator ultimately supplied a winning prediction. [Publicly frozen independent-refresh addendum](https://github.com/damiankucharski/FusionBench/blob/07a59cb/docs/ensemble-study/mutation-opportunity/refresh-addendum.md) · [Unfiltered comparison and independent checks](https://github.com/damiankucharski/FusionBench/tree/b93281d79c48b0190d8591b865f31595c94e0828/docs/ensemble-study/mutation-opportunity)
