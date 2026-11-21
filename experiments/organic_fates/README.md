# Ordinary operator edits during multigeneration search

After 1,800 native fitness evaluations in the independently registered comparison, changing an existing operator family gave **.05957** mean excess Brier on the unseen compositional population. Changing the provider at one child while preserving its own subtree gave **.06259**. The primary paired difference was **−.00302 [−.00923, +.00320]** across 12 data banks with two starts each. Changing both gave .05711 and ordinary unmodified GP gave .06228. The known seven-node tree has zero excess. One provider-only start recovered the exact tree. The 600-evaluation and equal-generation checkpoints also failed to resolve an ordering between point-mutation policies.

| Stationary conditional excess | Ordinary GP | Type | Provider | Both | Mean | Ranked-prefix Top-N | Regularized stack |
|---|---:|---:|---:|---:|---:|---:|---:|
| Two hidden MIN paths combined by MAX | .06228 | .05957 | .06259 | .05711 | .09305 | .06878 | .05587 |
| Logistic control | .00815 | .00629 | .00650 | .00696 | .07060 | .00626 | .00254 |

On interior-only composition examples the corresponding GP, type, provider and both errors were .02109, .01980, .01945 and .01980, versus .02509 for stacking. The protocol used 512 fit labels and the same 256 validation rows, held out 8,192 independent examples per evaluation regime, and charged each complete or partial training evaluation. Fit and validation archived winners are reported separately. An independent replay reproduced 345,600 native evaluations. [FusionBench prospective comparison, cross-law results and timing](https://github.com/damiankucharski/FusionBench/blob/d00e1a82e3d7590bba25dc34dc60d3ca3c3afd54/docs/ensemble-study/search-regret/organic-composition/README.md)

## Retrospective count by introduced family

We re-read the frozen native event ledger after the trial. We included 12,689 genuine type edits across the type-only and combined arms whose generation finished selection within 1,800 fit calls. This cutoff provides a first-survival outcome for each submitted event. Different destination families have different counts because a point edit always excludes the current family. For every destination we distinguish submitting the new object, actually evaluating it after duplicate removal, and keeping that exact object in the next population.

| Compositional law, type-only | Submitted | Scored | Survived first selection |
|---|---:|---:|---:|
| MEAN | 294 | 264 | 37 |
| MIN | 501 | 493 | 27 |
| MAX | 603 | 601 | 24 |
| WEIGHTED_MEAN | 486 | 486 | 34 |
| CLOSE | 529 | 529 | 34 |
| FAR | 563 | 563 | 17 |

In type-only runs, 2,936 of 2,976 submitted edits received a fitness score, and 173 survived first selection. The compositional target contains MIN and MAX, but their first-selection rates among evaluated edits were 27/493 and 24/601. MEAN survived 37/264 in these same runs. The original study counted about as many first-surviving objects per start under stock GP (7.7), type-only (7.2) and provider-only (7.7), despite the additional unique structures that edits reached. These descriptive counts mix sites and predecessor types; they provide no estimate of the benefit of preferring an operator in future search.

Only two evaluated type-only submissions and one combined submission on the compositional law had a complete program structurally equal to their run's eventual fit champion. All other descendant edits remain possible. The ledger records the actual initial survival of each submitted object. Full-program key recurrence alone cannot identify which subtrees passed through later crossover or mutation. [Pooled counts and bank-averaged rates](summary.csv) · [Per-bank counts for both tasks](by_bank.csv) · [Audit source](audit.py) · [Frozen event ledger](https://github.com/damiankucharski/FusionBench/blob/d00e1a82e3d7590bba25dc34dc60d3ca3c3afd54/docs/ensemble-study/search-regret/organic-composition/events.csv.gz)

The six native operator families and selection policy are unchanged unless the intervention is enabled. OKAPI's owner had deliberately deferred point mutation when completing the dissertation experiments. Langdon, Barrett and Buxton previously used function replacement that retained argument connections in evolved classifier fusion; their experiments also used other genetic changes. Our registered four-way comparison and the retrospective counts above are OKAPI/FusionBench measurements. [Original GP fusion paper, section 5.5](https://doi.org/10.1007/978-1-4471-0123-9_51) · [Native implementation](../../okapi/mutation.py) · [Exact one-operation bounds](../compositional_control/README.md)
