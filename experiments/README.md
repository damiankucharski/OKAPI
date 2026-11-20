# Inspecting an ensemble and training a single circuit

Run from this checkout with the companion FusionBench research branch importable. Both experiments use the NumPy backend of OKAPI. The circuit driver uses CPU PyTorch for the supervised network. Runs can be reproduced with these commands.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR
python experiments/operator_equivalence.py --output operator_equivalence.csv
python experiments/raw_circuit.py --output raw_circuit.csv
python experiments/rng_audit.py > rng.json
```

## Functional equivalence and attribution

On the Compound generator, a search over a bounded library of actual OKAPI programs selected the same logical composition on 12 independent seeds. The mean accuracy on 8,192 held-out items per seed was 85.96%. The mean counts of repairs and harms attributed to its max update were 857 and 168 relative to bypassing the update. Inserting a redundant second max preserved every prediction. Bypassing either copy on its own then yielded zero repairs and zero harms. Bypassing both copies reproduced the original effect.

A coherent perturbation of all occurrences of one provider offers a comparison across programs that compute the same function. Under a weaker rescue signal, the loss in accuracy from permuting rescue predictions decreased from 7.06 to 0.98 percentage points. A permutation can produce unusual combinations when providers are correlated. Counts measure the specified intervention on the fitted function.

## A single circuit from the program

On TwoPathways, an exhaustive search over 81 OKAPI programs used a separate set of 2,048 labeled examples with access to synthetic concept experts. It recovered the two-path structure on all 12 seeds. A fresh 36-parameter network trained on mixed raw sensor inputs and final labels achieved 85.67% with 2,048 student labels. The analytic expert ensemble achieved 86.15%. A dense tanh MLP given the source and student task labels achieved 85.22%. With only 256 student labels, pooling all available raw examples favored the dense network. All networks used a fixed optimization budget. Source supervision and the search space are part of the transfer method's cost.

## Seed isolation

The audit compares a single seeded GP trajectory with the same run interleaved with another optimizer and unrelated NumPy draws. Eight generation fingerprints include the population, fitness and predictions. The saved before/after files use the same explicitly ordered model inputs. Ordinary single-run reproducibility and variation between different valid GP seeds are separate properties. The standard benchmark for the latter is in the companion FusionBench branch.

[Measured outputs](results/) · [FusionBench research branch](https://github.com/damiankucharski/fusionbench/tree/research/ensemble-mechanisms-stability)

## Selecting a tree when the mix of operating conditions changes

If the source validation set identifies two operating conditions, estimate the mean loss of each candidate tree separately in each. The function `okapi.selection.estimate_context_prevalence` accepts the mean of unlabeled target features and the two source feature matrices. The FusionBench experiment uses pairwise disagreements among base experts. With an estimated changed-context frequency `p`, rank the trees by `(1-p) * reference_loss + p * changed_loss`. This requires stable within-context feature and outcome laws and distinguishable context feature means. The measured benefit and the cost under an unchanged distribution are recorded in the companion seed-selection study.
