# OKAPI

Automated ensemble design through genetic programming for classification tasks.

## Description

OKAPI evolves tree-based fusion structures that combine predictions from multiple machine learning models. It uses genetic programming with crossover and mutation operations to search for optimal ensemble configurations, balancing predictive performance with model complexity through Pareto optimization.

## Installation

```bash
git clone https://github.com/damiankucharski/OKAPI.git
cd OKAPI
uv sync --all-groups --all-extras
```

Or with pip:

```bash
pip install -e .
```

## Usage

```python
from pathlib import Path

import numpy as np

from okapi import Okapi
from okapi.pareto import minimize
from okapi.selection import select_tree

# Each entry in validation_groups / test_groups is (predictions_by_id, labels).
# IDs are fit filenames, including extensions. Here scores and labels have
# matching shapes and represent the probability of the positive class / 0 or 1.
def brier(scores, labels):
    return np.mean((scores - labels) ** 2)

# A prediction directory contains one .npy per provider.
providers = sorted(Path("path/to/fit/predictions").glob("*.npy"))
population_size = min(100, len(providers))
candidates = []
for seed in (11, 22, 33):
    ensemble = Okapi(
        preds_source=providers,
        gt_path="path/to/fit/ground_truth.npy",
        population_size=population_size,
        population_multiplier=2,
        tournament_size=max(1, min(5, population_size // 2)),
        backend="numpy",
        seed=seed,
        objective_functions=(brier,),
        objectives=(minimize,),
    )
    ensemble.train(iterations=50)
    candidates.extend(ensemble.pareto_trees)

# Score all candidates on the same independent validation groups. The selector
# favors small trees within one paired standard error of the best mean loss.
selection = select_tree(candidates, validation_groups, brier)
winner = candidates[selection.index]
test_brier = [
    brier(winner.do_pred_on_another_tensors(current_tensors=scores), labels)
    for scores, labels in test_groups
]
```

## Features

- Tree-based ensemble representation
- Genetic programming with crossover and mutation
- Pareto optimization for multi-objective optimization
- Multiple backend support (NumPy, PyTorch)
- Fusion operators: mean, weighted mean, min, max, threshold-based selection
- Extensible callback system for monitoring evolution
- Tree visualization

## Documentation

Full API documentation: [https://damiankucharski.github.io/OKAPI/](https://damiankucharski.github.io/OKAPI/)

## Algorithm research

The [algorithm lab](experiments/algorithm-lab.md) records opt-in changes, comparison budgets and measured generalization across independent data draws.

## Paper

OKAPI: Automated Ensemble Design Through Genetic Programming

## License

MIT License - see [LICENSE](LICENSE) file for details.
