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
from okapi import Okapi

# Initialize with model predictions and ground truth
ensemble = Okapi(
    preds_source="path/to/predictions/",
    gt_path="path/to/ground_truth.npy",
    population_size=100,
    population_multiplier=2,
    tournament_size=5,
    backend="numpy"
)

# Run evolution
ensemble.run(n_generations=50)

# Access best solution
best_tree = ensemble.population[0]
predictions = best_tree.root.value
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

## Paper

OKAPI: Automated Ensemble Design Through Genetic Programming

## License

MIT License - see [LICENSE](LICENSE) file for details.
