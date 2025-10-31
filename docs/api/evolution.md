# Evolution API

This section documents the evolutionary components of OKAPI.

## Population

Functions for initializing and managing populations of trees.

```python
from okapi.population import initialize_individuals, choose_n_best, choose_pareto, choose_pareto_then_sorted
```

::: okapi.population
    options:
      show_source: true

## Crossover

Functions for performing crossover between trees.

```python
from okapi.crossover import crossover, tournament_selection_indexes
```

::: okapi.crossover
    options:
      show_source: true

## Mutation

Functions for mutating trees.

```python
from okapi.mutation import append_new_node_mutation, lose_branch_mutation, new_tree_from_branch_mutation, get_allowed_mutations
```

::: okapi.mutation
    options:
      show_source: true

## Pareto Optimization

Functions for Pareto optimization and visualization.

```python
from okapi.pareto import paretoset, minimize, maximize, plot_pareto_frontier
```

::: okapi.pareto
    options:
      show_source: true
