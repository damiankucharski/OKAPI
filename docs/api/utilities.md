# Utilities API

This section documents the utility components of OKAPI.

## Fitness Functions

Functions for evaluating the fitness of trees.

```python
from okapi.fitness import average_precision_fitness, roc_auc_score_fitness
```

::: okapi.fitness
    options:
      show_source: true

## Callbacks

The callback system allows customizing the evolutionary process.

```python
from okapi.callback import Callback
```

::: okapi.callback.Callback
    options:
      show_source: true

## Visualization

Functions for visualizing trees.

```python
from okapi.draw import draw_tree
```

::: okapi.draw.draw_tree
    options:
      show_source: true

## Postprocessing Functions

Functions for postprocessing tree outputs.

```python
from okapi.functions import scale_vector_to_sum_1, set_multiclass_postprocessing
```

::: okapi.functions
    options:
      show_source: true

## Other Utilities

Additional utility functions.

```python
from okapi.utils import Pickle, first_uniques_mask
```

::: okapi.utils
    options:
      show_source: true
