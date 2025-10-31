# Nodes API

Nodes are the fundamental building blocks of trees in OKAPI. This section documents the different types of nodes available.

## Base Node Class

The `Node` class is the base class for all node types in OKAPI.

```python
from okapi.node import Node
```

::: okapi.node.Node
    options:
      show_root_heading: false
      show_source: true

## Value Node

The `ValueNode` class represents a node that holds tensor data (model predictions).

```python
from okapi.node import ValueNode
```

::: okapi.node.ValueNode
    options:
      show_root_heading: false
      show_source: true

## Operator Node

The `OperatorNode` class is the base class for all operation nodes that define how to combine tensor data.

```python
from okapi.node import OperatorNode
```

::: okapi.node.OperatorNode
    options:
      show_root_heading: false
      show_source: true

## Specific Operator Nodes

Various specific operator node implementations are provided:

### Mean Node

::: okapi.node.MeanNode
    options:
      show_root_heading: false
      show_source: true

### Weighted Mean Node

::: okapi.node.WeightedMeanNode
    options:
      show_root_heading: false
      show_source: true

### Min Node

::: okapi.node.MinNode
    options:
      show_root_heading: false
      show_source: true

### Max Node

::: okapi.node.MaxNode
    options:
      show_root_heading: false
      show_source: true

## Utility Functions

::: okapi.node.check_if_both_types_values
    options:
      show_source: true

::: okapi.node.check_if_both_types_operators
    options:
      show_source: true

::: okapi.node.check_if_both_types_same_node_variant
    options:
      show_source: true
