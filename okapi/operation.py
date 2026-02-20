from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np
from loguru import logger

from okapi.globals import BACKEND as B
from okapi.lib_types import Tensor

if TYPE_CHECKING:
    from okapi.node import OperatorNode, ValueNode


class Operation:
    """Base class for operator strategies used by OperatorNode."""

    def op(self, x: Tensor) -> Tensor:
        return x

    def on_child_added(self, node: "OperatorNode", child: "ValueNode") -> None:
        pass

    def on_child_removed(self, node: "OperatorNode", child: "ValueNode", index: int) -> None:
        pass

    def on_child_replaced(self, node: "OperatorNode", old: "ValueNode", new: "ValueNode") -> None:
        pass

    def on_before_calculate(self, node: "OperatorNode") -> None:
        pass

    def copy(self) -> "Operation":
        return self.__class__()

    @property
    def code(self) -> str:
        return "OP"

    def __str__(self) -> str:
        return "Operation"

    @classmethod
    def create_node(cls, children: Optional[Sequence["ValueNode"]] = None) -> "OperatorNode":
        from okapi.node import OperatorNode

        return OperatorNode(cls(), children)


class MeanOp(Operation):
    def op(self, x: Tensor) -> Tensor:
        return B.mean(x, axis=0)

    @property
    def code(self) -> str:
        return "MN"

    def __str__(self) -> str:
        return "MeanNode"


class MaxOp(Operation):
    def op(self, x: Tensor) -> Tensor:
        return B.max(x, axis=0)

    @property
    def code(self) -> str:
        return "MAX"

    def __str__(self) -> str:
        return "MaxNode"


class MinOp(Operation):
    def op(self, x: Tensor) -> Tensor:
        return B.min(x, axis=0)

    @property
    def code(self) -> str:
        return "MIN"

    def __str__(self) -> str:
        return "MinNode"


class WeightedMeanOp(Operation):
    def __init__(self, weights: Optional[List[float]] = None):
        logger.debug(f"Creating WeightedMeanOp with {len(weights) if weights else 0} weights")
        self._weights: List[float] = weights if weights is not None else [1.0]
        logger.trace(f"WeightedMeanOp initialized with weights: {self._weights}")

    def op(self, x: Tensor) -> Tensor:
        weight_shape = (-1, *([1] * (len(x.shape) - 1)))
        w = B.reshape(self.weights, weight_shape)
        w = B.to_device(w, x)
        x = x * w
        x = B.sum(x, axis=0)
        return x

    @property
    def weights(self) -> Tensor:
        return B.tensor(self._weights)

    def on_child_added(self, node: "OperatorNode", child: "ValueNode") -> None:
        logger.debug(f"Adding child to WeightedMeanOp with current weights: {self._weights}")
        child_weight = np.random.uniform(0, 1)
        adj = 1.0 - child_weight

        logger.trace(f"Generated child weight: {child_weight}, adjustment factor: {adj}")
        for i, val in enumerate(self._weights):
            self._weights[i] = val * adj
        self._weights.append(child_weight)
        self._weight_sum_assertion()
        self._weight_length_assertion(node)
        logger.debug(f"Child added, new weights: {self._weights}")

    def on_child_removed(self, node: "OperatorNode", child: "ValueNode", index: int) -> None:
        logger.debug(f"Removing child from WeightedMeanOp with current weights: {self._weights}")
        adj = 1.0 - self._weights[index + 1]  # adjust for parent weight being first
        weight_removed = self._weights[index + 1]
        self._weights.pop(index + 1)

        logger.trace(f"Removed weight at index {index + 1} with value {weight_removed}, adjustment factor: {adj}")

        for i, val in enumerate(self._weights):
            self._weights[i] = val / adj

        self._weight_sum_assertion()
        self._weight_length_assertion(node)
        logger.debug(f"Child removed, new weights: {self._weights}")

    def on_child_replaced(self, node: "OperatorNode", old: "ValueNode", new: "ValueNode") -> None:
        self._weight_length_assertion(node)

    def on_before_calculate(self, node: "OperatorNode") -> None:
        self._weight_length_assertion(node)
        self._weight_sum_assertion()

    def copy(self) -> "WeightedMeanOp":
        return WeightedMeanOp([x for x in self._weights])

    @property
    def code(self) -> str:
        return "WMN"

    def __str__(self) -> str:
        return f"WeightedMeanNode with weights: {B.to_numpy(B.tensor(self._weights)).round(2)}"

    @classmethod
    def create_node(cls, children: Optional[Sequence["ValueNode"]] = None) -> "OperatorNode":
        from okapi.node import OperatorNode

        if children is None:
            children = []

        logger.debug(f"Creating WeightedMeanOp node with {len(children)} children")
        if len(children) == 0:
            weights = [1.0]
            logger.trace("No children, setting weight to [1.0]")
        elif len(children) == 1:
            parent_weight = np.random.uniform(0, 1)
            weights = [parent_weight, 1 - parent_weight]
            logger.trace(f"One child, weights: [{parent_weight}, {1 - parent_weight}]")
        else:
            weights = [np.random.uniform(0, 1)]  # initial weight for parent
            weight_left = 1 - weights[0]
            logger.trace(f"Multiple children, parent weight: {weights[0]}, remaining: {weight_left}")

            for i in range(len(children) - 1):
                weights.append(np.random.uniform(0, weight_left))
                weight_left -= weights[-1]
                logger.trace(f"Child {i + 1} weight: {weights[-1]}, remaining: {weight_left}")

            weights.append(weight_left)
            logger.trace(f"Final child weight: {weight_left}")

        op = cls(weights)
        node = OperatorNode(op, children)
        logger.debug(f"Created WeightedMeanOp node with weights: {weights}")
        return node

    def _weight_sum_assertion(self) -> None:
        weight_sum = np.sum(self._weights)
        if not np.isclose(weight_sum, 1):
            logger.error(f"Weights sum to {weight_sum}, not 1.0: {self._weights}")
            assert np.isclose(weight_sum, 1), "Weights do not sum to 1"
        logger.trace(f"Weight sum assertion passed: {weight_sum}")

    def _weight_length_assertion(self, node: "OperatorNode") -> None:
        expected_length = len(node.children) + 1
        actual_length = len(self._weights)
        if actual_length != expected_length:
            logger.error(f"Weight array length ({actual_length}) does not match expected {expected_length}")
            assert actual_length == expected_length, "Length of weight array is different than number of adjacent nodes"
        logger.trace(f"Weight length assertion passed: {actual_length}")


class ThresholdOp(Operation):
    def __init__(self, threshold: float, close: bool = True):
        assert threshold >= 0 or threshold <= 1, f"Threshold must be between 0 and 1 (inclusive) but is equal {threshold}"
        self.threshold = threshold
        self.close = close
        self.strclose = "Close" if self.close else "Far"

    def op(self, x: Tensor) -> Tensor:
        orig_shape = B.shape(x)
        adjusted = (x - self.threshold) ** 2
        adjusted = B.reshape(adjusted, (x.shape[0], -1))

        if self.close:
            ixes = B.argmin(adjusted, axis=0)
        else:
            ixes = B.argmax(adjusted, axis=0)

        x_reshaped = B.reshape(x, (x.shape[0], -1))
        col_indices = B.arange(B.shape(x_reshaped)[1], device_ref=x)
        x_selected = x_reshaped[ixes, col_indices]
        x = B.reshape(x_selected, orig_shape[1:])

        return x

    def copy(self) -> "ThresholdOp":
        return self.__class__(self.threshold)

    @property
    def code(self) -> str:
        return f"TH{self.strclose}".upper()

    def __str__(self) -> str:
        return f"ThresholdNode{self.strclose} with Threshold = {self.threshold:.2f}"


class CloseThresholdOp(ThresholdOp):
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold, True)

    @classmethod
    def create_node(cls, children: Optional[Sequence["ValueNode"]] = None) -> "OperatorNode":
        from okapi.node import OperatorNode

        t = np.random.rand()
        return OperatorNode(cls(t), children)


class FarThresholdOp(ThresholdOp):
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold, False)

    @classmethod
    def create_node(cls, children: Optional[Sequence["ValueNode"]] = None) -> "OperatorNode":
        from okapi.node import OperatorNode

        t = np.random.rand()
        return OperatorNode(cls(t), children)
