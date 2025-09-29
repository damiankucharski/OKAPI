from typing import List, Optional, Sequence, TypeVar, Union, cast

import numpy as np
from loguru import logger

from giraffe.globals import BACKEND as B
from giraffe.globals import postprocessing_function as PF
from giraffe.lib_types import Tensor

T = TypeVar("T", bound="Node")


class Node:
    """
    Nodes act as the fundamental building blocks of a tree,
    capable of holding children and a reference to their parent node.

    When created, parent reference cannot be specified. The reason for it is to create uniderectional
    responsibility for link creation. A node should be responsible for creating and breaking links with its children,
    by setting their parent links.

    Attributes:
        parent (Union[Node, None]): A reference to a parent node, of which this node is a child.
        children (List[Node]): A list of references to a children nodes.
    """

    def __init__(self, children: Optional[Sequence["Node"]] = None):
        """
        Create a node

        Args:
            children (Optional[Sequence["Node"]]): An optional list like sequence of children of the node
        """
        self.parent: Union[Node, None] = None
        self.children: List[Node] = list(children) if children is not None else []

        for child in self.children:
            child.parent = self

    def add_child(self, child_node: "Node"):
        """
        Add a child to the Node.

        Parameters:
        - child_node: Node to be added as child
        """
        logger.debug(f"Adding child node to {self}")
        self.children.append(child_node)
        child_node.parent = self
        logger.trace(f"Child added. Node now has {len(self.children)} children")

    def remove_child(self, child_node: "Node") -> "Node":
        logger.debug(f"Removing child node {child_node} from {self}")
        self.children.remove(child_node)
        child_node.parent = None
        logger.trace(f"Child removed. Node now has {len(self.children)} children")
        return child_node

    def replace_child(self, child, replacement_node):
        """
        Replaces child in place. No add child or remove child is called, so no add/remove adjustments are made.
        """
        logger.debug(f"Replacing child node {child} with {replacement_node} in {self}")

        if replacement_node.parent is not None:
            logger.error(f"Replacement node {replacement_node} already has a parent")
            raise ValueError("Replacement node already has a parent")

        ix = self.children.index(child)
        self.children[ix] = replacement_node

        child.parent = None
        replacement_node.parent = self
        logger.trace(f"Child replaced at index {ix}")

    def get_nodes(self):
        """
        Get all nodes in the tree created by node and its subnodes.
        Returns:
        - List of all nodes in the tree in breadth-first order
        """
        nodes = [self]
        current_level = [self]

        while current_level:
            next_level = []
            for node in current_level:
                next_level.extend(node.children)
            nodes.extend(next_level)
            current_level = next_level

        return nodes

    def copy(self):
        """
        Create a copy of the node.
        It's children and parent references are not copied.

        Returns:
        - Copy of the node
        """
        return Node()

    def copy_subtree(self):
        """
        Copy the subtree rooted at this node.
        Does not call "add_child" method to avoid any other operations like weight adjustments.
        Directly sets parent and children references.
        Returns:
        - Copy of the subtree rooted at this node
        """
        logger.debug(f"Creating copy of subtree rooted at {self}")
        self_copy = self.copy()

        for child in self.children:
            logger.trace(f"Copying child subtree: {child}")
            child_copy = child.copy_subtree()
            self_copy.children.append(child_copy)  # not "append_child" to avoid any other operations
            child_copy.parent = self_copy

        logger.trace(f"Subtree copy complete with {len(self_copy.children)} children")
        return self_copy

    def calculate(self):
        """
        Abstract method for calculation logic.

        Returns:
        - Calculated Tensor object
        """
        raise NotImplementedError("Calculate method not implemented")

    @property
    def code(self) -> str:
        """
        Identifies node for duplicate handling.

        Returns:
        - Code string
        """
        return f"Node at {hex(id(self))}"

    def __repr__(self):
        return self.code


class ValueNode(Node):
    """
    Represents a Value Node in a computational tree.

    A Value Node holds a specific value or tensor.
    """

    def __init__(self, children: Optional[Sequence["OperatorNode"]], value, id: Union[int, str]):
        super().__init__(children)
        self.value = value
        self.evaluation: None | Tensor = None
        self.id = id

    def calculate(self):
        logger.trace(f"Calculating value for ValueNode {self.id}")
        if self.children:
            for child in self.children:
                logger.trace(f"Calculating from child node: {child}")
                self.evaluation = child.calculate()
        else:
            self.evaluation = self.value
            logger.trace(f"Using direct value for node {self.id}")
        return self.evaluation

    def __str__(self):
        return f"ValueNode with value at: {hex(id(self.value))}"  # and evaluation: {self.evaluation}"

    def add_child(self, child_node):
        logger.debug(f"Adding child to ValueNode {self.id}")
        super().add_child(child_node)
        self.evaluation = None
        logger.debug("Child added and evaluation reset")

    def copy(self) -> "ValueNode":
        return ValueNode(None, self.value, self.id)

    @property
    def code(self) -> str:
        return f"VN[{self.id}]"


class OperatorNode(Node):
    """
    Abstract Base Class for an Operator Node in a computational tree.

    Reduction Operator Nodes are specialized Operator Nodes capable
    of performing reduction operations like mean, max, min, etc., on tensors.
    """

    def __init__(
        self,
        children: Optional[Sequence[ValueNode]],
    ):
        super().__init__(children)

    def calculate(self):
        logger.trace(f"Calculating value for {self.__class__.__name__}")
        concat = self._concat()
        logger.trace(f"Concatenated tensor shape: {B.shape(concat)}")
        post_op = self.op(concat)
        logger.trace(f"Post-operation tensor shape: {B.shape(post_op)}")
        postprocessed = PF(post_op)  # by default passthrough, may change for different tasks
        return postprocessed

    def _concat(self):
        assert self.parent is not None, "OperatorNode must have a parent to be calculated"
        parent: ValueNode = cast(ValueNode, self.parent)
        parent_eval = parent.evaluation if parent.evaluation is not None else parent.value
        logger.trace(f"Concatenating parent and {len(self.children)} children tensors")
        return B.concat(
            [B.unsqueeze(parent_eval, axis=0)] + [B.unsqueeze(child.calculate(), axis=0) for child in self.children],
            axis=0,
        )

    @staticmethod
    def create_node(children):
        raise NotImplementedError()

    def op(self, x):
        return x


class MeanNode(OperatorNode):
    """
    Represents a Mean Node in a computational tree.

    A Mean Node computes the mean along a specified axis of a tensor.
    """

    def __init__(self, children: Optional[Sequence[ValueNode]]):
        super().__init__(children)

    def __str__(self) -> str:
        return "MeanNode"

    def copy(self):
        return MeanNode(None)

    @property
    def code(self) -> str:
        return "MN"

    def op(self, x):
        return B.mean(x, axis=0)

    @staticmethod
    def create_node(children):  # TODO: it could be derived from simple vs parametrized OperatorNode
        return MeanNode(children)


class WeightedMeanNode(OperatorNode):
    """
    Represents a Weighted Mean Node in a computational tree.

    A Weighted Mean Node computes the mean of a tensor,
    but with different weights applied to each element.
    """

    def __init__(
        self,
        children: Optional[Sequence[ValueNode]],
        weights: List[float],
    ):
        logger.debug(f"Creating WeightedMeanNode with {len(weights) if weights else 0} weights")
        self._weights = weights
        super().__init__(children)

        self._weight_sum_assertion()
        logger.trace(f"WeightedMeanNode initialized with weights: {weights}")

    def op(self, x):
        weight_shape = (-1, *([1] * (len(x.shape) - 1)))
        w = B.reshape(self.weights, weight_shape)
        x = x * w
        x = B.sum(x, axis=0)
        return x

    def copy(self):
        return WeightedMeanNode([], [x for x in self._weights])  # this needs to be rethought

    def add_child(self, child_node: Node):
        logger.debug(f"Adding child to WeightedMeanNode with current weights: {self._weights}")
        assert isinstance(child_node, ValueNode)
        child_weight = np.random.uniform(0, 1)
        adj = 1.0 - child_weight

        logger.trace(f"Generated child weight: {child_weight}, adjustment factor: {adj}")
        for i, val in enumerate(self._weights):
            self._weights[i] = val * adj
        self._weights.append(child_weight)
        self._weight_sum_assertion()

        super().add_child(child_node)
        self._weight_length_assertion()
        logger.debug(f"Child added, new weights: {self._weights}")

    def remove_child(self, child_node: Node):
        logger.debug(f"Removing child from WeightedMeanNode with current weights: {self._weights}")
        assert isinstance(child_node, ValueNode), "Child node of WMN must be a ValueNode"

        child_ix = self.children.index(child_node)
        adj = 1.0 - self._weights[child_ix + 1]  # adjust for parent weight being first
        weight_removed = self._weights[child_ix + 1]
        self._weights.pop(child_ix + 1)

        logger.trace(f"Removed weight at index {child_ix + 1} with value {weight_removed}, adjustment factor: {adj}")

        super().remove_child(child_node)

        for i, val in enumerate(self._weights):
            self._weights[i] = val / adj

        self._weight_sum_assertion()
        self._weight_length_assertion()

        logger.debug(f"Child removed, new weights: {self._weights}")
        return child_node

    def replace_child(self, child, replacement_node):
        super().replace_child(child, replacement_node)
        self._weight_length_assertion()

    def calculate(self):
        self._weight_length_assertion()
        self._weight_sum_assertion()
        return super().calculate()

    def __str__(self) -> str:
        return f"WeightedMeanNode with weights: {B.to_numpy(B.tensor(self._weights)).round(2)}"

    @property
    def code(self) -> str:
        return "WMN"

    @property
    def weights(self):
        w = B.tensor(self._weights)
        return w

    @staticmethod
    def create_node(children: Sequence[ValueNode]):  # TODO: add tests for that function
        logger.debug(f"Creating WeightedMeanNode with {len(children)} children")
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

        node = WeightedMeanNode(children, weights)
        logger.debug(f"Created WeightedMeanNode with weights: {weights}")
        return node

    def _weight_sum_assertion(self):
        weight_sum = np.sum(self._weights)
        if not np.isclose(weight_sum, 1):
            logger.error(f"Weights sum to {weight_sum}, not 1.0: {self._weights}")
            assert np.isclose(weight_sum, 1), "Weights do not sum to 1"
        logger.trace(f"Weight sum assertion passed: {weight_sum}")

    def _weight_length_assertion(self):
        expected_length = len(self.children) + 1
        actual_length = len(self._weights)
        if actual_length != expected_length:
            logger.error(f"Weight array length ({actual_length}) does not match expected {expected_length}")
            assert actual_length == expected_length, "Length of weight array is different than number of adjacent nodes"
        logger.trace(f"Weight length assertion passed: {actual_length}")


class MaxNode(OperatorNode):
    """
    Represents a Max Node in a computational tree.

    A Max Node computes the maximum value along a specified axis of a tensor.
    """

    def __init__(self, children: Optional[Sequence[ValueNode]]):
        super().__init__(children)

    def __str__(self) -> str:
        return "MaxNode"

    def copy(self):
        return MaxNode(None)

    @property
    def code(self) -> str:
        return "MAX"

    def op(self, x):
        return B.max(x, axis=0)

    def adjust_params(self):
        return

    @staticmethod
    def create_node(children):
        return MaxNode(children)


class MinNode(OperatorNode):
    """
    Represents a Min Node in a computational tree.

    A Min Node computes the minimum value along a specified axis of a tensor.
    """

    def __init__(self, children: Optional[Sequence[ValueNode]]):
        super().__init__(children)

    def __str__(self) -> str:
        return "MinNode"

    def copy(self):
        return MinNode(None)

    @property
    def code(self) -> str:
        return "MIN"

    def op(self, x):
        return B.min(x, axis=0)

    def adjust_params(self):
        return

    @staticmethod
    def create_node(children):
        return MinNode(children)


class ThresholdNode(OperatorNode):
    """
    Chooses values closest (or furthest away) from the provided threshold value)
    It does not select "whole samples" like the other nodes would in general.
    If the shape of x (excluding the batch dimension) is non-flat, for example
    [
        [
            [0.2, 0.3],
            [0.3, 0.3]
        ],
        [
            [0.29, 0.5],
            [0.5, 0.5]
        ],
    ]
    And threshold == 0.3, for "close" option it will select 3 out of 4 values from first batch
    element, and 1 out of the second batch element, resulting in the following output:
    [
        [0.29, 0.3],
        [0.3, 0.3]
    ]
    """

    def __init__(self, children: Optional[Sequence[ValueNode]], threshold: float, close=True):
        assert threshold >= 0 or threshold <= 1, f"Threshold must be between 0 and 1 (inclusive) but is equal {threshold}"
        super().__init__(children)
        self.close = close
        self.strclose = "Close" if self.close else "Far"
        self.threshold = threshold

    def __str__(self) -> str:
        return f"ThresholdNode{self.strclose} with Threshold = {self.threshold:.2f}"

    def copy(self):
        return ThresholdNode(None, self.threshold, self.close)

    @property
    def code(self) -> str:
        return f"TH{self.strclose}".upper()

    def op(self, x):
        orig_shape = B.shape(x)
        adjusted = (x - self.threshold) ** 2
        adjusted = B.reshape(adjusted, (x.shape[0], -1))

        if self.close:
            ixes = B.argmin(adjusted, axis=0)
        else:
            ixes = B.argmax(adjusted, axis=0)

        x_reshaped = B.reshape(x, (x.shape[0], -1))
        col_indices = B.tensor(np.arange(B.shape(x_reshaped)[1]).tolist())
        # Select the value from the row with min/max squared distance in each "column - place"
        x_selected = x_reshaped[ixes, col_indices]
        x = B.reshape(x_selected, orig_shape[1:])

        return x

    def adjust_params(self):
        return

    @staticmethod
    def create_node(children):
        raise NotImplementedError("This node is not supposed to be initialized, use child classes instead.")


class CloseThresholdNode(ThresholdNode):
    """
    Chooses values closest (or furthest away) from the provided threshold value)
    """

    def __init__(self, children: Optional[Sequence[ValueNode]], threshold: float):
        super().__init__(children, threshold, True)

    def copy(self):
        return CloseThresholdNode(None, self.threshold)

    @staticmethod
    def create_node(children):
        t = np.random.rand()
        return CloseThresholdNode(children, t)


class FarThresholdNode(ThresholdNode):
    """
    Chooses values closest (or furthest away) from the provided threshold value)
    """

    def __init__(self, children: Optional[Sequence[ValueNode]], threshold: float):
        super().__init__(children, threshold, False)

    def copy(self):
        return FarThresholdNode(None, self.threshold)

    @staticmethod
    def create_node(children):
        t = np.random.rand()
        return FarThresholdNode(children, t)


def check_if_both_types_values(node1, node2):
    if not isinstance(node1, type):
        node1 = type(node1)
    if not isinstance(node2, type):
        node2 = type(node2)

    return issubclass(node1, ValueNode) and issubclass(node2, ValueNode)


def check_if_both_types_operators(node1, node2):
    if not isinstance(node1, type):
        node1 = type(node1)
    if not isinstance(node2, type):
        node2 = type(node2)
    return issubclass(node1, OperatorNode) and issubclass(node2, OperatorNode)


def check_if_both_types_same_node_variant(node1, node2):
    if not isinstance(node1, type):
        node1 = type(node1)
    if not isinstance(node2, type):
        node2 = type(node2)
    return check_if_both_types_operators(node1, node2) or check_if_both_types_values(node1, node2)
