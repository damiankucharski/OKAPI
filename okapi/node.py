from typing import List, Optional, Sequence, TypeVar, Union, cast

import numpy as np
from loguru import logger

from okapi.globals import BACKEND as B
from okapi.globals import postprocessing_function as PF
from okapi.lib_types import Tensor
from okapi.operation import (
    CloseThresholdOp,
    FarThresholdOp,
    MaxOp,
    MeanOp,
    MinOp,
    Operation,
    WeightedMeanOp,
)

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
    Operator Node in a computational tree.

    Uses an Operation strategy to define the reduction operation (mean, max, min, etc.)
    and manage any associated state (e.g., weights for weighted mean).
    """

    def __init__(
        self,
        operation: Operation,
        children: Optional[Sequence[ValueNode]] = None,
    ):
        self.operation = operation
        super().__init__(children)

    def add_child(self, child_node: Node):
        super().add_child(child_node)
        self.operation.on_child_added(self, child_node)

    def remove_child(self, child_node: Node) -> Node:
        index = self.children.index(child_node)
        super().remove_child(child_node)
        self.operation.on_child_removed(self, child_node, index)
        return child_node

    def replace_child(self, child, replacement_node):
        super().replace_child(child, replacement_node)
        self.operation.on_child_replaced(self, child, replacement_node)

    def calculate(self):
        logger.trace(f"Calculating value for OperatorNode with {self.operation.__class__.__name__}")
        self.operation.on_before_calculate(self)
        concat = self._concat()
        logger.trace(f"Concatenated tensor shape: {B.shape(concat)}")
        post_op = self.operation.op(concat)
        logger.trace(f"Post-operation tensor shape: {B.shape(post_op)}")
        postprocessed = PF(post_op)
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

    def copy(self):
        return OperatorNode(self.operation.copy())

    @property
    def code(self) -> str:
        return self.operation.code

    def __str__(self) -> str:
        return str(self.operation)

    @staticmethod
    def create_node(children):
        raise NotImplementedError()

    def op(self, x):
        return self.operation.op(x)


# --- Deprecated backward-compatibility wrappers ---
# These thin subclasses exist for pickle compatibility and old import paths.
# Use OperatorNode(MeanOp(), children) directly instead.


class MeanNode(OperatorNode):
    """Deprecated. Use OperatorNode(MeanOp()) instead."""

    def __init__(self, children: Optional[Sequence[ValueNode]] = None):
        super().__init__(MeanOp(), children)

    @staticmethod
    def create_node(children):
        return MeanNode(children)


class WeightedMeanNode(OperatorNode):
    """Deprecated. Use OperatorNode(WeightedMeanOp(weights), children) instead."""

    def __init__(
        self,
        children: Optional[Sequence[ValueNode]] = None,
        weights: Optional[List[float]] = None,
    ):
        if weights is None:
            weights = [1.0]
        super().__init__(WeightedMeanOp(weights), children)

    @property
    def weights(self):
        return self.operation.weights

    @property
    def _weights(self):
        return self.operation._weights

    @staticmethod
    def create_node(children: Sequence[ValueNode]):
        return WeightedMeanOp.create_node(children)


class MaxNode(OperatorNode):
    """Deprecated. Use OperatorNode(MaxOp()) instead."""

    def __init__(self, children: Optional[Sequence[ValueNode]] = None):
        super().__init__(MaxOp(), children)

    @staticmethod
    def create_node(children):
        return MaxNode(children)


class MinNode(OperatorNode):
    """Deprecated. Use OperatorNode(MinOp()) instead."""

    def __init__(self, children: Optional[Sequence[ValueNode]] = None):
        super().__init__(MinOp(), children)

    @staticmethod
    def create_node(children):
        return MinNode(children)


class ThresholdNode(OperatorNode):
    """Deprecated. Use OperatorNode(CloseThresholdOp(threshold)) or OperatorNode(FarThresholdOp(threshold)) instead."""

    def __init__(self, children: Optional[Sequence[ValueNode]] = None, threshold: float = 0.5, close: bool = True):
        if close:
            super().__init__(CloseThresholdOp(threshold), children)
        else:
            super().__init__(FarThresholdOp(threshold), children)
        self.threshold = threshold
        self.close = close

    @staticmethod
    def create_node(children):
        raise NotImplementedError("This node is not supposed to be initialized, use child classes instead.")


class CloseThresholdNode(ThresholdNode):
    """Deprecated. Use OperatorNode(CloseThresholdOp(threshold)) instead."""

    def __init__(self, children: Optional[Sequence[ValueNode]] = None, threshold: float = 0.5):
        super().__init__(children, threshold, True)

    @staticmethod
    def create_node(children):
        t = np.random.rand()
        return CloseThresholdNode(children, t)


class FarThresholdNode(ThresholdNode):
    """Deprecated. Use OperatorNode(FarThresholdOp(threshold)) instead."""

    def __init__(self, children: Optional[Sequence[ValueNode]] = None, threshold: float = 0.5):
        super().__init__(children, threshold, False)

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
