from functools import partial
from typing import Literal

import torch

from okapi.globals import BACKEND as B
from okapi.lib_types import Tensor
from okapi.tree import Tree


def _infer_num_classes(gt: torch.Tensor, task: str) -> int:
    """
    Infer the number of classes from the ground truth tensor based on the task.

    Args:
        gt: Ground truth tensor
        task: Classification task type ('binary', 'multiclass', or 'multilabel')

    Returns:
        The inferred number of classes
    """
    if task == "binary":
        return 1
    elif task == "multiclass":
        # For multiclass, the number of classes is the maximum value + 1
        # (assuming class labels start from 0)
        return int(gt.max().item() + 1)
    elif task == "multilabel":
        # For multilabel, the number of classes is the number of columns
        if gt.dim() > 1:
            return gt.shape[1]
        else:
            return 1
    else:
        raise ValueError(f"Unknown task type: {task}")


def average_precision_fitness(
    tree_or_prediction: Tree | Tensor, gt: Tensor, task: Literal["binary", "multiclass", "multilabel"] = "binary"
) -> float:
    """
    Calculate the Average Precision (AP) score as a fitness measure using torchmetrics.

    Average Precision summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous threshold used
    as the weight. This implementation supports binary, multiclass, and multilabel classification.

    Args:
        tree_or_prediction: Either a Tree object (for backward compatibility) or a prediction
                           tensor directly (for memory-efficient usage with tree.predict())
        gt: Ground truth tensor containing labels
        task: Classification task type:
            - 'binary': Binary classification (default)
            - 'multiclass': Multiclass classification
            - 'multilabel': Multilabel classification

    Returns:
        Average Precision score as a float between 0 and 1 (higher is better)
    """
    from torchmetrics.classification import AveragePrecision

    # Handle both Tree objects and direct tensors
    if isinstance(tree_or_prediction, Tree):
        evaluation = tree_or_prediction.evaluation
    else:
        evaluation = tree_or_prediction

    # Convert to torch.Tensor for torchmetrics compatibility
    if not isinstance(evaluation, torch.Tensor):
        pred = torch.tensor(B.to_numpy(evaluation))
    else:
        pred = evaluation

    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(B.to_numpy(gt))

    # Infer number of classes from ground truth
    num_classes = _infer_num_classes(gt, task)
    gt = gt.squeeze().int()

    # Create metric with appropriate parameters based on task
    if task == "multiclass":
        metric = AveragePrecision(task=task, num_classes=num_classes)
    elif task == "multilabel":
        metric = AveragePrecision(task=task, num_labels=num_classes)
    else:  # binary
        if pred.dim() > 1:
            if pred.shape[1] == 2:
                pred = pred[:, 1]
            elif pred.shape[1] == 1:
                pred = pred.squeeze()
        metric = AveragePrecision(task=task)

    # Calculate and return the score
    return metric(pred, gt).item()


def roc_auc_score_fitness(
    tree_or_prediction: Tree | Tensor, gt: Tensor, task: Literal["binary", "multiclass", "multilabel"] = "binary"
) -> float:
    """
    Calculate the Area Under the ROC Curve (AUC-ROC) score as a fitness measure using torchmetrics.

    The AUC-ROC score represents the probability that a randomly chosen positive instance
    is ranked higher than a randomly chosen negative instance. This implementation supports
    binary, multiclass, and multilabel classification.

    Args:
        tree_or_prediction: Either a Tree object (for backward compatibility) or a prediction
                           tensor directly (for memory-efficient usage with tree.predict())
        gt: Ground truth tensor containing labels
        task: Classification task type:
            - 'binary': Binary classification (default)
            - 'multiclass': Multiclass classification
            - 'multilabel': Multilabel classification

    Returns:
        ROC AUC score as a float between 0 and 1 (higher is better)
    """
    from torchmetrics.classification import AUROC

    # Handle both Tree objects and direct tensors
    if isinstance(tree_or_prediction, Tree):
        evaluation = tree_or_prediction.evaluation
    else:
        evaluation = tree_or_prediction

    # Convert to torch.Tensor for torchmetrics compatibility
    if not isinstance(evaluation, torch.Tensor):
        pred = torch.tensor(B.to_numpy(evaluation))
    else:
        pred = evaluation

    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(B.to_numpy(gt))

    # Infer number of classes from ground truth
    num_classes = _infer_num_classes(gt, task)
    gt = gt.squeeze().int()

    # Create metric with appropriate parameters based on task
    if task == "multiclass":
        metric = AUROC(task=task, num_classes=num_classes)
    elif task == "multilabel":
        metric = AUROC(task=task, num_labels=num_classes)
    else:  # binary
        metric = AUROC(task=task)

    # Calculate and return the score
    return metric(pred, gt).item()


def accuracy_fitness(
    tree_or_prediction: Tree | Tensor, gt: Tensor, task: Literal["binary", "multiclass", "multilabel"] = "binary"
) -> float:
    """
    Calculate the Accuracy score as a fitness measure using torchmetrics.

    Accuracy is the proportion of correct predictions among the total number of cases processed.
    This implementation supports binary, multiclass, and multilabel classification.

    Args:
        tree_or_prediction: Either a Tree object (for backward compatibility) or a prediction
                           tensor directly (for memory-efficient usage with tree.predict())
        gt: Ground truth tensor containing labels
        task: Classification task type:
            - 'binary': Binary classification (default)
            - 'multiclass': Multiclass classification
            - 'multilabel': Multilabel classification

    Returns:
        Accuracy score as a float between 0 and 1 (higher is better)
    """
    from torchmetrics.classification import Accuracy

    # Handle both Tree objects and direct tensors
    if isinstance(tree_or_prediction, Tree):
        evaluation = tree_or_prediction.evaluation
    else:
        evaluation = tree_or_prediction

    # Convert to torch.Tensor for torchmetrics compatibility
    if not isinstance(evaluation, torch.Tensor):
        pred = torch.tensor(B.to_numpy(evaluation))
    else:
        pred = evaluation

    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(B.to_numpy(gt))

    # Infer number of classes from ground truth
    num_classes = _infer_num_classes(gt, task)
    gt = gt.squeeze().int()

    # Create metric with appropriate parameters based on task
    if task == "multiclass":
        metric = Accuracy(task=task, num_classes=num_classes)
    elif task == "multilabel":
        metric = Accuracy(task=task, num_labels=num_classes)
    else:  # binary
        metric = Accuracy(task=task)

    # Calculate and return the score
    return metric(pred, gt).item()


# Convenience partial functions for different classification tasks
average_precision_binary = partial(average_precision_fitness, task="binary")
average_precision_multiclass = partial(average_precision_fitness, task="multiclass")
average_precision_multilabel = partial(average_precision_fitness, task="multilabel")

roc_auc_binary = partial(roc_auc_score_fitness, task="binary")
roc_auc_multiclass = partial(roc_auc_score_fitness, task="multiclass")
roc_auc_multilabel = partial(roc_auc_score_fitness, task="multilabel")

accuracy_binary = partial(accuracy_fitness, task="binary")
accuracy_multiclass = partial(accuracy_fitness, task="multiclass")
accuracy_multilabel = partial(accuracy_fitness, task="multilabel")
