from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler


def _add_okapi_repo_to_path() -> None:
    """Allow running this script from the paper bundle next to a cloned OKAPI repo."""
    candidates = []
    if os.environ.get("OKAPI_REPO"):
        candidates.append(Path(os.environ["OKAPI_REPO"]))
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[2] / "OKAPI",
            here.parents[3] / "OKAPI" if len(here.parents) > 3 else here.parents[2] / "OKAPI",
        ]
    )
    for candidate in candidates:
        if (candidate / "okapi" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return


_add_okapi_repo_to_path()
os.environ.setdefault("BACKEND", "pytorch")

from okapi.fitness import (  # noqa: E402
    average_precision_binary,
    average_precision_multiclass,
    average_precision_multilabel,
    roc_auc_binary,
    roc_auc_multiclass,
    roc_auc_multilabel,
)
from okapi.node import ValueNode  # noqa: E402
from okapi.tree import Tree  # noqa: E402


PR_AUC_METRICS = {
    "binary": average_precision_binary,
    "multiclass": average_precision_multiclass,
    "multilabel": average_precision_multilabel,
}

ROC_AUC_METRICS = {
    "binary": roc_auc_binary,
    "multiclass": roc_auc_multiclass,
    "multilabel": roc_auc_multilabel,
}

DATASET_TASKS = {
    "pathmnist": "multiclass",
    "chestmnist": "multilabel",
    "dermamnist": "multiclass",
    "octmnist": "multiclass",
    "pneumoniamnist": "binary",
    "retinamnist": "multiclass",
    "breastmnist": "binary",
    "bloodmnist": "multiclass",
    "tissuemnist": "multiclass",
    "organamnist": "multiclass",
    "organcmnist": "multiclass",
    "organsmnist": "multiclass",
}

DATASETS = list(DATASET_TASKS)


@dataclass
class AlternativeMethodEvaluation:
    method: str
    id: int
    dataset: str
    train_roc_auc: float
    val_roc_auc: float
    test_roc_auc: float
    train_pr_auc: float
    val_pr_auc: float
    test_pr_auc: float
    seed: int


@dataclass
class NSGA3Results:
    weights: np.ndarray
    metrics: np.ndarray


def make_tree_from_tensor(tensor: torch.Tensor) -> Tree:
    root = ValueNode(None, tensor, 0)
    return Tree.create_tree_from_root(root)


class EnsembleOptimizationProblem(Problem):
    def __init__(self, model_predictions: torch.Tensor, y_true: torch.Tensor, metric_roc, metric_pr):
        self.model_predictions = model_predictions.detach().cpu().numpy()
        self.y_true = y_true
        self.metric_roc = metric_roc
        self.metric_pr = metric_pr
        super().__init__(
            n_var=len(model_predictions),
            n_obj=2,
            n_eq_constr=1,
            xl=0.0,
            xu=1.0,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        roc_aucs = np.zeros(X.shape[0])
        pr_aucs = np.zeros(X.shape[0])
        reshape = (X.shape[1],) + (1,) * (self.model_predictions.ndim - 1)
        for i, weights in enumerate(X):
            ensemble_pred = (self.model_predictions * weights.reshape(reshape)).sum(axis=0)
            tree = make_tree_from_tensor(torch.tensor(ensemble_pred))
            roc_aucs[i] = -self.metric_roc(tree, self.y_true)
            pr_aucs[i] = -self.metric_pr(tree, self.y_true)
        out["F"] = np.column_stack([roc_aucs, pr_aucs])
        out["H"] = np.sum(X, axis=1) - 1.0


class AlternativeMethods:
    def __init__(self, path_preds: str | Path, path_gt: str | Path, dataset: str, weights_dir: str | Path | None = None):
        if dataset not in DATASET_TASKS:
            raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {', '.join(DATASETS)}")
        self.path_preds = Path(path_preds)
        self.path_gt = Path(path_gt)
        self.dataset = dataset
        self.weights_dir = Path(weights_dir) if weights_dir else Path("outputs/medmnist_alternatives/nsga3_weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.train_tensors, self.train_gt = self.load_all_tensors("train")
        self.val_tensors, self.val_gt = self.load_all_tensors("val")
        self.test_tensors, self.test_gt = self.load_all_tensors("test")
        self.task = DATASET_TASKS[dataset]
        self.auroc = ROC_AUC_METRICS[self.task]
        self.aupr = PR_AUC_METRICS[self.task]

    def simple_average_tree(self, cat_tensors: torch.Tensor) -> Tree:
        return make_tree_from_tensor(cat_tensors.mean(dim=0))

    def evaluation_simple_average(self, seed=0) -> list[AlternativeMethodEvaluation]:
        train_tree = self.simple_average_tree(self.train_tensors)
        val_tree = self.simple_average_tree(self.val_tensors)
        test_tree = self.simple_average_tree(self.test_tensors)
        return [
            AlternativeMethodEvaluation(
                method="simple_average",
                id=0,
                dataset=self.dataset,
                train_roc_auc=self.auroc(train_tree, self.train_gt),
                val_roc_auc=self.auroc(val_tree, self.val_gt),
                test_roc_auc=self.auroc(test_tree, self.test_gt),
                train_pr_auc=self.aupr(train_tree, self.train_gt),
                val_pr_auc=self.aupr(val_tree, self.val_gt),
                test_pr_auc=self.aupr(test_tree, self.test_gt),
                seed=seed,
            )
        ]

    def logistic_regression_train(self):
        cat_tensors = self.val_tensors.movedim(0, 1)
        X = cat_tensors.reshape(cat_tensors.shape[0], -1).numpy()
        y = self.val_gt.numpy().squeeze()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=1000)
        if len(y.shape) != 1:
            lr = MultiOutputClassifier(lr)
        lr.fit(X, y)
        return lr, scaler

    def logistic_regression_get_tree(self, cat_tensors: torch.Tensor, lr, scaler: StandardScaler) -> Tree:
        cat_tensors = cat_tensors.movedim(0, 1)
        X = cat_tensors.reshape(cat_tensors.shape[0], -1).numpy()
        pred = lr.predict_proba(scaler.transform(X))
        if isinstance(pred, list):
            pred_np = np.column_stack([pr[:, 1] for pr in pred])
        elif isinstance(pred, np.ndarray):
            pred_np = pred
        else:
            raise TypeError(f"Unexpected predict_proba output type: {type(pred)}")
        return make_tree_from_tensor(torch.tensor(pred_np))

    def evaluation_logistic_regression(self, seed=0) -> list[AlternativeMethodEvaluation]:
        lr, scaler = self.logistic_regression_train()
        train_tree = self.logistic_regression_get_tree(self.train_tensors, lr, scaler)
        val_tree = self.logistic_regression_get_tree(self.val_tensors, lr, scaler)
        test_tree = self.logistic_regression_get_tree(self.test_tensors, lr, scaler)
        return [
            AlternativeMethodEvaluation(
                method="logistic_regression",
                id=0,
                dataset=self.dataset,
                train_roc_auc=self.auroc(train_tree, self.train_gt),
                val_roc_auc=self.auroc(val_tree, self.val_gt),
                test_roc_auc=self.auroc(test_tree, self.test_gt),
                train_pr_auc=self.aupr(train_tree, self.train_gt),
                val_pr_auc=self.aupr(val_tree, self.val_gt),
                test_pr_auc=self.aupr(test_tree, self.test_gt),
                seed=seed,
            )
        ]

    def optimize_ensemble_weights(self, pop_size=60, n_gen=100, seed=420) -> NSGA3Results:
        problem = EnsembleOptimizationProblem(self.val_tensors, self.val_gt, self.auroc, self.aupr)
        algorithm = NSGA3(pop_size=pop_size, ref_dirs=get_reference_directions("das-dennis", 2, n_partitions=12))
        result = minimize(problem, algorithm, ("n_gen", n_gen), verbose=True, seed=seed)
        return NSGA3Results(weights=result.X, metrics=-result.F)

    def nsga3_get_tree(self, tensor: torch.Tensor, weights: np.ndarray) -> Tree:
        reshape = (len(weights),) + (1,) * (tensor.ndim - 1)
        weighted = (tensor * torch.tensor(weights, dtype=tensor.dtype).reshape(reshape)).sum(dim=0)
        return make_tree_from_tensor(weighted)

    def evaluation_nsga3(self, seed=420) -> list[AlternativeMethodEvaluation]:
        nsga3_results = self.optimize_ensemble_weights(seed=seed)
        outs = []
        for i, weights in enumerate(nsga3_results.weights):
            train_tree = self.nsga3_get_tree(self.train_tensors, weights)
            val_tree = self.nsga3_get_tree(self.val_tensors, weights)
            test_tree = self.nsga3_get_tree(self.test_tensors, weights)
            np.save(self.weights_dir / f"{self.dataset}_{i}_seed_{seed}_nsga3weights.npy", weights)
            outs.append(
                AlternativeMethodEvaluation(
                    method="nsga3",
                    id=i,
                    dataset=self.dataset,
                    train_roc_auc=self.auroc(train_tree, self.train_gt),
                    val_roc_auc=self.auroc(val_tree, self.val_gt),
                    test_roc_auc=self.auroc(test_tree, self.test_gt),
                    train_pr_auc=self.aupr(train_tree, self.train_gt),
                    val_pr_auc=self.aupr(val_tree, self.val_gt),
                    test_pr_auc=self.aupr(test_tree, self.test_gt),
                    seed=seed,
                )
            )
        return outs

    def load_all_tensors(self, subset="train") -> tuple[torch.Tensor, torch.Tensor]:
        pred_subset = "valid" if subset == "val" else subset
        path_dataset = self.path_preds / self.dataset / pred_subset
        gt_dataset = self.path_gt / self.dataset / f"{subset}.pt"
        if not path_dataset.exists():
            raise FileNotFoundError(f"Prediction directory not found: {path_dataset}")
        if not gt_dataset.exists():
            raise FileNotFoundError(f"Ground-truth tensor not found: {gt_dataset}")
        tensors = [torch.load(path, map_location="cpu", weights_only=True).unsqueeze(0) for path in sorted(path_dataset.glob("*.pt"))]
        if not tensors:
            raise FileNotFoundError(f"No .pt prediction tensors found in: {path_dataset}")
        gt = torch.load(gt_dataset, map_location="cpu", weights_only=True)
        return torch.concatenate(tensors, dim=0), gt
