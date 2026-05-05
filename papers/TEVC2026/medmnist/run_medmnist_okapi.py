# run_single_experiment.py

import argparse
import dataclasses
import os
import json
import sys
from dataclasses import dataclass
from pathlib import Path

def _add_okapi_repo_to_path() -> None:
    candidates = []
    if os.environ.get("OKAPI_REPO"):
        candidates.append(Path(os.environ["OKAPI_REPO"]))
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "OKAPI")
    for candidate in candidates:
        if (candidate / "okapi" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return


_add_okapi_repo_to_path()
os.environ.setdefault("BACKEND", "pytorch")

# It's good practice to handle potential import errors if run standalone
try:
    from okapi.operators import CLOSE_THRESHOLD, FAR_THRESHOLD, MAX, MEAN, MIN, WEIGHTED_MEAN
    from okapi.pareto import maximize
    from okapi.globals import BACKEND as B
    from okapi.fitness import (
        roc_auc_binary,
        average_precision_binary,
        average_precision_multiclass,
        average_precision_multilabel,
        roc_auc_multiclass,
        roc_auc_multilabel,
    )
    from okapi.tree import Tree
    from okapi import Okapi
except ImportError as e:
    print(f"Error importing a required library: {e}")
    print("Please ensure that 'okapi' and 'torch' are installed, or set OKAPI_REPO to the cloned OKAPI repository.")
    exit(1)


# --- CONFIGURATION DICTIONARIES ---

pr_auc_metrics = {"binary": average_precision_binary, "multiclass": average_precision_multiclass, "multilabel": average_precision_multilabel}

roc_auc_metrics = {"binary": roc_auc_binary, "multiclass": roc_auc_multiclass, "multilabel": roc_auc_multilabel}

dataset_tasks = {
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

experiments = {
    1: {"metrics": [pr_auc_metrics], "minimize_node_count": True, "allowed_ops": [MEAN, MIN, MAX]},
    2: {"metrics": [pr_auc_metrics], "minimize_node_count": True, "allowed_ops": [MEAN, MIN, MAX, FAR_THRESHOLD, CLOSE_THRESHOLD, WEIGHTED_MEAN]},
    3: {
        "metrics": [roc_auc_metrics, pr_auc_metrics],
        "minimize_node_count": True,
        "allowed_ops": [MEAN, MIN, MAX, FAR_THRESHOLD, CLOSE_THRESHOLD, WEIGHTED_MEAN],
    },
    4: {"metrics": [pr_auc_metrics], "minimize_node_count": False, "allowed_ops": [MEAN, MIN, MAX]},
    5: {"metrics": [pr_auc_metrics], "minimize_node_count": False, "allowed_ops": [MEAN, MIN, MAX, FAR_THRESHOLD, CLOSE_THRESHOLD, WEIGHTED_MEAN]},
    6: {
        "metrics": [roc_auc_metrics, pr_auc_metrics],
        "minimize_node_count": False,
        "allowed_ops": [MEAN, MIN, MAX, FAR_THRESHOLD, CLOSE_THRESHOLD, WEIGHTED_MEAN],
    },
}


# --- DATA MODEL FOR RESULTS ---

@dataclass
class Evaluation:
    seed: int
    experiment_id: int
    dataset: str
    task: str
    tree_id: int
    n_val_nodes: int
    n_op_nodes: int
    n_unique_models: int
    unique_models: list[str]
    train_roc_auc: float
    val_roc_auc: float
    test_roc_auc: float
    train_pr_auc: float
    val_pr_auc: float
    test_pr_auc: float
    optimize_roc_auc: bool
    optimize_pr_auc: bool
    optimize_node_count: bool
    use_mean_node: bool
    use_min_max_nodes: bool
    use_weighted_mean_node: bool
    use_threshold_nodes: bool


# --- HELPER FUNCTIONS ---

def build_okapi(experiment_id, dataset, seed=0, preds_dir=Path("data_technical_paper/models"), gt_dir=Path("data_technical_paper/gt")):
    exp_config = experiments[experiment_id]
    task = dataset_tasks[dataset]
    metrics = [metric[task] for metric in exp_config["metrics"]]
    objectives = [maximize for i in range(len(metrics))]

    g = Okapi(
        [preds_dir / dataset / "valid"],
        [gt_dir / dataset / "val.pt"],
        20,
        2,
        5,
        backend="torch",
        objective_functions=metrics,
        objectives=objectives,
        minimize_node_count=exp_config["minimize_node_count"],
        allowed_ops=exp_config["allowed_ops"],
        seed=seed,
    )
    return g


def dump_json(ev: Evaluation, output_dir: Path):
    json_path = output_dir / f"seed_{ev.seed}_exp_{ev.experiment_id}_tree_{ev.tree_id}_dataset_{ev.dataset}.json"

    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as jfile:
        json.dump(dataclasses.asdict(ev), jfile)


def train_and_evaluate(okapi_run: Okapi, experiment_id: int, dataset: str, preds_dir: Path, gt_dir: Path, output_dir: Path):
    okapi_run.train(100)
    exp_config = experiments[experiment_id]
    task = dataset_tasks[dataset]

    train_gt = B.load(gt_dir / dataset / "train.pt")
    test_gt = B.load(gt_dir / dataset / "test.pt")

    for ix, pareto_tree in enumerate(okapi_run.pareto_trees):
        _, train_tree = pareto_tree.do_pred_on_another_tensors(
            preds_directory=preds_dir / dataset / "train", return_tree=True
        )

        _, test_tree = pareto_tree.do_pred_on_another_tensors(
            preds_directory=preds_dir / dataset / "test", return_tree=True
        )

        ev = Evaluation(
            seed=okapi_run.seed,
            experiment_id=experiment_id,
            dataset=dataset,
            task=task,
            tree_id=ix,
            n_val_nodes=len(pareto_tree.value_nodes),
            n_op_nodes=len(pareto_tree.op_nodes),
            n_unique_models=len(pareto_tree.unique_value_node_ids),
            unique_models=pareto_tree.unique_value_node_ids,
            train_roc_auc=roc_auc_metrics[task](train_tree, train_gt),
            val_roc_auc=roc_auc_metrics[task](pareto_tree, okapi_run.gt_tensor),
            test_roc_auc=roc_auc_metrics[task](test_tree, test_gt),
            train_pr_auc=pr_auc_metrics[task](train_tree, train_gt),
            val_pr_auc=pr_auc_metrics[task](pareto_tree, okapi_run.gt_tensor),
            test_pr_auc=pr_auc_metrics[task](test_tree, test_gt),
            optimize_pr_auc=pr_auc_metrics in exp_config["metrics"],
            optimize_roc_auc=roc_auc_metrics in exp_config["metrics"],
            optimize_node_count=exp_config["minimize_node_count"],
            use_mean_node=MEAN in exp_config["allowed_ops"],
            use_min_max_nodes=MIN in exp_config["allowed_ops"],
            use_weighted_mean_node=WEIGHTED_MEAN in exp_config["allowed_ops"],
            use_threshold_nodes=FAR_THRESHOLD in exp_config["allowed_ops"],
        )
        dump_json(ev, output_dir)


# --- MAIN EXPERIMENT FUNCTION ---

def do_experiment(dataset: str, experiment_id: int, seed: int, preds_dir: Path, gt_dir: Path, output_dir: Path, trees_dir: Path):
    """The main function that runs a single, complete experiment."""
    print(f"Starting: experiment_id={experiment_id}, dataset={dataset}, seed={seed}")

    tree_path = trees_dir / dataset / str(experiment_id) / str(seed)
    if not tree_path.exists():
    
      gir = build_okapi(experiment_id=experiment_id, dataset=dataset, seed=seed, preds_dir=preds_dir, gt_dir=gt_dir)
      train_and_evaluate(gir, experiment_id, dataset, preds_dir, gt_dir, output_dir)
      
      # Save pareto trees
      path = tree_path
      path.mkdir(parents=True, exist_ok=True)
      for ix, tree in enumerate(gir.pareto_trees):
          tree.save_tree_architecture(path / f'pareto_tree_{ix}')
      
      print(f"Completed: experiment_id={experiment_id}, dataset={dataset}, seed={seed}")
    else:

      print(f"Already completed: experiment_id={experiment_id}, dataset={dataset}, seed={seed}")

# --- COMMAND-LINE INTERFACE ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single OKAPI MedMNIST experiment.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--experiment_id", type=int, required=True, help="ID of the experiment config")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for the experiment")
    parser.add_argument("--preds-dir", type=Path, default=Path("data_technical_paper/models"), help="Directory containing model predictions")
    parser.add_argument("--gt-dir", type=Path, default=Path("data_technical_paper/gt"), help="Directory containing ground-truth tensors")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/medmnist_okapi/jsons"), help="Directory for JSON outputs")
    parser.add_argument("--trees-dir", type=Path, default=Path("outputs/medmnist_okapi/trees"), help="Directory for saved Pareto trees")
    
    args = parser.parse_args()
    
    do_experiment(args.dataset, args.experiment_id, args.seed, args.preds_dir, args.gt_dir, args.output_dir, args.trees_dir)
