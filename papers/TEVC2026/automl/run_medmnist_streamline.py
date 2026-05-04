"""
STREAMLINE benchmark for MedMNIST
"""

import argparse
import csv
import logging
import os
import pickle
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, jaccard_score,
    precision_score, recall_score, roc_auc_score,
)

logging.disable(logging.CRITICAL)  # suppress STREAMLINE's verbose logging

STREAMLINE_REPO = Path(__file__).resolve().parent / "venvs" / "streamline_repo"
sys.path.insert(0, str(STREAMLINE_REPO))

from streamline.runners.dataprocess_runner import DataProcessRunner
from streamline.runners.imputation_runner import ImputationRunner
from streamline.runners.model_runner import ModelExperimentRunner
from streamline.runners.stats_runner import StatsRunner
from streamline.runners.replicate_runner import ReplicationRunner

BASE = Path(__file__).resolve().parent
PRED_DIR = BASE / "data_technical_paper" / "models"
GT_DIR = BASE / "data_technical_paper" / "gt"

CSV_COLUMNS = [
    "seed", "dataset", "n_models", "n_classes",
    "val_roc_auc", "val_pr_auc", "val_f1", "val_precision", "val_recall",
    "test_roc_auc", "test_pr_auc", "test_f1", "test_precision", "test_recall",
    "pipeline",
]

# STREAMLINE is binary-only
BINARY_DATASETS = {"breastmnist", "pneumoniamnist"}

# Algorithms: core set, excludes slow/unstable ones for a 100-seed benchmark
ALGORITHMS = ["LR", "NB", "DT", "RF", "GB"]

N_SPLITS = 5
N_TRIALS = 100
TIMEOUT = 120      # seconds per algorithm Optuna sweep


def distance_from_ideal(*metrics):
    return float(np.sqrt(sum((1 - m) ** 2 for m in metrics)))


def load_split(dataset: str, split: str):
    split_dir = PRED_DIR / dataset / split
    gt_split = {"valid": "val", "test": "test"}[split]
    gt_file = GT_DIR / dataset / f"{gt_split}.pt"
    gt = torch.load(gt_file, map_location="cpu", weights_only=True)
    y = gt.squeeze().numpy().astype(int)
    preds, names = [], []
    for pt_file in sorted(split_dir.glob("*.pt")):
        pred = torch.load(pt_file, map_location="cpu", weights_only=True)
        preds.append(pred.numpy())
        names.append(pt_file.stem)
    X = np.hstack(preds)
    return X, y, names


def get_datasets():
    return sorted(
        d.name for d in PRED_DIR.iterdir()
        if d.is_dir() and (d / "test").exists()
    )


def _arr_to_csv(X, y, path: Path, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    df["Class"] = y
    df.to_csv(path, index=False)


def _parse_val_metrics(exp_output: Path, dataset_name: str):
    """Read pickled CV metrics for each algorithm, return averaged results."""
    metrics_dir = exp_output / dataset_name / "model_evaluation" / "pickled_metrics"
    results = {}
    for abbrev in ALGORITHMS:
        rocs, praucs, f1s, precs, recs = [], [], [], [], []
        for cv in range(N_SPLITS):
            pkl = metrics_dir / f"{abbrev}_CV_{cv}_metrics.pickle"
            if not pkl.exists():
                continue
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            # data: [metric_list, fpr, tpr, roc_auc, prec_curve, rec_curve, pr_auc, ave_prec, fi, probas]
            metric_list = data[0]
            rocs.append(data[3])   # roc_auc
            praucs.append(data[6]) # pr_auc
            f1s.append(metric_list[2])
            recs.append(metric_list[3])
            precs.append(metric_list[5])
        if rocs:
            results[abbrev] = {
                "roc_auc": float(np.mean(rocs)),
                "pr_auc": float(np.mean(praucs)),
                "f1": float(np.mean(f1s)),
                "precision": float(np.mean(precs)),
                "recall": float(np.mean(recs)),
            }
    return results


def _parse_rep_metrics(exp_output: Path, dataset_name: str, rep_name: str):
    """Read replication (test) pickled metrics for each algorithm, averaged over CV folds."""
    metrics_dir = (exp_output / dataset_name / "replication" / rep_name
                   / "model_evaluation" / "pickled_metrics")
    results = {}
    for abbrev in ALGORITHMS:
        rocs, praucs, f1s, precs, recs = [], [], [], [], []
        for cv in range(N_SPLITS):
            pkl = metrics_dir / f"{abbrev}_CV_{cv}_metrics.pickle"
            if not pkl.exists():
                continue
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            metric_list = data[0]
            rocs.append(data[3])
            praucs.append(data[6])
            f1s.append(metric_list[2])
            recs.append(metric_list[3])
            precs.append(metric_list[5])
        if rocs:
            results[abbrev] = {
                "roc_auc": float(np.mean(rocs)),
                "pr_auc": float(np.mean(praucs)),
                "f1": float(np.mean(f1s)),
                "precision": float(np.mean(precs)),
                "recall": float(np.mean(recs)),
            }
    return results


def run_seed(dataset, X_val, y_val, X_test, y_test, n_models, seed):
    feature_names = [f"m{i}" for i in range(X_val.shape[1])]
    dataset_name = "val_data"
    rep_name = "test_data"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir = tmp / "data"
        rep_dir = tmp / "rep"
        output_path = tmp / "output"
        data_dir.mkdir()
        rep_dir.mkdir()

        val_csv = data_dir / f"{dataset_name}.csv"
        test_csv = rep_dir / f"{rep_name}.csv"
        _arr_to_csv(X_val, y_val, val_csv, feature_names)
        _arr_to_csv(X_test, y_test, test_csv, feature_names)

        exp = "exp"

        # Phase 1: data processing + stratified CV splits
        DataProcessRunner(
            str(data_dir), str(output_path), exp,
            class_label="Class", n_splits=N_SPLITS,
            partition_method="Stratified",
            random_state=seed, show_plots=False,
            exclude_eda_output=["describe", "univariate_plots", "correlation_plots"],
        ).run(run_parallel=False)

        # Phase 2: scale data (important for LR), no imputation (no missing values)
        ImputationRunner(
            str(output_path), exp,
            class_label="Class", scale_data=True, impute_data=False,
            random_state=seed,
        ).run(run_parallel=False)

        # Inject metadata keys normally set by Phases 3-4 (skipped here).
        # ModelExperimentRunner reads them but only uses them for ExSTraCS, which we don't run.
        meta_path = output_path / exp / "metadata.pickle"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta.setdefault("Filter Poor Features", False)
        meta.setdefault("Top Features to Display", 40)
        meta.setdefault("Max Features to Keep", 2000)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        # Phase 5: model training with Optuna hyperparameter optimisation
        ModelExperimentRunner(
            str(output_path), exp,
            algorithms=ALGORITHMS,
            exclude=[],
            class_label="Class",
            scoring_metric="roc_auc",
            metric_direction="maximize",
            n_trials=N_TRIALS,
            timeout=TIMEOUT,
            save_plots=False,
            random_state=seed,
        ).run(run_parallel=False)

        # Phase 6: statistics — generates pickled val CV metrics
        StatsRunner(
            str(output_path), exp,
            algorithms=ALGORITHMS,
            class_label="Class",
            scoring_metric="roc_auc",
            show_plots=False,
            exclude_plots=["plot_ROC", "plot_PRC", "plot_FI_box", "plot_metric_boxplots"],
        ).run(run_parallel=False)

        # Phase 8: replication — apply trained models to test set
        ReplicationRunner(
            str(rep_dir), str(val_csv), str(output_path), exp,
            load_algo=True,
            exclude_plots=["plot_ROC", "plot_PRC", "plot_metric_boxplots", "feature_correlations"],
        ).run(run_parallel=False)

        exp_output = output_path / exp
        val_metrics_all = _parse_val_metrics(exp_output, dataset_name)
        rep_metrics_all = _parse_rep_metrics(exp_output, dataset_name, rep_name)

    if not val_metrics_all:
        return None

    # Select best algorithm by distance from ideal on val (ROC-AUC, PR-AUC)
    best_algo = min(
        val_metrics_all,
        key=lambda a: distance_from_ideal(
            val_metrics_all[a]["roc_auc"], val_metrics_all[a]["pr_auc"]
        )
    )
    val_m = val_metrics_all[best_algo]
    test_m = rep_metrics_all.get(best_algo)
    if test_m is None:
        return None

    return {
        "seed": seed, "dataset": dataset, "n_models": n_models, "n_classes": 2,
        "val_roc_auc": val_m["roc_auc"], "val_pr_auc": val_m["pr_auc"],
        "val_f1": val_m["f1"], "val_precision": val_m["precision"],
        "val_recall": val_m["recall"],
        "test_roc_auc": test_m["roc_auc"], "test_pr_auc": test_m["pr_auc"],
        "test_f1": test_m["f1"], "test_precision": test_m["precision"],
        "test_recall": test_m["recall"],
        "pipeline": best_algo,
    }


def _run_one_seed(args_tuple):
    dataset, X_val, y_val, X_test, y_test, n_models, seed = args_tuple
    return seed, run_seed(dataset, X_val, y_val, X_test, y_test, n_models, seed)


def run_dataset(dataset, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"streamline_medmnist_{dataset}_results.csv"

    print(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")

    print("Loading val set...")
    X_val, y_val, model_names = load_split(dataset, "valid")
    n_classes = len(np.unique(y_val))
    n_models = len(model_names)
    print(f"  {X_val.shape}, {n_models} models, {n_classes} classes")

    if n_classes != 2:
        print(f"[skip] STREAMLINE is binary-only ({n_classes} classes)")
        return

    print("Loading test set...")
    X_test, y_test, _ = load_split(dataset, "test")
    print(f"  {X_test.shape}")

    completed = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    completed.add(int(row["seed"]))
                except (ValueError, KeyError):
                    pass
        print(f"Resuming: {len(completed)} seeds done")

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    pending = [s for s in range(args.seed_start, args.seed_end) if s not in completed]
    total = len(pending)
    done = 0

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        work = [(dataset, X_val, y_val, X_test, y_test, n_models, s) for s in pending]
        with ProcessPoolExecutor(max_workers=args.parallel_seeds) as pool:
            futures = {pool.submit(_run_one_seed, w): w[-1] for w in work}
            for fut in as_completed(futures):
                seed = futures[fut]
                done += 1
                try:
                    _, row = fut.result()
                except Exception as e:
                    print(f"  [{done}/{total}] seed {seed}... FAILED: {e}")
                    continue
                if row is None:
                    print(f"  [{done}/{total}] seed {seed}... no result")
                    continue
                writer.writerow(row)
                csv_file.flush()
                print(f"  [{done}/{total}] seed {seed}... "
                      f"pipeline={row['pipeline']} "
                      f"valAUC={row['val_roc_auc']:.4f} testAUC={row['test_roc_auc']:.4f}")

    print(f"Results saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="STREAMLINE MedMNIST benchmark (binary datasets only)")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default=str(BASE / "tpot_results"))
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=100)
    parser.add_argument("--parallel_seeds", type=int, default=5)
    args = parser.parse_args()

    datasets = get_datasets() if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        run_dataset(dataset, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
