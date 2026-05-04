"""
TPOT ensemble benchmark for MedMNIST classification.
"""

import argparse
import csv
import gc
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, jaccard_score, make_scorer,
    precision_score, recall_score, roc_auc_score,
)

BASE = Path(__file__).resolve().parent
PRED_DIR = BASE / "data_technical_paper" / "models"
GT_DIR = BASE / "data_technical_paper" / "gt"

CSV_COLUMNS = [
    "seed", "dataset", "n_models", "n_classes",
    "val_roc_auc", "val_pr_auc", "val_f1", "val_precision", "val_recall",
    "test_roc_auc", "test_pr_auc", "test_f1", "test_precision", "test_recall",
    "pipeline",
]


def distance_from_ideal(*metrics):
    return float(np.sqrt(sum((1 - m) ** 2 for m in metrics)))


def _medmnist_distance_score(y_true, y_prob):
    if y_prob.ndim == 1 or y_prob.shape[1] <= 2:
        prob = y_prob[:, 1] if y_prob.ndim > 1 and y_prob.shape[1] == 2 else y_prob.ravel()
        pr_auc = average_precision_score(y_true, prob)
        roc_auc = roc_auc_score(y_true, prob)
    else:
        pr_auc = average_precision_score(y_true, y_prob, average="macro")
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    return distance_from_ideal(pr_auc, roc_auc)


MEDMNIST_SCORER = make_scorer(
    _medmnist_distance_score, greater_is_better=False, response_method="predict_proba"
)


def load_split(dataset: str, split: str):
    split_dir = PRED_DIR / dataset / split
    gt_split = {"valid": "val", "test": "test"}[split]
    gt_file = GT_DIR / dataset / f"{gt_split}.pt"

    gt = torch.load(gt_file, map_location="cpu", weights_only=True)
    y = gt.squeeze().numpy().astype(int)

    preds = []
    names = []
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


def evaluate(y_true, y_prob, n_classes):
    y_pred = np.argmax(y_prob, axis=1) if y_prob.ndim > 1 and y_prob.shape[1] > 1 else (y_prob.ravel() >= 0.5).astype(int)
    avg = "binary" if n_classes == 2 else "macro"
    metrics = {
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "iou": jaccard_score(y_true, y_pred, average=avg, zero_division=0),
    }
    if n_classes == 2:
        prob = y_prob[:, 1] if y_prob.ndim > 1 and y_prob.shape[1] == 2 else y_prob.ravel()
        metrics["roc_auc"] = roc_auc_score(y_true, prob)
        metrics["pr_auc"] = average_precision_score(y_true, prob)
    else:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        metrics["pr_auc"] = average_precision_score(y_true, y_prob, average="macro")
    return metrics


def run_seed(X_val, y_val, X_test, y_test, seed, n_classes, args):
    from tpot import TPOTClassifier

    est = TPOTClassifier(
        scorers=[MEDMNIST_SCORER],
        scorers_weights=[1],
        population_size=args.population_size,
        generations=args.generations,
        max_time_mins=args.max_time_mins,
        early_stop=args.early_stop,
        n_jobs=args.n_jobs,
        random_state=seed,
        cv=5,
        verbose=0,
        search_space="linear",
    )
    est.fit(X_val, y_val)

    val_prob = est.predict_proba(X_val)
    val_metrics = evaluate(y_val, val_prob, n_classes)

    test_prob = est.predict_proba(X_test)
    test_metrics = evaluate(y_test, test_prob, n_classes)

    pipeline_str = str(est.fitted_pipeline_) if hasattr(est, "fitted_pipeline_") else "N/A"

    del est
    gc.collect()
    return test_metrics, val_metrics, pipeline_str


def _save_xlsx(csv_path, xlsx_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df["seed"] != "seed"].reset_index(drop=True)
    df["seed"] = df["seed"].astype(int)
    for col in CSV_COLUMNS[2:-1]:
        try:
            df[col] = df[col].astype(float)
        except (ValueError, KeyError):
            pass
    df.to_excel(xlsx_path, index=False)


def run_dataset(dataset, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"tpot_medmnist_{dataset}_results.csv"

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")

    print("Loading val set...")
    X_val, y_val, model_names = load_split(dataset, "valid")
    n_classes = len(np.unique(y_val))
    print(f"  {X_val.shape}, {len(model_names)} models, {n_classes} classes")

    print("Loading test set...")
    X_test, y_test, _ = load_split(dataset, "test")
    print(f"  {X_test.shape}")

    xlsx_path = output_dir / f"tpot_medmnist_{dataset}_results.xlsx"

    # Resume support — read completed seeds from CSV (intermediate format)
    completed = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    completed.add(int(row["seed"]))
                except (ValueError, KeyError):
                    pass  # skip repeated header rows from old runs
        print(f"Resuming: {len(completed)} seeds already done")

    # Single-seed path: subprocess called by parent spawner — never writes header
    if args.parallel_seeds == 1 and args.seed_end - args.seed_start == 1:
        seed = args.seed_start
        if seed in completed:
            return
        with open(csv_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            test_metrics, val_metrics, pipeline_str = run_seed(
                X_val, y_val, X_test, y_test, seed, n_classes, args
            )
            writer.writerow({
                "seed": seed, "dataset": dataset,
                "n_models": len(model_names), "n_classes": n_classes,
                "val_roc_auc": val_metrics["roc_auc"], "val_pr_auc": val_metrics["pr_auc"],
                "val_f1": val_metrics["f1"], "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "test_roc_auc": test_metrics["roc_auc"], "test_pr_auc": test_metrics["pr_auc"],
                "test_f1": test_metrics["f1"], "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"], "pipeline": pipeline_str,
            })
        return

    # Spawn one subprocess per seed (parallel_seeds at a time) to avoid
    # nested multiprocessing with TPOT's internal joblib workers.
    pending = [s for s in range(args.seed_start, args.seed_end) if s not in completed]
    if not pending:
        print(f"All seeds done. {xlsx_path}")
        _save_xlsx(csv_path, xlsx_path)
        return

    # Parent writes the single header row before any subprocess touches the file
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    def _spawn_seed(seed):
        cmd = [
            sys.executable, __file__,
            "--dataset", dataset,
            "--output_dir", args.output_dir,
            "--seed_start", str(seed), "--seed_end", str(seed + 1),
            "--n_jobs", str(args.n_jobs),
            "--population_size", str(args.population_size),
            "--generations", str(args.generations),
            "--early_stop", str(args.early_stop),
            "--parallel_seeds", "1",
        ] + (["--max_time_mins", str(args.max_time_mins)] if args.max_time_mins else [])
        subprocess.run(cmd, check=False)

    total = len(pending)
    print(f"Running {total} seeds ({args.parallel_seeds} in parallel)...")
    try:
        with ThreadPoolExecutor(max_workers=args.parallel_seeds) as executor:
            futures = {executor.submit(_spawn_seed, s): s for s in pending}
            done = 0
            for future in as_completed(futures):
                future.result()
                done += 1
                print(f"  [{done}/{total}] seed {futures[future]} done", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted!")

    _save_xlsx(csv_path, xlsx_path)
    print(f"Results saved to {xlsx_path}")


def main():
    parser = argparse.ArgumentParser(description="TPOT MedMNIST benchmark")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset name or 'all'")
    parser.add_argument("--output_dir", type=str, default=str(BASE / "tpot_results"))
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=100)
    parser.add_argument("--population_size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--max_time_mins", type=float, default=None)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--parallel_seeds", type=int, default=10)
    args = parser.parse_args()

    datasets = get_datasets() if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        run_dataset(dataset, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
