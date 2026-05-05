"""
ReBATE feature selection + LogisticRegression benchmark for MedMNIST classification.
"""

import argparse
import csv
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, jaccard_score,
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

METHOD_NAMES = [
    "ReliefF+RandomForest",
    "SURF+RandomForest",
    "SURFstar+RandomForest",
    "MultiSURF+RandomForest",
    "MultiSURFstar+RandomForest",
]

PARALLEL_SEEDS = 20


def _make_rebate(method_name, n_features):
    from skrebate import MultiSURF, MultiSURFstar, ReliefF, SURF, SURFstar
    n_sel = max(1, n_features // 2)
    return {
        "ReliefF+RandomForest": ReliefF(n_features_to_select=n_sel, n_neighbors=20, n_jobs=-1),
        "SURF+RandomForest": SURF(n_features_to_select=n_sel, n_jobs=-1),
        "SURFstar+RandomForest": SURFstar(n_features_to_select=n_sel, n_jobs=-1),
        "MultiSURF+RandomForest": MultiSURF(n_features_to_select=n_sel, n_jobs=-1),
        "MultiSURFstar+RandomForest": MultiSURFstar(n_features_to_select=n_sel, n_jobs=-1),
    }[method_name]


def run_feature_selection(X_val, y_val, n_features, cache_path):
    """Fit all ReBATE selectors once and save to cache. Returns selectors dict."""
    selectors = {}
    for method_name in METHOD_NAMES:
        print(f"  [{method_name}] fitting...", flush=True)
        fs = _make_rebate(method_name, n_features)
        fs.fit(X_val, y_val)
        selectors[method_name] = fs
        print(f"  [{method_name}] done", flush=True)
    with open(cache_path, "wb") as f:
        pickle.dump(selectors, f)
    print(f"Selectors saved to {cache_path}", flush=True)
    return selectors


def distance_from_ideal(*metrics):
    return float(np.sqrt(sum((1 - m) ** 2 for m in metrics)))


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


def _run_one_seed(args):
    """Worker: RF only using pre-fitted selectors. n_jobs=1 — parallelism comes from ProcessPoolExecutor."""
    seed, selectors, X_val, y_val, X_test, y_test, n_classes = args
    best_dist = float("inf")
    best_val_metrics = None
    best_test_metrics = None
    best_pipeline = None

    for method_name, fs in selectors.items():
        try:
            X_val_sel = fs.transform(X_val)
            X_test_sel = fs.transform(X_test)
            clf = LogisticRegression(random_state=seed, n_jobs=1)
            clf.fit(X_val_sel, y_val)
            val_prob = clf.predict_proba(X_val_sel)
            val_metrics = evaluate(y_val, val_prob, n_classes)
            test_prob = clf.predict_proba(X_test_sel)
            test_metrics = evaluate(y_test, test_prob, n_classes)
        except Exception as e:
            continue

        dist = distance_from_ideal(val_metrics["pr_auc"], val_metrics["roc_auc"])
        if dist < best_dist:
            best_dist = dist
            best_val_metrics = val_metrics
            best_pipeline = method_name
            best_test_metrics = test_metrics

    return seed, best_pipeline, best_val_metrics, best_test_metrics


def run_dataset(dataset, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"rebate_rf_medmnist_{dataset}_results.csv"
    features_cache = output_dir / f"rebate_medmnist_{dataset}_features.pkl"

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")

    print("Loading val set...", flush=True)
    X_val, y_val, model_names = load_split(dataset, "valid")
    if y_val.ndim > 1:
        print(f"  Skipping {dataset}: multi-label (y shape {y_val.shape})", flush=True)
        return
    n_classes = len(np.unique(y_val))
    n_models = len(model_names)
    n_features = X_val.shape[1]
    print(f"  {X_val.shape}, {n_models} models, {n_classes} classes, {n_features} features")

    print("Loading test set...", flush=True)
    X_test, y_test, _ = load_split(dataset, "test")
    print(f"  {X_test.shape}")

    if features_cache.exists():
        print(f"Loading cached selectors...", flush=True)
        with open(features_cache, "rb") as f:
            selectors = pickle.load(f)
    else:
        print("Running feature selection (once)...", flush=True)
        selectors = run_feature_selection(X_val, y_val, n_features, features_cache)

    # Resume support
    completed = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                completed.add(int(row["seed"]))
        print(f"Resuming: {len(completed)} seeds already done")

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()

    pending = [s for s in range(args.seed_start, args.seed_end) if s not in completed]
    total = len(pending)
    done = 0

    try:
        with ProcessPoolExecutor(max_workers=PARALLEL_SEEDS) as executor:
            futures = {
                executor.submit(_run_one_seed, (s, selectors, X_val, y_val, X_test, y_test, n_classes)): s
                for s in pending
            }
            for future in as_completed(futures):
                seed, best_pipeline, best_val_metrics, best_test_metrics = future.result()
                done += 1
                if best_pipeline is None:
                    print(f"  [{done}/{total}] Seed {seed}: all methods failed", flush=True)
                    continue
                writer.writerow({
                    "seed": seed,
                    "dataset": dataset,
                    "n_models": n_models,
                    "n_classes": n_classes,
                    "val_roc_auc": best_val_metrics["roc_auc"],
                    "val_pr_auc": best_val_metrics["pr_auc"],
                    "val_f1": best_val_metrics["f1"],
                    "val_precision": best_val_metrics["precision"],
                    "val_recall": best_val_metrics["recall"],
                    "test_roc_auc": best_test_metrics["roc_auc"],
                    "test_pr_auc": best_test_metrics["pr_auc"],
                    "test_f1": best_test_metrics["f1"],
                    "test_precision": best_test_metrics["precision"],
                    "test_recall": best_test_metrics["recall"],
                    "pipeline": best_pipeline,
                })
                csv_file.flush()
                print(f"  [{done}/{total}] Seed {seed}: {best_pipeline} valAUC={best_val_metrics['roc_auc']:.4f} testAUC={best_test_metrics['roc_auc']:.4f}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted! Results saved.")
    finally:
        csv_file.close()

    print(f"Results saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="ReBATE MedMNIST benchmark")
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default=str(BASE / "tpot_results"))
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=100)
    args = parser.parse_args()

    datasets = get_datasets() if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        run_dataset(dataset, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
