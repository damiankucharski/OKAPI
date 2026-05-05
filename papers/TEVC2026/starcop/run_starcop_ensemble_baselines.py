"""
STARCOP ensemble baseline comparisons.

Computes test-set performance for various ensemble strategies on STARCOP:
- Top-K naive average (K=1..13, ranked by val F1)
- Weighted average (optimize weights on val via scipy)
- Random-K average (100 random subsets per K)

Memory-efficient: processes images in chunks, never flattens all at once.

Usage:
    python starcop/run_starcop_ensemble_baselines.py
"""

import csv
import gc
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize as scipy_minimize

# Paths
STARCOP_DIR = Path(__file__).resolve().parent.parent.parent / "STARCOP"
OKAPI_DATA_DIR = STARCOP_DIR / "okapi_data"
GT_VAL_PATH = OKAPI_DATA_DIR / "gt" / "y_val.pt"
GT_TEST_PATH = OKAPI_DATA_DIR / "gt" / "y_test.pt"
VAL_DIR = OKAPI_DATA_DIR / "val"
TEST_DIR = OKAPI_DATA_DIR / "test"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "starcop_results"

# 13 models in order
MODEL_NAMES = sorted([p.name for p in VAL_DIR.glob("*.pt")])

# Base model val F1 scores (from analyze_starcop_100seeds.py)
BASE_MODEL_VAL_F1 = {
    "focal_loss.pt": 0.6806,
    "posweight_05.pt": 0.7013,
    "tversky_f05.pt": 0.5980,
    "bce_dice.pt": 0.6594,
    "tversky_f065.pt": 0.6270,
    "bce_baseline.pt": 0.7002,
    "posweight_5.pt": 0.6239,
    "tversky_f1.pt": 0.5531,
    "tversky_f2.pt": 0.5412,
    "tversky_f15.pt": 0.5326,
    "posweight_10.pt": 0.5089,
    "hard_specialist.pt": 0.5604,
    "easy_specialist.pt": 0.6560,
}

# Models ranked by val F1 (best first)
MODELS_RANKED = sorted(MODEL_NAMES, key=lambda m: BASE_MODEL_VAL_F1.get(m, 0), reverse=True)


def compute_tp_fp_fn_chunked(split_dir: Path, gt_path: Path, model_names: list,
                              weights: np.ndarray | None = None,
                              threshold: float = 0.5, chunk_size: int = 50):
    """Compute TP/FP/FN by processing images in chunks. Memory-efficient.

    Args:
        model_names: which models to average
        weights: if None, uniform average; otherwise weighted combination
        chunk_size: number of images to process at once
    """
    gt = torch.load(gt_path, map_location="cpu", weights_only=True)
    n_images = gt.shape[0]
    tp_total, fp_total, fn_total = 0, 0, 0

    for start in range(0, n_images, chunk_size):
        end = min(start + chunk_size, n_images)
        gt_chunk = gt[start:end].numpy()  # [chunk, 512, 512]

        # Accumulate weighted average for this chunk
        avg = np.zeros_like(gt_chunk, dtype=np.float32)
        for i, name in enumerate(model_names):
            pred = torch.load(split_dir / name, map_location="cpu", weights_only=True)
            pred_chunk = pred[start:end].numpy()
            w = weights[i] if weights is not None else 1.0 / len(model_names)
            avg += pred_chunk * w
            del pred, pred_chunk

        binary = (avg >= threshold).astype(np.int32)
        gt_int = gt_chunk.astype(np.int32)
        tp_total += int(np.sum(binary & gt_int))
        fp_total += int(np.sum(binary & (1 - gt_int)))
        fn_total += int(np.sum((1 - binary) & gt_int))
        del avg, binary, gt_chunk, gt_int

    return tp_total, fp_total, fn_total


def metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "iou": iou}


def compute_metrics_for_ensemble(split_dir, gt_path, model_names, weights=None):
    """Full metrics for an ensemble, memory-efficient."""
    tp, fp, fn = compute_tp_fp_fn_chunked(split_dir, gt_path, model_names, weights)
    return metrics_from_counts(tp, fp, fn)


def weighted_average_optimize(n_subsample: int = 500_000):
    """Optimize 13 weights on subsampled val pixels, evaluate on test."""
    # Subsample val pixels for optimization (full val = 680*512*512 = 178M pixels)
    gt_val = torch.load(GT_VAL_PATH, map_location="cpu", weights_only=True)
    n_images = gt_val.shape[0]
    n_pixels = n_images * 512 * 512

    rng = np.random.RandomState(42)
    # Pick random pixel indices
    idx = rng.choice(n_pixels, size=min(n_subsample, n_pixels), replace=False)
    img_idx = idx // (512 * 512)
    pix_idx = idx % (512 * 512)
    row_idx = pix_idx // 512
    col_idx = pix_idx % 512

    # Extract subsampled val data: [n_subsample, 13] and gt [n_subsample]
    gt_sub = gt_val.numpy()[img_idx, row_idx, col_idx].astype(np.int32)
    del gt_val

    val_sub = np.zeros((len(idx), len(MODEL_NAMES)), dtype=np.float32)
    for i, name in enumerate(MODEL_NAMES):
        pred = torch.load(VAL_DIR / name, map_location="cpu", weights_only=True)
        val_sub[:, i] = pred.numpy()[img_idx, row_idx, col_idx]
        del pred
    gc.collect()

    print(f"  Optimizing on {len(idx):,} subsampled val pixels...")

    def neg_f1(logits):
        w = np.exp(logits - logits.max())
        w = w / w.sum()
        avg = val_sub @ w
        binary = (avg >= 0.5).astype(np.int32)
        tp = np.sum(binary & gt_sub)
        fp = np.sum(binary & (1 - gt_sub))
        fn = np.sum((1 - binary) & gt_sub)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        return -f1

    best_result = None
    best_f1 = -1.0
    for seed in range(5):
        r = np.random.RandomState(seed)
        x0 = r.randn(13) * 0.1
        result = scipy_minimize(neg_f1, x0, method="Nelder-Mead",
                                options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
        if -result.fun > best_f1:
            best_f1 = -result.fun
            best_result = result

    w_final = np.exp(best_result.x - best_result.x.max())
    w_final = w_final / w_final.sum()
    print(f"  Optimized weights (val F1={best_f1:.4f}):")
    for m, w in zip(MODEL_NAMES, w_final):
        if w > 0.01:
            print(f"    {m}: {w:.3f}")

    # Evaluate on full test set (chunked)
    tp, fp, fn = compute_tp_fp_fn_chunked(TEST_DIR, GT_TEST_PATH, MODEL_NAMES, w_final)
    metrics = metrics_from_counts(tp, fp, fn)
    n_effective = int((w_final > 0.01).sum())
    return metrics, w_final, n_effective


def random_k_average(k: int, n_trials: int = 100, seed: int = 42):
    """Random-K average on test set. Loads one model at a time per trial."""
    rng = np.random.RandomState(seed)
    model_list = list(MODEL_NAMES)
    f1s = []

    for trial in range(n_trials):
        selected = list(rng.choice(model_list, size=k, replace=False))
        tp, fp, fn = compute_tp_fp_fn_chunked(TEST_DIR, GT_TEST_PATH, selected)
        m = metrics_from_counts(tp, fp, fn)
        f1s.append(m["f1"])

    return {"mean_f1": np.mean(f1s), "std_f1": np.std(f1s),
            "min_f1": np.min(f1s), "max_f1": np.max(f1s)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # --- Individual models ---
    print("=== Individual Models ===")
    for name in MODELS_RANKED:
        m = compute_metrics_for_ensemble(TEST_DIR, GT_TEST_PATH, [name])
        results.append({
            "method": f"individual_{name.replace('.pt', '')}",
            "test_f1": m["f1"], "test_precision": m["precision"],
            "test_recall": m["recall"], "test_iou": m["iou"],
            "n_models_used": 1,
            "notes": f"val_f1={BASE_MODEL_VAL_F1[name]:.4f}",
        })
        print(f"  {name}: F1={m['f1']:.4f}")

    # --- Top-K naive average ---
    print("\n=== Top-K Naive Average ===")
    for k in range(1, 14):
        selected = MODELS_RANKED[:k]
        m = compute_metrics_for_ensemble(TEST_DIR, GT_TEST_PATH, selected)
        models_used = ", ".join([n.replace(".pt", "") for n in selected])
        results.append({
            "method": f"top_{k}_avg",
            "test_f1": m["f1"], "test_precision": m["precision"],
            "test_recall": m["recall"], "test_iou": m["iou"],
            "n_models_used": k,
            "notes": models_used,
        })
        print(f"  Top-{k:2d}: F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  IoU={m['iou']:.4f}")

    # --- Weighted average ---
    print("\n=== Weighted Average (optimized on val) ===")
    wt_metrics, wt_weights, wt_n_eff = weighted_average_optimize()
    results.append({
        "method": "weighted_avg_optimized",
        "test_f1": wt_metrics["f1"], "test_precision": wt_metrics["precision"],
        "test_recall": wt_metrics["recall"], "test_iou": wt_metrics["iou"],
        "n_models_used": wt_n_eff,
        "notes": f"scipy Nelder-Mead, {wt_n_eff} effective models (w>0.01)",
    })
    print(f"  Weighted avg: F1={wt_metrics['f1']:.4f}  ({wt_n_eff} effective models)")

    # --- Random-K average ---
    print("\n=== Random-K Average (100 trials each) ===")
    for k in [2, 3, 4, 5, 7, 10, 13]:
        print(f"  Computing Random-{k}...")
        rk = random_k_average(k)
        results.append({
            "method": f"random_{k}_avg",
            "test_f1": rk["mean_f1"], "test_precision": 0.0,
            "test_recall": 0.0, "test_iou": 0.0,
            "n_models_used": k,
            "notes": f"mean+/-std over 100 trials: {rk['mean_f1']:.4f}+/-{rk['std_f1']:.4f}, range=[{rk['min_f1']:.4f}, {rk['max_f1']:.4f}]",
        })
        print(f"  Random-{k:2d}: F1={rk['mean_f1']:.4f} +/- {rk['std_f1']:.4f}")

    # --- Save CSV ---
    out_path = OUTPUT_DIR / "starcop_ensemble_baselines.csv"
    fieldnames = ["method", "test_f1", "test_precision", "test_recall", "test_iou",
                  "n_models_used", "notes"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
