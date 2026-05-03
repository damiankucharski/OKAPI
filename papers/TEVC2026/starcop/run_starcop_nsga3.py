"""
NSGA-III weighted average baseline for STARCOP.

Optimizes 13 weights (one per base model) using NSGA-III with 3 objectives:
F1, Precision, Recall — matching OKAPI's config 3. Weights are constrained
to be non-negative and sum to 1. The weighted average of model predictions
is thresholded at 0.5 to produce binary segmentation masks.

Operates on full tensors on GPU (no subsampling), same as the learned weighting
baseline. All 92 solutions per generation are evaluated in a single batched
matrix multiply on GPU.

Produces a Pareto front per seed; the best solution per seed is selected by val F1
(matching OKAPI evaluation protocol).

Usage:
    cd /home/s/Projects/OKAPI_PAPER_AFTER_REVIEW/STARCOP
    pixi run python ../REBUTTAL_WORK_CLEAN/starcop_nsga3_baseline.py --seed_start 0 --seed_end 100
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.util.ref_dirs import get_reference_directions

# Paths
STARCOP_DIR = Path(__file__).resolve().parent.parent.parent / "STARCOP"
OKAPI_DATA_DIR = STARCOP_DIR / "okapi_data"
GT_VAL_PATH = OKAPI_DATA_DIR / "gt" / "y_val.pt"
GT_TEST_PATH = OKAPI_DATA_DIR / "gt" / "y_test.pt"
VAL_DIR = OKAPI_DATA_DIR / "val"
TEST_DIR = OKAPI_DATA_DIR / "test"
OUTPUT_DIR = Path(__file__).resolve().parent / "starcop_results"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "starcop_results" / "nsga3_weights"

MODEL_NAMES = sorted([p.name for p in VAL_DIR.glob("*.pt")])
N_MODELS = len(MODEL_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stacked_predictions(pred_dir: Path, device: torch.device) -> torch.Tensor:
    """Load all 13 model predictions as [13, total_pixels] on the given device."""
    preds = []
    for name in MODEL_NAMES:
        p = torch.load(pred_dir / name, map_location="cpu", weights_only=True)
        preds.append(p.reshape(1, -1))  # [1, N*H*W]
    stacked = torch.cat(preds, dim=0).to(device)  # [13, total_pixels]
    return stacked


def compute_metrics_from_counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "iou": iou}


class SegmentationEnsembleProblem(Problem):
    """NSGA-III problem: optimize 13 model weights for F1, Precision, Recall.

    All solutions in a generation are evaluated in a single batched GPU matmul.
    """

    def __init__(self, val_preds_gpu: torch.Tensor, gt_val_gpu: torch.Tensor):
        """
        val_preds_gpu: [13, total_pixels] float32 on GPU
        gt_val_gpu: [total_pixels] bool on GPU
        """
        self.val_preds = val_preds_gpu  # [13, P]
        self.gt_val = gt_val_gpu  # [P]
        super().__init__(
            n_var=N_MODELS,
            n_obj=3,  # F1, Precision, Recall (all maximized → negated)
            n_eq_constr=1,  # sum of weights = 1
            xl=0.0,
            xu=1.0,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_solutions = X.shape[0]
        all_f1 = np.zeros(n_solutions)
        all_prec = np.zeros(n_solutions)
        all_rec = np.zeros(n_solutions)

        gt = self.gt_val  # [P] bool on GPU

        for i in range(n_solutions):
            W = torch.tensor(
                X[i], dtype=torch.float32, device=self.val_preds.device
            )  # [13]
            avg = W @ self.val_preds  # [P]
            binary = avg >= 0.5
            del avg

            tp = (binary & gt).sum().item()
            fp = (binary & ~gt).sum().item()
            fn = (~binary & gt).sum().item()
            del binary

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            all_f1[i] = -f
            all_prec[i] = -p
            all_rec[i] = -r

        out["F"] = np.column_stack([all_f1, all_prec, all_rec])
        out["H"] = np.sum(X, axis=1) - 1.0


def evaluate_weights_on_split(
    preds: torch.Tensor, gt: torch.Tensor, weights: np.ndarray
) -> dict:
    """Evaluate a single weight vector on a split (val or test)."""
    W = torch.tensor(weights, dtype=torch.float32, device=preds.device)  # [13]
    avg = W @ preds  # [P]
    binary = avg >= 0.5

    tp = int((binary & gt).sum())
    fp = int((binary & ~gt).sum())
    fn = int((~binary & gt).sum())
    del W, avg, binary
    return compute_metrics_from_counts(tp, fp, fn)


CSV_COLUMNS = [
    "seed",
    "solution_id",
    "val_f1",
    "val_precision",
    "val_recall",
    "val_iou",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_iou",
    "n_effective_models",
]


def main():
    parser = argparse.ArgumentParser(
        description="STARCOP NSGA-III weighted average baseline"
    )
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=92)
    parser.add_argument("--n_gen", type=int, default=100)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "starcop_nsga3_results.csv"

    print(f"Device: {DEVICE}")

    print("Loading val predictions to GPU...")
    val_preds_gpu = load_stacked_predictions(VAL_DIR, DEVICE)  # [13, ~178M] on GPU
    gt_val_gpu = (
        torch.load(GT_VAL_PATH, map_location="cpu", weights_only=True)
        .reshape(-1)
        .bool()
        .to(DEVICE)
    )
    print(f"  Val: {val_preds_gpu.shape}, GT: {gt_val_gpu.shape}")
    gpu_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory after val: {gpu_gb:.1f} GB")

    print("Loading test predictions to CPU...")
    test_preds_cpu = load_stacked_predictions(
        TEST_DIR, torch.device("cpu")
    )  # [13, ~90M] on CPU
    gt_test_cpu = (
        torch.load(GT_TEST_PATH, map_location="cpu", weights_only=True)
        .reshape(-1)
        .bool()
    )
    print(f"  Test: {test_preds_cpu.shape}, GT: {gt_test_cpu.shape} (CPU)")

    # Reference directions for 3 objectives
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    print(
        f"NSGA-III: pop_size={args.pop_size}, n_gen={args.n_gen}, ref_dirs={len(ref_dirs)}"
    )

    # Resume support
    completed_seeds = set()
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_seeds.add(int(row["seed"]))
        print(f"Found {len(completed_seeds)} already-completed seeds")

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()

    problem = SegmentationEnsembleProblem(val_preds_gpu, gt_val_gpu)

    seeds = range(args.seed_start, args.seed_end)
    total = len(seeds)
    done = 0

    try:
        for seed in seeds:
            if seed in completed_seeds:
                done += 1
                continue

            t0 = time.time()
            print(f"[{done + 1}/{total}] Seed {seed}...", end=" ", flush=True)

            algorithm = NSGA3(pop_size=args.pop_size, ref_dirs=ref_dirs)
            result = pymoo_minimize(
                problem,
                algorithm,
                ("n_gen", args.n_gen),
                seed=seed,
                verbose=False,
            )

            # result.X: [n_solutions, 13], result.F: [n_solutions, 3] (negated)
            if result.X is None or len(result.X) == 0:
                print("no solutions found, skipping")
                done += 1
                continue

            weights_set = result.X if result.X.ndim == 2 else result.X.reshape(1, -1)
            n_solutions = len(weights_set)
            seed_weights_dir = WEIGHTS_DIR / f"seed_{seed}"
            seed_weights_dir.mkdir(parents=True, exist_ok=True)

            for sol_id, weights in enumerate(weights_set):
                val_m = evaluate_weights_on_split(val_preds_gpu, gt_val_gpu, weights)
                test_m = evaluate_weights_on_split(test_preds_cpu, gt_test_cpu, weights)
                n_effective = int((weights > 0.01).sum())

                # Mirror the OKAPI tree artifact layout: one file per Pareto solution.
                np.savez_compressed(
                    seed_weights_dir / f"pareto_weights_{sol_id}.npz",
                    weights=np.asarray(weights, dtype=np.float32),
                    model_names=np.asarray(MODEL_NAMES, dtype="U64"),
                )

                row = {
                    "seed": seed,
                    "solution_id": sol_id,
                    "val_f1": val_m["f1"],
                    "val_precision": val_m["precision"],
                    "val_recall": val_m["recall"],
                    "val_iou": val_m["iou"],
                    "test_f1": test_m["f1"],
                    "test_precision": test_m["precision"],
                    "test_recall": test_m["recall"],
                    "test_iou": test_m["iou"],
                    "n_effective_models": n_effective,
                }
                writer.writerow(row)

            csv_file.flush()
            elapsed = time.time() - t0
            done += 1
            remaining = total - done
            eta_min = (elapsed * remaining) / 60
            print(f"{n_solutions} solutions, {elapsed:.1f}s. ETA: {eta_min:.0f}min")

    except KeyboardInterrupt:
        print("\nInterrupted! Results so far are saved.")
    finally:
        csv_file.close()

    print(f"Done. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
