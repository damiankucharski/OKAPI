"""
STARCOP 100-seed OKAPI benchmark.

Runs OKAPI GP evolution on STARCOP methane detection data across multiple seeds,
evaluates Pareto trees on both validation and test sets, and saves results to CSV.

Usage:
    # Run from STARCOP/ directory (needs pixi env for okapi + starcop packages)
    cd /home/s/Projects/OKAPI_PAPER_AFTER_REVIEW/STARCOP
    pixi run python ../REBUTTAL_WORK/starcop_benchmark.py --config_id 3 --seed_start 0 --seed_end 100

    # Quick test with 2 seeds
    pixi run python ../REBUTTAL_WORK/starcop_benchmark.py --config_id 3 --seed_start 0 --seed_end 2

    # Specify branch name (for tracking main vs dev)
    pixi run python ../REBUTTAL_WORK/starcop_benchmark.py --config_id 3 --seed_start 0 --seed_end 100 --branch dev
"""

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

# Set backend before any okapi imports
os.environ["DEVICE"] = "cuda"
os.environ["BACKEND"] = "pytorch"

import torch

# Reduce GPU memory fragmentation for large tensor operations
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from loguru import logger

from okapi import Okapi
from okapi.callback import FitnessNoChangeEarlyStoppingCallback, MemoryCleanupCallback
from okapi.operators import (
    CLOSE_THRESHOLD,
    FAR_THRESHOLD,
    MAX,
    MEAN,
    MIN,
    WEIGHTED_MEAN,
)
from okapi.pareto import maximize
from starcop.okapi_fitness import (
    f1_fitness,
    iou_fitness,
    precision_fitness,
    recall_fitness,
)

# Paths relative to STARCOP/ directory
STARCOP_DIR = Path(__file__).resolve().parent.parent.parent / "STARCOP"
OKAPI_DATA_DIR = STARCOP_DIR / "okapi_data"
GT_VAL_PATH = OKAPI_DATA_DIR / "gt" / "y_val.pt"
GT_TEST_PATH = OKAPI_DATA_DIR / "gt" / "y_test.pt"

# Experiment configurations for different model subsets
EXPERIMENTS = {
    "all": {
        "val_dir": OKAPI_DATA_DIR / "val",
        "test_dir": OKAPI_DATA_DIR / "test",
        "description": "All 13 models",
    },
    "no_specialists": {
        "val_dir": OKAPI_DATA_DIR / "val_no_specialists",
        "test_dir": OKAPI_DATA_DIR / "test_no_specialists",
        "description": "11 models (no easy_specialist, no hard_specialist)",
    },
}

# Default (for backward compatibility)
VAL_PREDS_DIR = OKAPI_DATA_DIR / "val"
TEST_PREDS_DIR = OKAPI_DATA_DIR / "test"

# Default output directory
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "starcop_results"

# Experiment configs (matching MedMNIST pattern: 6 configs)
# Config 3 is the primary one matching existing STARCOP experiments
CONFIGS = {
    3: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": True,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "description": "F1+Precision+Recall, all ops, parsimony",
    },
    6: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": False,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "description": "F1+Precision+Recall, all ops, NO parsimony",
    },
    33: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": True,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "tournament_size": 3,
        "description": "F1+Precision+Recall, all ops, parsimony, tournament_size=3",
    },
    304: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": True,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "tournament_size": 4,
        "iterations": 50,
        "description": "F1+Precision+Recall, all ops, parsimony, pop=8 tournament=4 50 gens (larger search budget)",
    },
    305: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": True,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "tournament_size": 4,
        "iterations": 70,
        "description": "F1+Precision+Recall, all ops, parsimony, pop=8 tournament=4 70 gens (even larger search budget)",
    },
    306: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": True,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "tournament_size": 4,
        "iterations": 100,
        "description": "F1+Precision+Recall, all ops, parsimony, pop=8 tournament=4 100 gens (huge search budget)",
    },
    307: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": True,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "population_size": 13,
        "population_multiplier": 6,
        "tournament_size": 3,
        "iterations": 50,
        "description": "F1+Precision+Recall, all ops, parsimony, max pop=13 mult=6 tournament=3 50 gens (wide and deep search)",
    },
    308: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": True,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "population_size": 13,
        "population_multiplier": 6,
        "tournament_size": 3,
        "iterations": 75,
        "description": "F1+Precision+Recall, all ops, parsimony, max pop=13 mult=6 tournament=3 75 gens (extended wide search)",
    },
    309: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": True,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "population_size": 13,
        "population_multiplier": 8,
        "tournament_size": 3,
        "iterations": 50,
        "description": "F1+Precision+Recall, all ops, parsimony, max pop=13 mult=8 tournament=3 50 gens (ultra wide search)",
    },
    310: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": True,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "population_size": 13,
        "population_multiplier": 8,
        "tournament_size": 3,
        "iterations": 75,
        "description": "F1+Precision+Recall, all ops, parsimony, max pop=13 mult=8 tournament=3 75 gens (ultra wide & deep)",
    },
    311: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": True,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "population_size": 13,
        "population_multiplier": 6,
        "tournament_size": 3,
        "iterations": 100,
        "description": "F1+Precision+Recall, all ops, parsimony, max pop=13 mult=6 tournament=3 100 gens (extreme depth to force overfitting)",
    },
    312: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": True,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "population_size": 13,
        "population_multiplier": 8,
        "tournament_size": 5,
        "iterations": 50,
        "description": "F1+Precision+Recall, all ops, parsimony, max pop=13 mult=8 tournament=5 50 gens (ultra wide + aggressive selection)",
    },
    607: {
        "objective_functions": (f1_fitness, precision_fitness, recall_fitness),
        "objectives": (maximize, maximize, maximize),
        "minimize_node_count": False,
        "allowed_ops": (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD),
        "population_size": 13,
        "population_multiplier": 6,
        "tournament_size": 3,
        "iterations": 50,
        "description": "F1+Precision+Recall, all ops, NO parsimony, max pop=13 mult=6 tournament=3 50 gens (wide and deep search)",
    },
    # Configs 1-2, 4-5 to be defined if needed
}

# Shared hyperparameters
POPULATION_SIZE = 8
POPULATION_MULTIPLIER = 4
TOURNAMENT_SIZE = 5
ITERATIONS = 20
MUTATION_STRENGTH = 0.1
EARLY_STOPPING_PATIENCE = 10

CSV_COLUMNS = [
    "seed",
    "config_id",
    "branch",
    "tree_id",
    "n_val_nodes",
    "n_op_nodes",
    "n_unique_models",
    "unique_models",
    "val_f1",
    "val_precision",
    "val_recall",
    "val_iou",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_iou",
    "optimize_f1",
    "optimize_precision",
    "optimize_recall",
    "minimize_node_count",
    "use_basic_ops_only",
]


def build_okapi(config_id: int, seed: int, val_preds_dir: Path) -> Okapi:
    """Instantiate Okapi with the given config and seed."""
    config = CONFIGS[config_id]
    tournament_size = config.get("tournament_size", TOURNAMENT_SIZE)
    population_size = config.get("population_size", POPULATION_SIZE)
    population_multiplier = config.get("population_multiplier", POPULATION_MULTIPLIER)
    return Okapi(
        preds_source=val_preds_dir,
        gt_path=GT_VAL_PATH,
        population_size=population_size,
        population_multiplier=population_multiplier,
        tournament_size=tournament_size,
        minimize_node_count=config["minimize_node_count"],
        objective_functions=config["objective_functions"],
        objectives=config["objectives"],
        allowed_ops=config["allowed_ops"],
        callbacks=[
            MemoryCleanupCallback(gc_every_n_generations=1, clear_cuda_cache=True),
            FitnessNoChangeEarlyStoppingCallback(n_iterations=EARLY_STOPPING_PATIENCE),
        ],
        seed=seed,
        mutation_strength=MUTATION_STRENGTH,
    )


def evaluate_tree_on_split(tree, preds_dir: Path, gt_tensor) -> dict:
    """Evaluate a loaded tree (no other tensors on GPU) on a given split."""
    prediction = tree.evaluation
    metrics = {
        "f1": f1_fitness(prediction, gt_tensor),
        "precision": precision_fitness(prediction, gt_tensor),
        "recall": recall_fitness(prediction, gt_tensor),
        "iou": iou_fitness(prediction, gt_tensor),
    }
    tree._clean_evals()
    return metrics


def run_single_seed(
    config_id: int,
    seed: int,
    branch: str,
    gt_test: torch.Tensor,
    trees_dir: Path,
    val_preds_dir: Path,
    test_preds_dir: Path,
) -> list[dict]:
    """Run one OKAPI evolution seed and evaluate all Pareto trees.

    Memory strategy:
    1. Train OKAPI (val tensors on GPU ~9 GB)
    2. Collect val metrics + IoU using the already-loaded val tensors
    3. Save tree architectures, then DELETE Okapi to free GPU
    4. Load trees one-by-one with test tensors for test evaluation
    """
    config = CONFIGS[config_id]
    basic_ops = {MEAN, MIN, MAX}

    logger.info(f"[seed={seed}] Building OKAPI...")
    okp = build_okapi(config_id, seed, val_preds_dir)

    iterations = config.get("iterations", ITERATIONS)
    logger.info(f"[seed={seed}] Training for {iterations} iterations...")
    okp.train(iterations)

    pareto_trees = okp.pareto_trees
    pareto_fitnesses = okp.pareto_fitnesses
    n_trees = len(pareto_trees)
    logger.info(f"[seed={seed}] Evolution done. {n_trees} Pareto trees found.")

    # Save trees and collect val metrics while val tensors are still loaded
    seed_trees_dir = trees_dir / f"config_{config_id}" / f"seed_{seed}"
    seed_trees_dir.mkdir(parents=True, exist_ok=True)

    tree_infos = []
    for tree_id, (tree, fitnesses) in enumerate(zip(pareto_trees, pareto_fitnesses)):
        tree_path = seed_trees_dir / f"pareto_tree_{tree_id}.tree"
        tree.save_tree_architecture(tree_path)

        # Val metrics: use the tree's existing evaluation (val tensors already loaded)
        prediction = tree.predict(clear_cache=True)
        val_metrics = {
            "f1": f1_fitness(prediction, okp.gt_tensor),
            "precision": precision_fitness(prediction, okp.gt_tensor),
            "recall": recall_fitness(prediction, okp.gt_tensor),
            "iou": iou_fitness(prediction, okp.gt_tensor),
        }
        del prediction

        tree_infos.append(
            {
                "tree_path": tree_path,
                "tree_id": tree_id,
                "n_val_nodes": len(tree.value_nodes),
                "n_op_nodes": len(tree.op_nodes),
                "n_unique_models": len(tree.unique_value_node_ids),
                "unique_models": str(tree.unique_value_node_ids),
                "val_metrics": {k: float(v) for k, v in val_metrics.items()},
            }
        )

    # Free all val tensors from GPU before loading test tensors
    # Explicitly clear loop variables that hold references to GPU tensors!
    if "tree" in locals():
        del tree
    if "fitnesses" in locals():
        del fitnesses
    del okp, pareto_trees, pareto_fitnesses
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate on test set: load test tensors to CPU to avoid GPU memory pressure.
    # The fitness functions handle CPU tensors fine.
    from okapi.tree import Tree

    # Load test tensors to CPU once, reuse across all trees
    test_tensors_cpu = {}
    for pt_file in test_preds_dir.glob("*"):
        test_tensors_cpu[pt_file.name] = torch.load(
            pt_file, map_location="cpu", weights_only=True
        )
    gt_test_cpu = gt_test.cpu() if gt_test.is_cuda else gt_test

    results = []
    for info in tree_infos:
        loaded_tree = Tree.load_tree_architecture(info["tree_path"])
        # Load tensors into tree value nodes from our CPU cache
        for vnode in loaded_tree.value_nodes:
            vnode.value = test_tensors_cpu[vnode.id]

        prediction = loaded_tree.predict(clear_cache=True)
        test_metrics = {
            "f1": f1_fitness(prediction, gt_test_cpu),
            "precision": precision_fitness(prediction, gt_test_cpu),
            "recall": recall_fitness(prediction, gt_test_cpu),
            "iou": iou_fitness(prediction, gt_test_cpu),
        }
        del loaded_tree, prediction

        row = {
            "seed": seed,
            "config_id": config_id,
            "branch": branch,
            "tree_id": info["tree_id"],
            "n_val_nodes": info["n_val_nodes"],
            "n_op_nodes": info["n_op_nodes"],
            "n_unique_models": info["n_unique_models"],
            "unique_models": info["unique_models"],
            "val_f1": info["val_metrics"]["f1"],
            "val_precision": info["val_metrics"]["precision"],
            "val_recall": info["val_metrics"]["recall"],
            "val_iou": info["val_metrics"]["iou"],
            "test_f1": float(test_metrics["f1"]),
            "test_precision": float(test_metrics["precision"]),
            "test_recall": float(test_metrics["recall"]),
            "test_iou": float(test_metrics["iou"]),
            "optimize_f1": f1_fitness in config["objective_functions"],
            "optimize_precision": precision_fitness in config["objective_functions"],
            "optimize_recall": recall_fitness in config["objective_functions"],
            "minimize_node_count": config["minimize_node_count"],
            "use_basic_ops_only": set(config["allowed_ops"]) == basic_ops,
        }
        results.append(row)

    # Clean up CPU test tensors
    del test_tensors_cpu
    if "gt_test_cpu" in locals():
        del gt_test_cpu
    if "loaded_tree" in locals():
        del loaded_tree
    if "prediction" in locals():
        del prediction

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="STARCOP OKAPI benchmark (100 seeds)")
    parser.add_argument(
        "--config_id",
        type=int,
        default=3,
        choices=list(CONFIGS.keys()),
        help="Experiment config ID",
    )
    parser.add_argument(
        "--seed_start", type=int, default=0, help="First seed (inclusive)"
    )
    parser.add_argument(
        "--seed_end", type=int, default=100, help="Last seed (exclusive)"
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="OKAPI branch name (for tracking in results)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=list(EXPERIMENTS.keys()),
        help="Model subset to use: 'all' (13 models) or 'no_specialists' (11 models)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for CSV and trees",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress library logs, keep only progress output",
    )
    args = parser.parse_args()

    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
        import logging

        logging.disable(logging.WARNING)
        # Disable tqdm progress bars from okapi internals
        import unittest.mock
        import tqdm

        tqdm.tqdm = unittest.mock.MagicMock(side_effect=lambda it, *a, **kw: it)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trees_dir = output_dir / "trees" / args.branch
    trees_dir.mkdir(parents=True, exist_ok=True)

    # Get experiment-specific paths
    exp_config = EXPERIMENTS[args.experiment]
    val_preds_dir = exp_config["val_dir"]
    test_preds_dir = exp_config["test_dir"]

    # Include experiment name in CSV filename to avoid mixing results
    csv_path = (
        output_dir
        / f"starcop_results_{args.branch}_config{args.config_id}_{args.experiment}.csv"
    )

    # Verify data exists
    for path in [val_preds_dir, test_preds_dir, GT_VAL_PATH, GT_TEST_PATH]:
        if not path.exists():
            logger.error(f"Required path not found: {path}")
            sys.exit(1)

    print(f"Config: {args.config_id} ({CONFIGS[args.config_id]['description']})")
    print(f"Experiment: {args.experiment} ({exp_config['description']})")
    print(f"Val dir: {val_preds_dir}")
    print(f"Seeds: {args.seed_start} to {args.seed_end - 1}")
    print(f"Branch: {args.branch}")
    print(f"Output: {csv_path}")

    # Load test ground truth once (stays in memory for all seeds)
    print("Loading test ground truth...")
    gt_test = torch.load(GT_TEST_PATH, map_location="cpu", weights_only=True)

    # Determine which seeds are already done (for resuming interrupted runs)
    completed_seeds = set()
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_seeds.add(int(row["seed"]))
        print(f"Found {len(completed_seeds)} already-completed seeds in {csv_path}")

    # Open CSV in append mode
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()

    seeds = range(args.seed_start, args.seed_end)
    total = len(seeds)
    done = 0

    try:
        for seed in seeds:
            if seed in completed_seeds:
                done += 1
                continue

            t0 = time.time()
            gpu_alloc = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"[{done + 1}/{total}] Seed {seed} (GPU: {gpu_alloc:.1f}G alloc, {gpu_reserved:.1f}G reserved)...",
                end=" ",
                flush=True,
            )

            results = run_single_seed(
                config_id=args.config_id,
                seed=seed,
                branch=args.branch,
                gt_test=gt_test,
                trees_dir=trees_dir,
                val_preds_dir=val_preds_dir,
                test_preds_dir=test_preds_dir,
            )

            # Write results incrementally
            for row in results:
                writer.writerow(row)
            csv_file.flush()

            elapsed = time.time() - t0
            done += 1
            remaining = total - done
            eta_min = (elapsed * remaining) / 60
            print(f"{len(results)} trees, {elapsed:.1f}s. ETA: {eta_min:.0f}min")
    except KeyboardInterrupt:
        print("\nInterrupted! Results so far are saved.")
    finally:
        csv_file.close()

    print(f"Done. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
