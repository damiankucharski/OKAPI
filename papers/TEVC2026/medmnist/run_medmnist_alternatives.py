# run_single_alternative.py

import argparse
import json
from pathlib import Path
from alternative_methods_benchmark import AlternativeMethods, datasets, path_preds, path_gt
import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run alternative methods benchmark for a single dataset and seed.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for NSGA3")
    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed

    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{start_time}] Running alternative methods for dataset={dataset}, seed={seed}")

    # Output file paths for all three methods
    out_dir = Path("/home/s/Git/python/GIRAFFE/benchmark/jsons_alternative")
    simple_path = out_dir / f"{dataset}_evaluation_simple_average_0_seed_{seed}.json"
    lr_path = out_dir / f"{dataset}_evaluation_logistic_regression_0_seed_{seed}.json"
    nsga3_prefix = out_dir / f"{dataset}_evaluation_nsga3_"
    # Check if outputs exist for this dataset/seed
    if seed == 0:
        # For seed 0, check all three outputs
        nsga3_exists = any(p.name.startswith(f"{dataset}_evaluation_nsga3_") and f"_seed_{seed}.json" in p.name for p in out_dir.glob(f"{dataset}_evaluation_nsga3_*_seed_{seed}.json"))
        if simple_path.exists() and lr_path.exists() and nsga3_exists:
            print(f"Already completed: dataset={dataset}, seed={seed}")
            exit(0)
    else:
        # For other seeds, only check NSGA3
        nsga3_exists = any(p.name.startswith(f"{dataset}_evaluation_nsga3_") and f"_seed_{seed}.json" in p.name for p in out_dir.glob(f"{dataset}_evaluation_nsga3_*_seed_{seed}.json"))
        if nsga3_exists:
            print(f"Already completed: dataset={dataset}, seed={seed}")
            exit(0)

    # Set the random seed for reproducibility
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    calculator = AlternativeMethods(path_preds, path_gt, dataset)
    # Only run simple average and logistic regression for seed 0
    if seed == 0:
        for method in [calculator.evaluation_simple_average, calculator.evaluation_logistic_regression]:
            results = method(seed=seed)
            for result in results:
                json_dump = result.model_dump()
                out_path = Path(f"/home/s/Git/python/GIRAFFE/benchmark/jsons_alternative/{dataset}_{method.__name__}_{result.id}_seed_{seed}.json")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w') as jfile:
                    json.dump(json_dump, jfile)
    # Always run NSGA3 for every seed
    results = calculator.evaluation_nsga3(seed=seed)
    for result in results:
        json_dump = result.model_dump()
        out_path = Path(f"/home/s/Git/python/GIRAFFE/benchmark/jsons_alternative/{dataset}_evaluation_nsga3_{result.id}_seed_{seed}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as jfile:
            json.dump(json_dump, jfile)
