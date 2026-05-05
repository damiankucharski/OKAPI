import argparse
import dataclasses
import json
from pathlib import Path
import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run alternative methods benchmark for a single dataset and seed.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for NSGA3")
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=Path("data_technical_paper/models"),
        help="Directory containing model predictions: {dataset}/{train,valid,test}/*.pt",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("data_technical_paper/gt"),
        help="Directory containing ground-truth tensors: {dataset}/{train,val,test}.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/medmnist_alternatives"),
        help="Directory for JSON results and NSGA-III weights",
    )
    args = parser.parse_args()

    from alternative_methods_benchmark import AlternativeMethods

    dataset = args.dataset
    seed = args.seed
    out_dir = args.output_dir
    weights_dir = out_dir / "nsga3_weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{start_time}] Running alternative methods for dataset={dataset}, seed={seed}")

    simple_path = out_dir / f"{dataset}_evaluation_simple_average_0_seed_{seed}.json"
    lr_path = out_dir / f"{dataset}_evaluation_logistic_regression_0_seed_{seed}.json"
    if seed == 0:
        nsga3_exists = any(p.name.startswith(f"{dataset}_evaluation_nsga3_") and f"_seed_{seed}.json" in p.name for p in out_dir.glob(f"{dataset}_evaluation_nsga3_*_seed_{seed}.json"))
        if simple_path.exists() and lr_path.exists() and nsga3_exists:
            print(f"Already completed: dataset={dataset}, seed={seed}")
            exit(0)
    else:
        nsga3_exists = any(p.name.startswith(f"{dataset}_evaluation_nsga3_") and f"_seed_{seed}.json" in p.name for p in out_dir.glob(f"{dataset}_evaluation_nsga3_*_seed_{seed}.json"))
        if nsga3_exists:
            print(f"Already completed: dataset={dataset}, seed={seed}")
            exit(0)

    # Set the random seed for reproducibility
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    calculator = AlternativeMethods(args.preds_dir, args.gt_dir, dataset, weights_dir=weights_dir)
    # Only run simple average and logistic regression for seed 0
    if seed == 0:
        for method in [calculator.evaluation_simple_average, calculator.evaluation_logistic_regression]:
            results = method(seed=seed)
            for result in results:
                json_dump = dataclasses.asdict(result)
                out_path = out_dir / f"{dataset}_{method.__name__}_{result.id}_seed_{seed}.json"
                with open(out_path, 'w') as jfile:
                    json.dump(json_dump, jfile)
    # Always run NSGA3 for every seed
    results = calculator.evaluation_nsga3(seed=seed)
    for result in results:
        json_dump = dataclasses.asdict(result)
        out_path = out_dir / f"{dataset}_evaluation_nsga3_{result.id}_seed_{seed}.json"
        with open(out_path, 'w') as jfile:
            json.dump(json_dump, jfile)
