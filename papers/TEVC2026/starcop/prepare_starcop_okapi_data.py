#!/usr/bin/env python3
"""
Prepare predictions for OKAPI GP fusion.

Reorganizes predictions from per-model directories into OKAPI-friendly structure:

Input structure (predictions/):
    predictions/
    ├── 01_focal_loss/
    │   ├── val_predictions.pt
    │   ├── val_ground_truths.pt
    │   ├── test_predictions.pt
    │   └── test_ground_truths.pt
    └── ...

Output structure (okapi_data/):
    okapi_data/
    ├── gt/
    │   ├── y_val.pt      # Ground truth for validation set
    │   └── y_test.pt     # Ground truth for test set
    ├── val/
    │   ├── focal_loss.pt
    │   ├── posweight_05.pt
    │   └── ...
    └── test/
        ├── focal_loss.pt
        ├── posweight_05.pt
        └── ...

Usage:
    pixi run python scripts/prepare_okapi_data.py
    pixi run python scripts/prepare_okapi_data.py --predictions_dir predictions --output_dir okapi_data
"""

import argparse
import shutil
from pathlib import Path
import torch


def get_model_name(dir_name: str) -> str:
    """Extract clean model name from directory name (remove numeric prefix)."""
    # e.g., "01_focal_loss" -> "focal_loss"
    parts = dir_name.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit():
        return parts[1]
    return dir_name


def main():
    parser = argparse.ArgumentParser(description="Prepare predictions for OKAPI")
    parser.add_argument("--predictions_dir", type=str, default="predictions",
                        help="Directory containing model predictions")
    parser.add_argument("--output_dir", type=str, default="okapi_data",
                        help="Output directory for OKAPI-formatted data")
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Preparing data for OKAPI GP fusion")
    print("=" * 60)
    print(f"Input:  {predictions_dir}")
    print(f"Output: {output_dir}")
    print()

    # Create output directory structure
    gt_dir = output_dir / "gt"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for d in [gt_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Find all model directories
    model_dirs = sorted([d for d in predictions_dir.iterdir() if d.is_dir()])
    print(f"Found {len(model_dirs)} models")
    print()

    # Copy ground truth from first model (should be identical for all)
    gt_copied = False
    for model_dir in model_dirs:
        val_gt = model_dir / "val_ground_truths.pt"
        test_gt = model_dir / "test_ground_truths.pt"

        if val_gt.exists() and test_gt.exists():
            print("Copying ground truth...")
            shutil.copy(val_gt, gt_dir / "y_val.pt")
            shutil.copy(test_gt, gt_dir / "y_test.pt")
            print(f"  y_val.pt  <- {val_gt}")
            print(f"  y_test.pt <- {test_gt}")
            gt_copied = True
            break

    if not gt_copied:
        print("ERROR: No ground truth files found!")
        return

    print()
    print("Copying predictions...")
    print("-" * 60)

    # Copy predictions from each model
    for model_dir in model_dirs:
        model_name = get_model_name(model_dir.name)

        val_pred = model_dir / "val_predictions.pt"
        test_pred = model_dir / "test_predictions.pt"

        if val_pred.exists():
            shutil.copy(val_pred, val_dir / f"{model_name}.pt")
            print(f"  val/{model_name}.pt")

        if test_pred.exists():
            shutil.copy(test_pred, test_dir / f"{model_name}.pt")
            print(f"  test/{model_name}.pt")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    # Verify and summarize
    val_files = list(val_dir.glob("*.pt"))
    test_files = list(test_dir.glob("*.pt"))

    print(f"Ground truth: {gt_dir}/")
    y_val = torch.load(gt_dir / "y_val.pt", weights_only=True)
    y_test = torch.load(gt_dir / "y_test.pt", weights_only=True)
    print(f"  y_val.pt:  {y_val.shape}")
    print(f"  y_test.pt: {y_test.shape}")

    print(f"\nValidation predictions: {val_dir}/")
    print(f"  {len(val_files)} models")
    for f in sorted(val_files):
        print(f"    - {f.name}")

    print(f"\nTest predictions: {test_dir}/")
    print(f"  {len(test_files)} models")
    for f in sorted(test_files):
        print(f"    - {f.name}")

    print()
    print("=" * 60)
    print(f"Done! Data ready for OKAPI at: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
