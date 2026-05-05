# Experiment Reproduction Scripts

Minimal script bundle for reproducing the experiments around the OKAPI paper.

This directory intentionally contains scripts and small configuration files only. It does not include model checkpoints, prediction tensors, datasets, Python environments, Git metadata, caches, notebooks, generated tables, or generated plots.

## Contents

### `medmnist/`

- `train_medmnist_lightning.py`: train MedMNIST base neural networks with the Lightning implementation.
- `run_medmnist_okapi.py`: run one MedMNIST OKAPI experiment for a dataset/config/seed using the `okapi` package.
- `run_medmnist_alternatives.py`: run one MedMNIST alternative-method experiment (Simple Average, Logistic Regression stacking, and NSGA-III).
- `alternative_methods_benchmark.py`: helper module used by `run_medmnist_alternatives.py`.

### `starcop/`

- `train_starcop_model.py`: train one STARCOP segmentation model from a Hydra config.
- `prepare_starcop_okapi_data.py`: prepare STARCOP prediction tensors for OKAPI.
- `run_starcop_okapi.py`: run STARCOP OKAPI over a seed range.
- `run_starcop_nsga3.py`: run the STARCOP NSGA-III baseline.
- `run_starcop_ensemble_baselines.py`: run STARCOP averaging/weighting baselines.

### `automl/`

- `run_medmnist_tpot.py`: run the MedMNIST TPOT comparison.
- `run_medmnist_streamline.py`: run the MedMNIST STREAMLINE comparison.
- `run_medmnist_rebate.py`: run the MedMNIST ReBATE comparison.

### `configs/`

- `starcop/config_*.yaml`: representative STARCOP training configs.

### Environment

- `pixi.toml`: the single reviewer-facing environment for all TEVC2026 reproduction scripts.

## Environment Setup

From this directory, create the reviewer environment with:

```bash
pixi install
```

Then run scripts through `pixi run`, for example:

```bash
pixi run help-medmnist-alternatives
pixi run python medmnist/run_medmnist_alternatives.py --dataset pneumoniamnist --seed 0
```

## Notes

- MedMNIST scripts expect prediction tensors under `data_technical_paper/models/{dataset}/{train,valid,test}/` and ground-truth tensors under `data_technical_paper/gt/{dataset}/{train,val,test}.pt` by default. Override these with `--preds-dir` and `--gt-dir`.
- The Pixi environment installs the local OKAPI package in editable mode. If running these scripts outside this repository, set `OKAPI_REPO=/path/to/OKAPI` before running MedMNIST OKAPI or alternative-method scripts.
- STREAMLINE is expected at `automl/venvs/streamline_repo`, matching `automl/run_medmnist_streamline.py`. Clone the STREAMLINE source there before running that specific comparison.
- STARCOP commands should be run in the STARCOP environment, typically with `pixi run python ...`.
- AutoML STARCOP scripts are not included here because the final paper-facing STARCOP AutoML results were not used unless rerun on the same final `okapi_data` tensors.

## Typical Order

1. Train MedMNIST base models with `medmnist/train_medmnist_lightning.py`.
2. Run MedMNIST OKAPI with `medmnist/run_medmnist_okapi.py`.
3. Run MedMNIST alternative methods with `medmnist/run_medmnist_alternatives.py` and the scripts under `automl/`.
4. Train STARCOP base models with `starcop/train_starcop_model.py`.
5. Prepare STARCOP OKAPI tensors with `starcop/prepare_starcop_okapi_data.py`.
6. Run STARCOP OKAPI and baselines with the remaining scripts under `starcop/`.
