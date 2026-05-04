# Experiment Reproduction Scripts

Minimal script bundle for reproducing the experiments around the OKAPI paper.

This directory intentionally contains scripts and small configuration files only. It does not include model checkpoints, prediction tensors, datasets, Python environments, Git metadata, caches, notebooks, generated tables, or generated plots.

## Contents

### `medmnist/`

- `train_medmnist_lightning.py`: train MedMNIST base neural networks with the Lightning implementation.
- `run_medmnist_okapi.py`: run one MedMNIST OKAPI experiment for a dataset/config/seed.
- `run_medmnist_alternatives.py`: run one MedMNIST alternative-method experiment.

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

- `medmnist_training_pyproject.toml` and `medmnist_training_requirements.txt`: dependency references for MedMNIST model training.
- `starcop/pixi.toml`: STARCOP Pixi environment reference.
- `starcop/config_*.yaml`: representative STARCOP training configs.

## Typical Order

1. Train MedMNIST base models with `medmnist/train_medmnist_lightning.py`.
2. Run MedMNIST OKAPI with `medmnist/run_medmnist_okapi.py`.
3. Run MedMNIST alternative methods with `medmnist/run_medmnist_alternatives.py` and the scripts under `automl/`.
4. Train STARCOP base models with `starcop/train_starcop_model.py`.
5. Prepare STARCOP OKAPI tensors with `starcop/prepare_starcop_okapi_data.py`.
6. Run STARCOP OKAPI and baselines with the remaining scripts under `starcop/`.
