#!/usr/bin/env python3
"""
Unified STARCOP training script

Supports both methodologies through config files:
- Original: Single train/test split (config.yaml)
- HyperSTARCOP: Proper train/val/test separation (config_hyperstarcop.yaml)

Key features:
- Flexible train/val/test dataset configuration
- Best checkpoint loading for evaluation
- Comprehensive validation on all configured datasets
- W&B artifact upload
- Configurable through YAML configs only
"""

import matplotlib
matplotlib.use('agg')

import os
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb

# Import STARCOP modules
from starcop.dataset_setup import get_dataset
from starcop.model_setup import get_model
from starcop.data.data_logger import ImageLogger
from starcop.validation import run_validation


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(settings: DictConfig) -> None:
    """
    Unified training function that supports multiple methodologies.

    Config requirements:
    - For original methodology: Only dataset.train_csv required
    - For HyperSTARCOP: dataset.train_csv, dataset.val_csv, dataset.test_csv required
    """

    # Setup logging
    log = logging.getLogger(__name__)

    experiment_path = os.getcwd()
    checkpoint_path = os.path.join(experiment_path, "checkpoints").replace("\\", "/")
    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    log.info("=" * 80)
    log.info("STARCOP TRAINING")
    log.info("=" * 80)
    log.info(f"Experiment path: {experiment_path}")
    log.info(f"Checkpoints path: {checkpoint_path}")

    OmegaConf.set_struct(settings, False)
    settings["experiment_path"] = experiment_path
    OmegaConf.set_struct(settings, True)

    # Check methodology based on config
    has_val_csv = "val_csv" in settings.dataset and settings.dataset.val_csv is not None
    has_test_csv = "test_csv" in settings.dataset and settings.dataset.test_csv is not None
    use_proper_split = has_val_csv and has_test_csv

    if use_proper_split:
        log.info("Using HyperSTARCOP methodology (train/val/test separation)")
        log.info(f"  Train CSV: {settings.dataset.train_csv}")
        log.info(f"  Val CSV: {settings.dataset.val_csv}")
        log.info(f"  Test CSV: {settings.dataset.test_csv}")
    else:
        log.info("Using original methodology (train/test only)")
        log.info(f"  Train CSV: {settings.dataset.train_csv}")

    log.info(f"\nConfiguration:\n{OmegaConf.to_yaml(settings)}")

    # SEED
    seed_everything((None if settings.seed == "None" else settings.seed))

    # LOGGING SETUP
    log.info("\n" + "=" * 80)
    log.info("WANDB LOGGER SETUP")
    log.info("=" * 80)

    wandb_logger = WandbLogger(
        name=settings.experiment_name,
        project=settings.wandb.wandb_project,
        entity=settings.wandb.wandb_entity,
        log_model=False,  # Just do the manual upload
    )
    wandb_logger.experiment.config.update(OmegaConf.to_container(settings, resolve=True))

    log.info(f"W&B Run: {wandb_logger.experiment.name}")
    log.info(f"W&B ID: {wandb_logger.experiment.id}")

    # DATASET SETUP
    log.info("\n" + "=" * 80)
    log.info("DATASET SETUP")
    log.info("=" * 80)

    data_module = get_dataset(settings)
    data_module.prepare_data()

    log.info(f"Train samples (tiled patches): {len(data_module.train_dataset)}")
    log.info(f"Train samples (full images): {len(data_module.train_dataset_non_tiled)}")

    if use_proper_split:
        log.info(f"Val samples: {len(data_module.val_dataset)}")
        log.info(f"Test samples: {len(data_module.test_dataset)}")
    else:
        log.info(f"Test samples: {len(data_module.test_dataset)}")

    # MODEL SETUP
    log.info("\n" + "=" * 80)
    log.info("MODEL SETUP")
    log.info("=" * 80)

    model = get_model(settings, settings.experiment_name)
    log.info(f"Model type: {settings.model.model_type}")
    log.info(f"Backbone: {settings.model.semseg_backbone}")
    log.info(f"Loss: {settings.model.loss}")
    log.info(f"Learning rate: {settings.model.lr}")

    # Log loss-specific parameters
    if settings.model.loss == 'BCEWithLogitsLoss':
        if hasattr(settings.model, 'pos_weight'):
            log.info(f"  Loss param - pos_weight: {settings.model.pos_weight}")
    elif settings.model.loss == 'dice':
        if hasattr(settings.model, 'dice_smooth'):
            log.info(f"  Loss param - dice_smooth: {settings.model.dice_smooth}")
    elif settings.model.loss == 'tversky':
        if hasattr(settings.model, 'tversky_alpha'):
            log.info(f"  Loss param - tversky_alpha: {settings.model.tversky_alpha}")
        if hasattr(settings.model, 'tversky_beta'):
            log.info(f"  Loss param - tversky_beta: {settings.model.tversky_beta}")
        if hasattr(settings.model, 'tversky_smooth'):
            log.info(f"  Loss param - tversky_smooth: {settings.model.tversky_smooth}")
    elif settings.model.loss == 'focal':
        if hasattr(settings.model, 'focal_alpha'):
            log.info(f"  Loss param - focal_alpha: {settings.model.focal_alpha}")
        if hasattr(settings.model, 'focal_gamma'):
            log.info(f"  Loss param - focal_gamma: {settings.model.focal_gamma}")
    elif settings.model.loss == 'bce_dice':
        if hasattr(settings.model, 'bce_dice_bce_weight'):
            log.info(f"  Loss param - bce_dice_bce_weight: {settings.model.bce_dice_bce_weight}")
        if hasattr(settings.model, 'bce_dice_dice_weight'):
            log.info(f"  Loss param - bce_dice_dice_weight: {settings.model.bce_dice_dice_weight}")
        if hasattr(settings.model, 'pos_weight'):
            log.info(f"  Loss param - pos_weight: {settings.model.pos_weight}")

    # CHECKPOINTING SETUP
    log.info("\n" + "=" * 80)
    log.info("CALLBACKS SETUP")
    log.info("=" * 80)

    # Model selection metric (configurable)
    # Options: val_loss, val_f1score, val_f05score, val_f2score, val_precision, val_recall
    selection_metric = getattr(settings.model, 'selection_metric', 'val_f1score')

    # Determine mode based on metric type
    # val_loss should be minimized, all other metrics should be maximized
    if selection_metric == "val_loss":
        selection_mode = "min"
    else:
        selection_mode = "max"

    log.info(f"Selection metric: {selection_metric} (mode: {selection_mode})")

    # Create filename with the selection metric
    checkpoint_filename = f'best-{{epoch:02d}}-{{{selection_metric}:.4f}}'

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=checkpoint_filename,
        save_top_k=1,
        verbose=True,
        monitor=selection_metric,
        mode=selection_mode
    )

    early_stop_callback = EarlyStopping(
        monitor=selection_metric,
        patience=settings.model.early_stopping_patience,
        strict=False,
        verbose=True,
        mode=selection_mode
    )

    # IMAGE LOGGER SETUP
    batch_train = next(iter(data_module.train_plot_dataloader(batch_size=settings.plot_samples)))

    if use_proper_split:
        # Use validation set for logging
        batch_val = next(iter(data_module.val_plot_dataloader(batch_size=settings.plot_samples)))
    else:
        # Use test set for logging (original methodology)
        batch_val = next(iter(data_module.test_plot_dataloader(batch_size=settings.plot_samples)))

    il = ImageLogger(
        batch_train=batch_train,
        batch_test=batch_val,
        products_plot=settings.products_plot,
        input_products=settings.dataset.input_products
    )

    callbacks = [checkpoint_callback, early_stop_callback, il]
    log.info(f"Early stopping patience: {settings.model.early_stopping_patience}")

    # TRAINING SETUP
    log.info("\n" + "=" * 80)
    log.info("TRAINING START")
    log.info("=" * 80)

    trainer = Trainer(
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=experiment_path,
        accelerator=settings.training.accelerator,
        devices=settings.training.devices,
        max_epochs=settings.training.max_epochs,
        val_check_interval=settings.training.val_check_interval,
        log_every_n_steps=settings.training.train_log_every_n_steps,
    )

    trainer.fit(model, data_module)

    # SAVE LAST CHECKPOINT
    log.info("\n" + "=" * 80)
    log.info("CHECKPOINT MANAGEMENT")
    log.info("=" * 80)

    final_checkpoint_last = os.path.join(experiment_path, "final_checkpoint_last.ckpt")
    trainer.save_checkpoint(final_checkpoint_last)
    log.info(f"Saved last checkpoint: {final_checkpoint_last}")

    # LOAD BEST CHECKPOINT FOR EVALUATION
    best_checkpoint_path = checkpoint_callback.best_model_path
    log.info(f"\nLoading BEST checkpoint for evaluation: {best_checkpoint_path}")
    log.info(f"Best {selection_metric}: {checkpoint_callback.best_model_score:.4f}")

    from starcop.models.model_module import ModelModule
    model = ModelModule.load_from_checkpoint(best_checkpoint_path)
    model.eval()

    # Copy best checkpoint with clear name for local reference
    import shutil
    best_checkpoint_final = os.path.join(experiment_path, "final_checkpoint_best.ckpt")
    shutil.copy2(best_checkpoint_path, best_checkpoint_final)
    log.info(f"Copied best checkpoint to: {best_checkpoint_final}")

    # Upload ORIGINAL best checkpoint as W&B artifact (not the copy!)
    # CRITICAL: Must upload the actual best checkpoint from checkpoints/ folder
    # NOT the copy, as trainer.save_checkpoint() creates a different file!
    log.info("\nUploading best checkpoint to W&B...")
    artifact = wandb.Artifact(
        name=f"model-{wandb_logger.experiment.name}",
        type="model",
        description=f"Best model checkpoint ({selection_metric}={checkpoint_callback.best_model_score:.4f})"
    )
    artifact.add_file(best_checkpoint_path)  # Upload ORIGINAL, not copy!
    wandb.log_artifact(artifact)
    log.info(f"Best checkpoint uploaded to W&B: {best_checkpoint_path}")

    if_run_validation = True # for debugging purposes - will be removed at some point but useful for quick debug runs.

    if if_run_validation:
        # ========================================================================
        # VALIDATION ON ALL CONFIGURED DATASETS
        # ========================================================================

        log.info("\n" + "=" * 80)
        log.info("VALIDATION (using BEST checkpoint)")
        log.info("=" * 80)

        results_dir = Path(experiment_path) / "validation_results"
        results_dir.mkdir(exist_ok=True)

        # Helper function for safe metric formatting
        def safe_format(metrics_dict, key):
            val = metrics_dict.get(key)
            return f"{val:.4f}" if val is not None else "N/A"

        def safe_format_table(val):
            return f"{val:<10.4f}" if val is not None else f"{'N/A':<10}"

        # Store results for summary table
        validation_results = {}

        # 1. VALIDATION ON TRAINING SET (always run)
        log.info("\n1. Validation on TRAINING set (full images)...")
        train_results_dir = results_dir / "train"
        train_results_dir.mkdir(exist_ok=True)

        dataloader_train = data_module.train_non_tiled_dataloader(
            batch_size=1,
            num_workers=data_module.num_workers
        )

        results_train, metrics_train = run_validation(
            model,
            dataloader_train,
            products_plot=[],  # No plots for speed
            verbose=False,
            show_plots=False,
            path_save_results=str(train_results_dir),
            skip_saving_plots=True
        )

        validation_results['train'] = {
            'samples': len(results_train),
            'metrics': metrics_train
        }

        log.info(f"  Samples: {len(results_train)}")
        log.info(f"  F1_easy: {safe_format(metrics_train, 'f1score_easy')}")
        log.info(f"  F1_hard: {safe_format(metrics_train, 'f1score_hard')}")
        log.info(f"  Overall F1: {safe_format(metrics_train, 'f1score')}")

        # 2. VALIDATION SET (only if using proper split)
        if use_proper_split:
            log.info("\n2. Validation on VALIDATION set...")
            val_results_dir = results_dir / "val"
            val_results_dir.mkdir(exist_ok=True)

            dataloader_val = data_module.val_plot_dataloader(
                batch_size=1,
                num_workers=data_module.num_workers
            )

            results_val, metrics_val = run_validation(
                model,
                dataloader_val,
                products_plot=settings.products_plot,
                verbose=False,
                show_plots=False,
                path_save_results=str(val_results_dir),
                skip_saving_plots=False
            )

            validation_results['val'] = {
                'samples': len(results_val),
                'metrics': metrics_val
            }

            log.info(f"  Samples: {len(results_val)}")
            log.info(f"  F1_easy: {safe_format(metrics_val, 'f1score_easy')}")
            log.info(f"  F1_hard: {safe_format(metrics_val, 'f1score_hard')}")
            log.info(f"  Overall F1: {safe_format(metrics_val, 'f1score')}")

        # 3. TEST SET (always run)
        log.info("\n3. Validation on TEST set...")
        test_results_dir = results_dir / "test"
        test_results_dir.mkdir(exist_ok=True)

        dataloader_test = data_module.test_plot_dataloader(
            batch_size=1,
            num_workers=data_module.num_workers
        )

        results_test, metrics_test = run_validation(
            model,
            dataloader_test,
            products_plot=settings.products_plot,
            verbose=False,
            show_plots=False,
            path_save_results=str(test_results_dir),
            skip_saving_plots=False
        )

        validation_results['test'] = {
            'samples': len(results_test),
            'metrics': metrics_test
        }

        log.info(f"  Samples: {len(results_test)}")
        log.info(f"  F1_easy: {safe_format(metrics_test, 'f1score_easy')}")
        log.info(f"  F1_hard: {safe_format(metrics_test, 'f1score_hard')}")
        log.info(f"  Overall F1: {safe_format(metrics_test, 'f1score')}")

        # SUMMARY TABLE
        log.info("\n" + "=" * 80)
        log.info("VALIDATION SUMMARY (BEST checkpoint)")
        log.info("=" * 80)
        log.info(f"{'Set':<15} {'Samples':<10} {'F1_easy':<10} {'F1_hard':<10} {'F1_overall':<10}")
        log.info("-" * 80)

        for set_name in validation_results:
            result = validation_results[set_name]
            metrics = result['metrics']
            log.info(
                f"{set_name.capitalize():<15} "
                f"{result['samples']:<10} "
                f"{safe_format_table(metrics.get('f1score_easy'))} "
                f"{safe_format_table(metrics.get('f1score_hard'))} "
                f"{safe_format_table(metrics.get('f1score'))}"
            )

        log.info("=" * 80)

        # FINISH
        log.info(f"\nResults saved to: {results_dir}")
        log.info(f"Best checkpoint: {best_checkpoint_final}")

    wandb.finish()

    log.info("\n" + "=" * 80)
    log.info("TRAINING COMPLETE!")
    log.info("=" * 80)


if __name__ == "__main__":
    train()
