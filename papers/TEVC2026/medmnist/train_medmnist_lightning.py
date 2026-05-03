import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset # Make sure Subset is imported
from torchvision import models, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger # Removed, using only WandbLogger
from pytorch_lightning.loggers import WandbLogger
import wandb # Import wandb directly for artifact logging
from medmnist import INFO
import medmnist
import numpy as np
from PIL import Image
import pickle
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import StratifiedKFold # Removed, using custom StratifiedSplitter
from torchmetrics import AUROC, AveragePrecision, Accuracy
import pandas as pd
import os
import logging

DATA_ROOT = '/home/s/.medmnist' if '/home/s' in os.getcwd() else './.medmnist'

class StratifiedSplitter:
    def __init__(self):
        pass

    def split(self, strat_array, n_splits):
        self.n_splits = n_splits
        self.strat_array = strat_array
        self.masks = {i: np.zeros(len(strat_array)).astype(bool) for i in range(n_splits)}
        self.index_array = np.arange(len(strat_array))
        skf = StratifiedKFold(n_splits=self.n_splits, random_state=42, shuffle=True)
        for i, (train_index, test_index) in enumerate(skf.split(self.strat_array, self.strat_array)):
            self.masks[i][train_index] = True

    def get_train(self, fold):
        return self.index_array[self.masks[fold]]

    def get_test(self, fold):
        return self.index_array[~self.masks[fold]]

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            loaded = pickle.load(file)
            return loaded

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)


class MedMNISTLightningDataModule(pl.LightningDataModule):
    def __init__(self, data_flag, batch_size=128, load_size=28, resize_to=None, download=False, as_rgb=False, val_fold=None, k_folds=5):
        super().__init__()
        self.data_flag = data_flag
        self.batch_size = batch_size
        self.load_size = load_size
        self.resize_to = resize_to
        self.download = download
        self.as_rgb = as_rgb
        self.val_fold = val_fold # Which fold to use for validation (1 to k_folds)
        self.k_folds = k_folds
        # Define path for saving/loading the splitter object - uses fixed seed 42 from StratifiedSplitter
        self.splitter_dir = f"./kfold_splitters/{self.data_flag}"
        self.splitter_path = os.path.join(self.splitter_dir, f"k{self.k_folds}_splitter.pkl")
        self.train_indices = None
        self.val_indices = None

        # Get dataset info
        self.info = INFO[data_flag]
        self.task = self.info['task']
        # Determine channels based on as_rgb flag
        self.n_channels = 3 if as_rgb else INFO[data_flag]['n_channels']
        self.n_classes = len(INFO[data_flag]['label'])

        # Get the appropriate dataset class
        self.DataClass = getattr(medmnist, self.info['python_class'])

        self._dsettrain_val_set_up = False
        self._shuffle_train = True

    def prepare_data(self):
        # Download the data if needed
        # Download the base 28x28 dataset if needed
        # Download the base dataset if needed
        if self.download:
            # Need train split for fitting splitter or training
            self.DataClass(split='train', download=True, size=self.load_size, as_rgb=self.as_rgb)
            # Need standard val split only if not doing K-Fold
            if self.val_fold is None:
                 self.DataClass(split='val', download=True, size=self.load_size, as_rgb=self.as_rgb)
            # Always need test split for final evaluation
            self.DataClass(split='test', download=True, size=self.load_size, as_rgb=self.as_rgb)

    def setup(self, stage=None):
        # Define transformations
        transform_list = []
        # Apply resize only if resize_to is specified and different from load_size
        if self.resize_to is not None and self.resize_to != self.load_size:
             # Use BICUBIC interpolation for resizing as it's often better for natural images
            transform_list.append(transforms.Resize((self.resize_to, self.resize_to), interpolation=transforms.InterpolationMode.BICUBIC))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.data_transform = transforms.Compose(transform_list)

        if stage == 'test':
        # Load base test dataset first
            print("Setting up test set")
            self.test_dataset = self.DataClass(split='test', transform=self.data_transform,
                                       download=self.download, as_rgb=self.as_rgb, size=self.load_size, root=DATA_ROOT)

        elif stage == 'fit' or stage == 'pre':
            if not self._dsettrain_val_set_up:
                print("Setting up train and val sets")
            # Attempt K-Fold setup if val_fold is specified
                kfold_setup_success = False
                if self.val_fold is not None and 0 <= self.val_fold < self.k_folds:
                    print("Setting up kfolds")
                    kfold_setup_success = self._setup_kfold_split()
                    if not kfold_setup_success:
                        print("Falling back to standard train/val split due to K-Fold setup error.")
                        self.val_fold = None # Ensure we use standard split on error

                # Create train/val datasets based on K-Fold indices or standard splits
                if kfold_setup_success:
                    # Load the full training dataset *once* to apply subsets
                    full_train_dataset_with_transform = self.DataClass(split='train', transform=self.data_transform,
                                                                        download=self.download, as_rgb=self.as_rgb, size=self.load_size, root=DATA_ROOT)
                    labels = full_train_dataset_with_transform.labels
                    self.train_dataset = Subset(full_train_dataset_with_transform, self.train_indices) # type: ignore
                    self.train_dataset.labels = labels[self.train_indices] # type: ignore
                    self.val_dataset = Subset(full_train_dataset_with_transform, self.val_indices) # type: ignore
                    self.val_dataset.labels = labels[self.val_indices] # type: ignore
                    print(f"Using K-Fold: {len(self.train_dataset)} train samples, {len(self.val_dataset)} val samples.")
                else: # Handles fallback or initial non-kfold run
                    print("Using standard train/val splits.")
                    self.train_dataset = self.DataClass(split='train', transform=self.data_transform,
                                                    download=self.download, as_rgb=self.as_rgb, size=self.load_size, root=DATA_ROOT)
                    self.val_dataset = self.DataClass(split='val', transform=self.data_transform,
                                                download=self.download, as_rgb=self.as_rgb, size=self.load_size, root=DATA_ROOT)
                self._dsettrain_val_set_up = True
            else:
                print("Setup for train and val was already performed, skipping...")
        elif stage == 'predict':
            self.train_dataset = self.DataClass(split='train', transform=self.data_transform,
                                            download=self.download, as_rgb=self.as_rgb, size=self.load_size, root=DATA_ROOT)
            self.val_dataset = self.DataClass(split='val', transform=self.data_transform,
                                        download=self.download, as_rgb=self.as_rgb, size=self.load_size, root=DATA_ROOT)
            self.test_dataset = self.DataClass(split='test', transform=self.data_transform,
                                download=self.download, as_rgb=self.as_rgb, size=self.load_size, root=DATA_ROOT)

            self._shuffle_train = False
        else:
            raise ValueError("undefined stage")
        print("Finished setup()")

    def train_dataloader(self):
        dl =  torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self._shuffle_train, num_workers=4
        )
        self._shuffle_train = True # always reset the variable so only explicit predict stage does set it to False
        return dl

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def _setup_kfold_split(self):
        """Loads or creates StratifiedSplitter and sets train/val indices for the specified fold."""
        print(f"Setting up K-Fold validation using fold {self.val_fold}/{self.k_folds}")
        os.makedirs(self.splitter_dir, exist_ok=True)

        try:
            if os.path.exists(self.splitter_path):
                print(f"Loading existing StratifiedSplitter from {self.splitter_path}")
                splitter = StratifiedSplitter.load(self.splitter_path)
                # Check if loaded splitter has the correct number of splits (masks)
                if len(splitter.masks) != self.k_folds:
                     print(f"Warning: Loaded splitter has {len(splitter.masks)} folds, expected {self.k_folds}. Regenerating.")
                     raise FileNotFoundError # Trigger regeneration
                print("Splitter loaded successfully.")
            else:
                print(f"Generating new StratifiedSplitter and saving to {self.splitter_path}")
                # Load the full training data *without* transforms just for splitting
                full_train_dataset_for_split = self.DataClass(split='train', download=self.download,
                                                              as_rgb=self.as_rgb, size=self.load_size, root=DATA_ROOT)
                targets = full_train_dataset_for_split.labels

                # Handle multi-label stratification by creating unique string representations
                if self.info['task'] == 'multi-label, binary-class':
                    print("Generating string representations for multi-label stratification.")
                    stratify_targets = np.array([''.join(map(str, row)) for row in targets])
                else:
                     # For single-label tasks, squeeze the targets array
                     stratify_targets = targets.squeeze()

                splitter = StratifiedSplitter()
                splitter.split(strat_array=stratify_targets, n_splits=self.k_folds)

                # Save the splitter object using its own save method
                splitter.save(self.splitter_path)
                print(f"Saved splitter object to {self.splitter_path}")

                # Log splitter as W&B artifact
                if wandb.run:
                    try:
                        artifact_name = f'{self.data_flag}_load{self.load_size}_k{self.k_folds}_splitter'
                        artifact = wandb.Artifact(artifact_name, type='splitter_object')
                        artifact.add_file(self.splitter_path)
                        wandb.log_artifact(artifact)
                        print("Logged StratifiedSplitter artifact to W&B.")
                    except Exception as e:
                        print(f"Warning: Could not log StratifiedSplitter artifact to W&B: {e}")

            # Get boolean masks and indices for the current fold (0-based index)

            self.train_indices = splitter.get_train(self.val_fold)
            self.val_indices = splitter.get_test(self.val_fold)
            return True # Indicate successful K-Fold setup

        except Exception as e:
            print(f"Error during K-Fold setup: {e}")
            return False # Indicate failure


class MedMNISTLightningModel(pl.LightningModule):
    # Added load_size to constructor for Evaluator
    def __init__(self, data_flag, model_name='resnet18', in_channels=3, num_classes=2, task='multi-class',
                learning_rate=0.001, milestones=[0.5, 0.75], weights = None, gamma=0.1, num_epochs=100, load_size=28):
        super().__init__()
        # Manually save relevant hyperparameters if needed, or rely on logger
        # Added load_size to saved hyperparameters
        self.save_hyperparameters('data_flag', 'model_name', 'in_channels', 'num_classes', 'task', 'learning_rate', 'milestones', 'weights', 'gamma', 'num_epochs', 'load_size')
        self.data_flag = data_flag
        self.model_name = model_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.task = task
        self.learning_rate = learning_rate
        self.milestones = milestones
        self.load_size = load_size # Store load_size for Evaluator
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.weights = weights
        # self.size = size # Size is now implicitly handled by datamodule's resize logic

        # Model initialization
        self._build_model()

        # Loss function based on task
        if task == 'multi-label, binary-class':
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif task == 'binary-class':
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

        self._init_metrics(task)

    def _init_metrics(self, task):
        # --- TorchMetrics Initialization ---
        tm_task_type = None
        tm_num_outputs = None
        if task == 'binary-class':
            tm_task_type = 'binary'
            # For binary, AUROC/AP expect num_classes=None or 1, but we have 2 outputs from model.
            # We'll likely use the positive class prediction. Let's keep num_classes=None for now.
            tm_num_outputs = None  # Torchmetrics handles binary implicitly
        elif task == 'multi-class':
            tm_task_type = 'multiclass'
            tm_num_outputs = self.num_classes
        elif task == 'multi-label, binary-class':
            tm_task_type = 'multilabel'
            tm_num_outputs = self.num_classes  # num_labels for multilabel
        elif task == 'ordinal-regression':
            tm_task_type = 'multiclass'
            tm_num_outputs = self.num_classes
        else:
            raise ValueError(f"Unsupported task type for torchmetrics: {task}")
        if tm_task_type:
            common_args = {"task": tm_task_type}
            if tm_num_outputs is not None:
                # Add num_classes/num_labels only if needed (multiclass/multilabel)
                if tm_task_type in ['multiclass', 'multilabel']:
                    param_name = 'num_labels' if tm_task_type == 'multilabel' else 'num_classes'
                    common_args[param_name] = tm_num_outputs

            self.val_auroc = AUROC(**common_args)
            self.val_ap = AveragePrecision(**common_args)
            self.val_acc = Accuracy(**common_args)
            self.test_auroc = AUROC(**common_args)
            self.test_ap = AveragePrecision(**common_args)
            self.test_acc = Accuracy(**common_args)
            # Add training metrics
            self.train_auroc = AUROC(**common_args)
            self.train_ap = AveragePrecision(**common_args)
            self.train_acc = Accuracy(**common_args)
        # --- End TorchMetrics Initialization ---

    def _build_model(self):
        if self.model_name == 'resnet18':
            # Use weights=None instead of pretrained=False (updated param)
            self.model = models.resnet18(weights=None) # TODO: that actually should use pretrained weights as the original implementation uses them at 224 and not uses them at 28x28. However we can probably skip that.
            if self.in_channels == 1:
                # Modify the first layer to accept 1 channel
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # Modify the last layer for the number of classes
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        elif self.model_name == 'resnet50':
            # Use weights=None instead of pretrained=False (updated param)
            self.model = models.resnet50(weights=None)
            if self.in_channels == 1:
                # Modify the first layer to accept 1 channel
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # Modify the last layer for the number of classes
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        elif self.model_name == "efficientnet_S":
            self.model = models.efficientnet_v2_s(weights=None)
            if self.in_channels == 1:
                # Modify the first layer to accept 1 channel
                self.model.conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

            # Modify the last layer for the number of classes
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, self.num_classes) # type: ignore
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented")

    def weigh_loss(self, loss):
        if self.task == 'multi-label, binary-class' or self.task == 'ordinal-regression':
            pass
        if self.task == 'multi-class':
            pass
        if self.task == 'binary-class':
            pass

        return loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        if self.task == 'multi-label, binary-class' or self.task == 'binary-class':
            targets = targets.to(torch.float32)
            loss = self.criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long()
            loss = self.criterion(outputs, targets)

        if self.weights is not None:
            self.weigh_loss(loss)

        loss = loss.mean()

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)

        # Update Training TorchMetrics
        # Ensure targets are integers for binary/multiclass tasks
        if self.task in ['binary-class', 'multi-class', 'multi-label, binary-class']:
            targets_for_tm = targets.int()
        else: # multi-label expects float/long
            targets_for_tm = targets


        self.train_auroc(outputs.detach(), targets_for_tm)
        self.train_ap(outputs.detach(), targets_for_tm)
        self.train_acc(outputs.detach(), targets_for_tm)

        self.log("train_auroc", self.train_auroc, on_epoch=True, on_step=False)
        self.log("train_ap", self.train_ap, on_epoch=True, on_step=False)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        if self.task == 'multi-label, binary-class' or self.task == 'binary-class':
            targets = targets.to(torch.float32)
            loss = self.criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long()
            loss = self.criterion(outputs, targets)

        if self.weights is not None:
            self.weigh_loss(loss)

        loss = loss.mean()

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        # Update TorchMetrics
        # Ensure targets are integers for binary/multiclass tasks
        if self.task in ['binary-class', 'multi-class', 'multi-label, binary-class']:
            targets_for_tm = targets.int()
        else: # multi-label expects float/long
            targets_for_tm = targets

        self.val_auroc(outputs.detach(), targets_for_tm)
        self.val_ap(outputs.detach(), targets_for_tm)
        self.val_acc(outputs.detach(), targets_for_tm)

        self.log("val_auroc", self.val_auroc, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_ap", self.val_ap, on_epoch=True, on_step=False)
        self.log("val_acc", self.val_acc, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        if self.task == 'multi-label, binary-class' or self.task == 'binary-class':
            targets = targets.to(torch.float32)
            loss = self.criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long()
            loss = self.criterion(outputs, targets)

        if self.weights is not None:
            loss = self.weigh_loss(loss)

        loss = loss.mean()

        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        # Update TorchMetrics
        # Ensure targets are integers for binary/multiclass tasks
        if self.task in ['binary-class', 'multi-class', 'multi-label, binary-class']:
            targets_for_tm = targets.int()
        else: # multi-label expects float/long
            targets_for_tm = targets

        self.test_auroc(outputs.detach(), targets_for_tm)
        self.test_ap(outputs.detach(), targets_for_tm)
        self.test_acc(outputs.detach(), targets_for_tm)

        self.log("test_auroc", self.test_auroc, on_epoch=True, on_step=False)
        self.log("test_ap", self.test_ap, on_epoch=True, on_step=False)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

        return loss

    def predict_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        return outputs, targets

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(m * self.num_epochs) for m in self.milestones],
            gamma=self.gamma
        )
        return [optimizer], [scheduler]


    def on_load_checkpoint(self, checkpoint):
        print("Loading checkpoint")
        try:
            return super().on_load_checkpoint(checkpoint)
        except Exception as ex:
            print(f"{ex=}")

def get_weights(task, labels):
    if task == 'multi-label, binary-class':
        weights = 1 / labels.mean(axis=0)
        weights = weights / weights.min()
        weights = weights.reshape(1,-1)
    elif task == 'binary-class':
        weights = torch.tensor([1., 1 / labels.mean()]).reshape(2,1)
    elif task == 'multi-class' or task == 'ordinal-regression':
        weights = pd.Series(labels.squeeze()).value_counts()
        weights = weights / weights.min()
        weights = weights.sort_index()
        weights = weights.values
    else:
        raise ValueError('Task must be multi-label, binary-class, multi-class, or ordinal-regression')

    return torch.tensor(weights).float()


def main():
    parser = argparse.ArgumentParser(description='MedMNIST PyTorch Lightning Implementation')
    parser.add_argument('--data_flag', default='pathmnist', type=str, help='dataset flag')
    parser.add_argument('--output_root', default='./output', type=str, help='output root')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--gpu', action='store_true', help='use GPU')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--load_size', default=28, type=int, help='Size of the dataset images to load (e.g., 28, 64, 128, 224)')
    parser.add_argument('--resize_to', default=None, type=int, help='Target size to resize loaded images to (e.g., 224)')
    parser.add_argument('--model_name', default='resnet18', type=str, help='model name: resnet18 or resnet50')
    parser.add_argument('--download', action='store_true', help='download the dataset')
    parser.add_argument('--as_rgb', action='store_true', help='convert data to RGB')
    parser.add_argument('--weigh_loss', action='store_true', help='Whether to weigh loss based on class imbalance')
    # parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging') # Always use W&B now
    parser.add_argument('--val_fold', default=None, type=int,
                        help='Validation fold number (0-k_folds - 1) for K-Fold CV. If None, use standard val split.')
    parser.add_argument('--k_folds', default=5, type=int, help='Number of folds for K-Fold CV.')
    parser.add_argument("--monitor", default='val_auroc', help="What metric is used for early stopping")
    parser.add_argument("--monitor_objective", default='max', help="Whether the monitored objective should be minimized (min) or maximized (max). Loss generally should be minimized, most of other metrics should be maximized")
    # random_state is fixed in StratifiedSplitter, removing argument
    # parser.add_argument('--random_state', default=42, type=int, help='Random state for K-Fold shuffle (Note: StratifiedSplitter uses fixed 42).')

    args = parser.parse_args()

    # Setup data module
    logging.info("Loading_datamodule")
    data_module = MedMNISTLightningDataModule(
        data_flag=args.data_flag,
        batch_size=args.batch_size,
        load_size=args.load_size,
        resize_to=args.resize_to,
        download=args.download,
        as_rgb=args.as_rgb,
        val_fold=args.val_fold,
        k_folds=args.k_folds
        # random_state=args.random_state # Removed
    )

    # Setup model
    data_module.prepare_data()
    data_module.setup(stage='pre')


    # Get task and class info
    info = INFO[args.data_flag]
    task = info['task']
    n_channels = 3 if args.as_rgb else info['n_channels']
    if task != 'binary-class':
        n_classes = len(info['label'])
    else:
        n_classes = 1



    train_dataset = data_module.train_dataset
    train_labels = train_dataset.labels
    weights = get_weights(task, train_labels) if args.weigh_loss else None
    model = MedMNISTLightningModel(
        data_flag=args.data_flag,
        model_name=args.model_name,
        in_channels=n_channels,
        num_classes=n_classes,
        task=task,
        num_epochs=args.num_epochs,
        load_size=args.load_size, # Pass load_size to the model
        weights=weights,
    )


    logger = WandbLogger(project='medmnist-repro', log_model=True) # Only log the best checkpoint as artifact


    # Log hyperparameters (logger handles model hparams automatically)
    # Log relevant args and calculated values
    hparams_to_log = vars(args)
    hparams_to_log['n_channels'] = n_channels
    hparams_to_log['n_classes'] = n_classes
    hparams_to_log['task'] = task
    hparams_to_log['actual_val_fold'] = data_module.val_fold # Log the fold actually used
    logger.log_hyperparams(hparams_to_log)

    # Define output dir based on run name for local checkpoints (optional with W&B artifact logging)
    output_dir = os.path.join(args.output_root, logger.experiment.name)
    os.makedirs(output_dir, exist_ok=True)

    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='model',
            monitor=args.monitor,
            mode=args.monitor_objective,
            save_top_k=1
        ),
        EarlyStopping(
            monitor=args.monitor,
            min_delta=0.001, # Stop if improvement is less than 0.001
            patience=10,      # Stop after 10 epochs with no significant improvement
            mode=args.monitor_objective,
            verbose=True     # Print message when stopping
        )
    ]

    # Logger is already defined as WandbLogger above

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu' if args.gpu else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1
    )

    # Train and test
    trainer.fit(model, data_module)
    # Test the best model using the standard test set
    print("\nTesting the best model on the standard test set...")
    # The test method automatically loads the best checkpoint based on monitor/mode if ckpt_path='best'
    trainer.test(datamodule=data_module, ckpt_path="best")

    if os.getcwd() == '/app':
        print(f'{os.getcwd()=}')
        print(f"{logger.save_dir=}")
        attempted_path = f'/app/lightning_output/{logger.experiment.name}/checkpoints'
        print(f"Attempted path: {attempted_path}/model.ckpt")
        print(f"listdir: {os.listdir(attempted_path)}")
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(f"{attempted_path}/model.ckpt")
        wandb.log_artifact(artifact)
        print("WAITING FOR ARTIFACT...")
        artifact.wait()

    wandb.finish() # Ensure W&B run finishes


if __name__ == '__main__':
    main()
