"""Simple integration tests for checkpoint functionality.

Tests that checkpoint symlinks and files are properly managed based on
logged metrics during training.
"""

from __future__ import annotations

from pathlib import Path

import nshconfig as C
import torch
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

import nshtrainer as nt
from nshtrainer._checkpoint.metadata import CheckpointMetadata
from nshtrainer.trainer._config import TrainerConfig
from nshtrainer.trainer.trainer import Trainer as NSHTrainer


class SimpleDataset(Dataset):
    """Simple dataset that returns known indices."""

    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    @override
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx])


class SimpleModuleConfig(C.Config):
    """Simple configuration for test module."""

    # Sequence of metric values to use during validation
    metric_values: list[float] = [5.0, 4.0, 3.0, 2.0, 1.0]
    # Name of the metric to log
    metric_name: str = "loss"


class SimpleModule(nt.LightningModuleBase):
    """Simple module that logs predefined metric values."""

    @override
    @classmethod
    def hparams_cls(cls):
        return SimpleModuleConfig

    def __init__(self, hparams):
        super().__init__(hparams)
        self.layer = torch.nn.Linear(1, 1)

    @override
    def forward(self, x: torch.Tensor):
        return self.layer(x.float())

    @override
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # Simple training step that isn't important for the test
        preds = self(batch)
        loss = torch.mean(preds)
        return loss

    @override
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # Log predefined metric values
        if self.current_epoch < len(self.hparams.metric_values):
            metric_value = self.hparams.metric_values[self.current_epoch]
            # Log the metric - this is what will be used for checkpointing
            self.log(self.hparams.metric_name, metric_value)

    @override
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _verify_last_checkpoints(checkpoint_dir: Path, epoch: int, step: int):
    # Check for last checkpoint
    dir = checkpoint_dir / "last"
    symlink = checkpoint_dir / "last.ckpt"

    # Verify directories and symlinks exist
    assert dir.exists(), f"Last checkpoint directory {dir} does not exist"
    assert symlink.exists(follow_symlinks=False), (
        f"Last checkpoint symlink {symlink} does not exist"
    )

    # Verify symlinks point to real files
    target = symlink.resolve()
    assert target.exists(), f"Last symlink target {target} does not exist"

    # Find the last checkpoint file in the directory
    assert (checkpoint_file := next(dir.glob("*.ckpt"), None)) is not None, (
        f"Last checkpoint file not found in {dir}"
    )
    assert checkpoint_file.is_file(), (
        f"Last checkpoint file {checkpoint_file} is not a file"
    )
    assert checkpoint_file.absolute() == symlink.resolve().absolute(), (
        f"Last checkpoint symlink {symlink} does not point to the last checkpoint file {checkpoint_file}"
    )

    # Verify the last checkpoint file has the expected epoch and step
    metadata = CheckpointMetadata.from_ckpt_path(checkpoint_file)
    metadata_epoch = metadata.epoch
    metadata_step = metadata.global_step
    assert metadata_epoch == epoch, (
        f"Expected last checkpoint epoch {epoch}, but got {metadata_epoch}"
    )
    assert metadata_step == step, (
        f"Expected last checkpoint step {step}, but got {metadata_step}"
    )


def _verify_best_checkpoints(
    checkpoint_dir: Path,
    metric_name: str,
    metric_value: float,
):
    # Check for best checkpoint based on the metric
    dir = checkpoint_dir / f"best_{metric_name}"
    symlink = checkpoint_dir / f"best_{metric_name}.ckpt"

    # Verify directories and symlinks exist
    assert dir.exists(), f"Best checkpoint directory {dir} does not exist"
    assert symlink.exists(), f"Best checkpoint symlink {symlink} does not exist"

    # Verify symlinks point to real files
    target = symlink.resolve()
    assert target.exists(), f"Best symlink target {target} does not exist"

    # Find the best checkpoint file in the directory
    assert (checkpoint_file := next(dir.glob("*.ckpt"), None)) is not None, (
        f"Best checkpoint file not found in {dir}"
    )
    assert checkpoint_file.is_file(), (
        f"Best checkpoint file {checkpoint_file} is not a file"
    )
    assert checkpoint_file.absolute() == symlink.resolve().absolute(), (
        f"Best checkpoint symlink {symlink} does not point to the best checkpoint file {checkpoint_file}"
    )

    # Verify best checkpoint has the best metric value
    metadata = CheckpointMetadata.from_ckpt_path(checkpoint_file)
    assert abs(metadata.metrics[metric_name] - metric_value) < 1e-5, (
        f"Expected best metric value {metric_value}, but got {metadata.metrics[metric_name]} in {target.name}"
    )


def test_improving_metrics_checkpoint(tmp_path):
    """Test checkpoints with improving metrics using self.log()."""
    logs_dir = Path(tmp_path)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Model config with improving metric values (lower is better)
    model_config = SimpleModuleConfig(metric_values=[5.0, 4.0, 3.0, 2.0, 1.0])

    # Create trainer config with checkpoint callbacks
    # Set primary_metric to match our logged metric
    max_epochs = len(model_config.metric_values)
    trainer_config = TrainerConfig(
        barebones=False,
        loggers=[nt.configs.CSVLoggerConfig()],
        max_epochs=max_epochs,
        accelerator="cpu",
        primary_metric=nt.MetricConfig(monitor=model_config.metric_name, mode="min"),
    ).with_project_root(logs_dir)

    # Create trainer and model
    trainer = NSHTrainer(trainer_config)
    model = SimpleModule(model_config)

    # Create data loaders
    train_dl = DataLoader(
        SimpleDataset(10),
        batch_size=2,
        num_workers=0,
    )
    val_dl = DataLoader(
        SimpleDataset(10),
        batch_size=2,
        num_workers=0,
    )

    # Run training
    trainer.fit(model, train_dl, val_dl)

    # Verify checkpoints - best value should be 1.0 (the last and best value)
    ckpt_dir = trainer.hparams.directory.resolve_subdirectory(
        trainer.hparams.id, "checkpoint"
    )
    _verify_last_checkpoints(
        ckpt_dir,
        epoch=max_epochs - 1,
        step=max_epochs * len(train_dl),
    )
    _verify_best_checkpoints(
        ckpt_dir,
        metric_name=model_config.metric_name,
        metric_value=model_config.metric_values[-1],
    )


def test_worsening_metrics_checkpoint(tmp_path):
    """Test checkpoints with worsening metrics using self.log()."""
    logs_dir = Path(tmp_path)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Model config with worsening metric values (lower is better)
    model_config = SimpleModuleConfig(metric_values=[1.0, 2.0, 3.0, 4.0, 5.0])

    # Create trainer config with checkpoint callbacks
    # Set primary_metric to match our logged metric
    max_epochs = len(model_config.metric_values)
    trainer_config = TrainerConfig(
        barebones=False,
        loggers=[nt.configs.CSVLoggerConfig()],
        max_epochs=max_epochs,
        accelerator="cpu",
        primary_metric=nt.MetricConfig(monitor=model_config.metric_name, mode="min"),
    ).with_project_root(logs_dir)

    # Create trainer and model
    trainer = NSHTrainer(trainer_config)
    model = SimpleModule(model_config)

    # Create data loaders
    train_dl = DataLoader(
        SimpleDataset(10),
        batch_size=2,
        num_workers=0,
    )
    val_dl = DataLoader(
        SimpleDataset(10),
        batch_size=2,
        num_workers=0,
    )

    # Run training
    trainer.fit(model, train_dl, val_dl)

    # Verify checkpoints - best value should be 1.0 (the first and best value)
    ckpt_dir = trainer.hparams.directory.resolve_subdirectory(
        trainer.hparams.id, "checkpoint"
    )
    _verify_last_checkpoints(
        ckpt_dir,
        epoch=max_epochs - 1,
        step=max_epochs * len(train_dl),
    )
    _verify_best_checkpoints(
        ckpt_dir,
        metric_name=model_config.metric_name,
        metric_value=model_config.metric_values[0],
    )


def test_fluctuating_metrics_checkpoint(tmp_path):
    """Test checkpoints with fluctuating metrics using self.log()."""
    logs_dir = Path(tmp_path)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Model config with fluctuating metric values (lower is better)
    model_config = SimpleModuleConfig(metric_values=[3.0, 2.0, 4.0, 1.0, 5.0])

    # Create trainer config with checkpoint callbacks
    # Set primary_metric to match our logged metric
    max_epochs = len(model_config.metric_values)
    trainer_config = TrainerConfig(
        barebones=False,
        loggers=[nt.configs.CSVLoggerConfig()],
        max_epochs=max_epochs,
        accelerator="cpu",
        primary_metric=nt.MetricConfig(monitor=model_config.metric_name, mode="min"),
    ).with_project_root(logs_dir)

    # Create trainer and model
    trainer = NSHTrainer(trainer_config)
    model = SimpleModule(model_config)

    # Create data loaders
    train_dl = DataLoader(
        SimpleDataset(10),
        batch_size=2,
        num_workers=0,
    )
    val_dl = DataLoader(
        SimpleDataset(10),
        batch_size=2,
        num_workers=0,
    )

    # Run training
    trainer.fit(model, train_dl, val_dl)

    # Verify checkpoints - best value should be 1.0 (the fourth value, which is best)
    ckpt_dir = trainer.hparams.directory.resolve_subdirectory(
        trainer.hparams.id, "checkpoint"
    )
    _verify_last_checkpoints(
        ckpt_dir,
        epoch=max_epochs - 1,
        step=max_epochs * len(train_dl),
    )
    _verify_best_checkpoints(
        ckpt_dir,
        metric_name=model_config.metric_name,
        metric_value=model_config.metric_values[3],
    )
