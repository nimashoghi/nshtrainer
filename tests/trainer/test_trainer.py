"""Test for the distributed prediction writer and reader.

This test creates a simple lightning module that returns constant predictions,
runs distributed prediction, and verifies that the reader works correctly.
"""

from __future__ import annotations

from pathlib import Path

import nshconfig as C
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

import nshtrainer
from nshtrainer.trainer._config import TrainerConfig
from nshtrainer.trainer.trainer import Trainer as NSHTrainer


class SimpleDataset(Dataset):
    """Simple dataset that returns known indices and constants."""

    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    @override
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx])


class SimpleModuleConfig(C.Config):
    pass


class SimpleModule(nshtrainer.LightningModuleBase):
    """Simple module that returns known constants for testing prediction."""

    @override
    @classmethod
    def hparams_cls(cls):
        return SimpleModuleConfig

    def __init__(self, hparams):
        super().__init__(hparams)
        self.linear = torch.nn.Linear(2, 3)

    @override
    def forward(self, x: torch.Tensor):
        # Return predictable outputs based on the input
        # We'll create a dictionary to test complex output handling
        x = torch.stack([x, x * 2], dim=1).float().squeeze(dim=-1)  # Create a 2D tensor
        x = self.linear(x)  # Apply linear transformation
        return torch.stack(
            [
                x,  # Return the indices
                x * 2,  # Doubled indices
                x + 100,  # Indices + 100
            ],
            dim=1,
        )

    @override
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # Simulate a training step
        preds = self(batch)
        loss = F.mse_loss(preds, preds * 0.5)
        return loss

    @override
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # Simulate a validation step
        _ = self(batch)

    @override
    def configure_optimizers(self):
        # Use a simple optimizer for the test
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_fast_dev_run_fit(tmp_path):
    """Test the distributed prediction writer and reader with a simple module."""
    # Set up output directory using pytest tmp_path
    logs_dir = Path(tmp_path)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer config with distributed prediction writer
    hparams = TrainerConfig(
        barebones=True,
        max_epochs=1,
        accelerator="cpu",
    ).with_project_root(logs_dir)

    # Create trainer, module, and datamodule
    trainer = NSHTrainer(hparams)
    hparams = SimpleModuleConfig()
    model = SimpleModule(hparams)
    train_dl = DataLoader(
        SimpleDataset(20),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    val_dl = DataLoader(
        SimpleDataset(20),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    # Run prediction
    trainer.fit(model, train_dl, val_dl)


def test_fast_dev_run_validate(tmp_path):
    """Test the distributed prediction writer and reader with a simple module."""
    # Set up output directory using pytest tmp_path
    logs_dir = Path(tmp_path)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer config with distributed prediction writer
    hparams = TrainerConfig(
        barebones=True,
        max_epochs=1,
        accelerator="cpu",
    ).with_project_root(logs_dir)

    # Create trainer, module, and datamodule
    trainer = NSHTrainer(hparams)
    hparams = SimpleModuleConfig()
    model = SimpleModule(hparams)
    val_dl = DataLoader(
        SimpleDataset(20),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    # Run prediction
    trainer.validate(model, val_dl)


def test_fast_dev_run_predict(tmp_path):
    """Test the distributed prediction writer and reader with a simple module."""
    # Set up output directory using pytest tmp_path
    logs_dir = Path(tmp_path)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer config with distributed prediction writer
    hparams = TrainerConfig(
        barebones=True,
        max_epochs=1,
        accelerator="cpu",
    ).with_project_root(logs_dir)

    # Create trainer, module, and datamodule
    trainer = NSHTrainer(hparams)
    hparams = SimpleModuleConfig()
    model = SimpleModule(hparams)
    predict_dl = DataLoader(
        SimpleDataset(20),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    # Run prediction
    trainer.predict(model, predict_dl)
