"""Test for the distributed prediction writer and reader.

This test creates a simple lightning module that returns constant predictions,
runs distributed prediction, and verifies that the reader works correctly.
"""

from __future__ import annotations

from pathlib import Path

import nshconfig as C
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import nshtrainer
from nshtrainer.callbacks.distributed_prediction_writer import (
    DistributedPredictionWriter,
    DistributedPredictionWriterConfig,
)
from nshtrainer.trainer._config import TrainerConfig
from nshtrainer.trainer.trainer import Trainer as NSHTrainer


class SimpleDataset(Dataset):
    """Simple dataset that returns known indices and constants."""

    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx])


class SimpleModuleConfig(C.Config):
    pass


class SimpleModule(nshtrainer.LightningModuleBase):
    """Simple module that returns known constants for testing prediction."""

    @classmethod
    def hparams_cls(cls):
        return SimpleModuleConfig

    def __init__(self, hparams):
        super().__init__(hparams)
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Return predictable outputs based on the input
        # We'll create a dictionary to test complex output handling
        batch_indices = x.int()  # First column has the index
        return {
            "predictions": torch.stack(
                [
                    batch_indices,  # Return the indices
                    batch_indices * 2,  # Doubled indices
                    batch_indices + 100,  # Indices + 100
                ],
                dim=1,
            ),
            "batch_indices": batch_indices,
        }


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs for distributed testing",
)
def test_distributed_prediction_writer_and_reader(tmp_path):
    """Test the distributed prediction writer and reader with a simple module."""
    # Set up output directory using pytest tmp_path
    logs_dir = Path(tmp_path)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer config with distributed prediction writer
    hparams = TrainerConfig(
        barebones=True,
        fast_dev_run=False,
        max_epochs=1,
        devices=(0, 1),  # Run on 2 devices
        accelerator="gpu",
        strategy="ddp_notebook",
        distributed_predict=DistributedPredictionWriterConfig(),
    ).with_project_root(logs_dir)

    # Create trainer, module, and datamodule
    trainer = NSHTrainer(hparams)
    hparams = SimpleModuleConfig()
    model = SimpleModule(hparams)
    dl = DataLoader(
        SimpleDataset(20),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    # Run prediction
    trainer.distributed_predict(model, dl)

    trainer.strategy.barrier()  # Ensure all processes finish before checking output
    if not trainer.strategy.is_global_zero:
        # Skip the test for non-global zero processes
        return

    # Check if the output directory exists
    assert (
        writer_callback := next(
            (
                c
                for c in trainer.callbacks
                if isinstance(c, DistributedPredictionWriter)
            ),
            None,
        )
    ) is not None, "Writer callback not found in trainer callbacks."
    output_dir = writer_callback.output_dir
    assert output_dir.exists(), "Output directory does not exist."

    from nshtrainer.callbacks.distributed_prediction_writer import (
        DistributedPredictionReader,
    )

    reader = DistributedPredictionReader(output_dir / "processed" / "dataloader_0")
    assert len(reader) == 20, "Reader length does not match dataset size."

    # Check if the predictions are as expected
    for i, (batch, predictions) in enumerate(reader):
        assert predictions["predictions"].tolist() == [[i], [i * 2], [i + 100]], (
            f"Prediction mismatch at index {i}."
        )
        assert predictions["batch_indices"].tolist() == [i], (
            f"Batch index mismatch at index {i}."
        )
        assert batch.tolist() == [i], f"Batch index mismatch at index {i}."
