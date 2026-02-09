"""Helper script for test_distributed_prediction_writer.py.

This runs in a separate process to ensure a clean CUDA state.
Exit code 77 means "skip" (not enough GPUs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import nshconfig as C
import torch
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
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_indices = x.int()
        return {
            "predictions": torch.stack(
                [
                    batch_indices,
                    batch_indices * 2,
                    batch_indices + 100,
                ],
                dim=1,
            ),
            "batch_indices": batch_indices,
        }


def main():
    if torch.cuda.device_count() < 2:
        sys.exit(77)  # Skip signal

    tmp_path = Path(sys.argv[1])
    logs_dir = Path(tmp_path)
    logs_dir.mkdir(parents=True, exist_ok=True)

    hparams = TrainerConfig(
        barebones=True,
        fast_dev_run=False,
        max_epochs=1,
        devices=(0, 1),
        accelerator="gpu",
        strategy="ddp_notebook",
    ).with_project_root(logs_dir)

    trainer = NSHTrainer(hparams)
    hparams = SimpleModuleConfig()
    model = SimpleModule(hparams)
    dl = DataLoader(
        SimpleDataset(20),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    result = trainer.distributed_predict(model, dl)
    if not trainer.strategy.is_global_zero:
        return

    output_dir = result.root_dir
    assert output_dir.exists(), "Output directory does not exist."

    reader = result.get_processed_reader()
    assert reader is not None, "Reader should not be None."
    assert len(reader) == 20, "Reader length does not match dataset size."

    for i, sample in enumerate(reader):
        batch, predictions = sample["batch"], sample["prediction"]
        assert predictions["predictions"].tolist() == [[i], [i * 2], [i + 100]], (
            f"Prediction mismatch at index {i}: {sample['index']=}."
        )
        assert predictions["batch_indices"].tolist() == [i], (
            f"Batch index mismatch at index {i}: {sample['index']=}."
        )
        assert batch.tolist() == [i], (
            f"Batch index mismatch at index {i}: {sample['index']=}."
        )


if __name__ == "__main__":
    main()
