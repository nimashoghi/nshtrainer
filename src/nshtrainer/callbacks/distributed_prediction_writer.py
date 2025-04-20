from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Literal

import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from typing_extensions import final, override

from .base import CallbackConfigBase, CallbackMetadataConfig, callback_registry

log = logging.getLogger(__name__)


@final
@callback_registry.register
class DistributedPredictionWriterConfig(CallbackConfigBase):
    metadata: ClassVar[CallbackMetadataConfig] = CallbackMetadataConfig(
        enabled_for_barebones=True
    )
    """Metadata for the callback."""

    name: Literal["distributed_prediction_writer"] = "distributed_prediction_writer"

    write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch"
    """When to write the predictions. Can be 'batch', 'epoch' or 'batch_and_epoch'."""

    dirpath: Path | None = None
    """Directory to save the predictions to. If None, will use the default directory."""

    @override
    def create_callbacks(self, trainer_config):
        if (dirpath := self.dirpath) is None:
            dirpath = trainer_config.directory.resolve_subdirectory(
                trainer_config.id, "predictions"
            )

        yield DistributedPredictionWriter(self, dirpath)


class DistributedPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        config: DistributedPredictionWriterConfig,
        output_dir: Path,
    ):
        self.config = config

        super().__init__(write_interval=self.config.write_interval)

        self.output_dir = output_dir

    @override
    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        output_dir = (
            self.output_dir
            / "per_batch"
            / f"dataloader_{dataloader_idx}"
            / f"rank_{trainer.global_rank}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if trainer.is_global_zero:
            torch.save(trainer.world_size, output_dir / "world_size.pt")

        torch.save(
            prediction,
            output_dir / f"predictions_{batch_idx}.pt",
        )
        torch.save(
            batch,
            output_dir / f"batch_{batch_idx}.pt",
        )
        torch.save(
            batch_indices,
            output_dir / f"batch_indices_{batch_idx}.pt",
        )

    @override
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        output_dir = self.output_dir / "per_epoch" / f"rank_{trainer.global_rank}"
        output_dir.mkdir(parents=True, exist_ok=True)
        if trainer.is_global_zero:
            torch.save(trainer.world_size, output_dir / "world_size.pt")

        torch.save(
            predictions,
            output_dir / f"predictions.pt",
        )
        torch.save(
            batch_indices,
            output_dir / f"batch_indices.pt",
        )
