from __future__ import annotations

import logging
import math
from typing import Any, Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from typing_extensions import final, override

from .base import CallbackConfigBase, callback_registry

log = logging.getLogger(__name__)


@final
@callback_registry.register
class LogEpochCallbackConfig(CallbackConfigBase):
    name: Literal["log_epoch"] = "log_epoch"

    metric_name: str = "computed_epoch"
    """The name of the metric to log the epoch as."""

    train: bool = True
    """Whether to log the epoch during training."""

    val: bool = True
    """Whether to log the epoch during validation."""

    test: bool = True
    """Whether to log the epoch during testing."""

    @override
    def create_callbacks(self, trainer_config):
        yield LogEpochCallback(self)


def _log_on_step(
    trainer: Trainer,
    pl_module: LightningModule,
    num_batches_prop: str,
    dataloader_idx: int | None = None,
    *,
    metric_name: str,
):
    if trainer.logger is None:
        return

    # If trainer.num_{training/val/test}_batches is not set or is nan/inf, we cannot calculate the epoch
    if not (num_batches := getattr(trainer, num_batches_prop, None)):
        log.warning(f"Trainer has no valid `{num_batches_prop}`. Cannot log epoch.")
        return

    # If the trainer has a dataloader_idx, num_batches is a list of num_batches for each dataloader.
    if dataloader_idx is not None:
        assert isinstance(num_batches, list), (
            f"Expected num_batches to be a list, got {type(num_batches)}"
        )
        assert 0 <= dataloader_idx < len(num_batches), (
            f"Expected dataloader_idx to be between 0 and {len(num_batches)}, got {dataloader_idx}"
        )
        num_batches = num_batches[dataloader_idx]

    if (
        not isinstance(num_batches, (int, float))
        or math.isnan(num_batches)
        or math.isinf(num_batches)
    ):
        log.warning(
            f"Trainer has no valid `{num_batches_prop}` (got {num_batches=}). Cannot log epoch."
        )
        return

    epoch = pl_module.global_step / num_batches
    pl_module.log(metric_name, epoch, on_step=True, on_epoch=False)


def _log_on_epoch(
    trainer: Trainer,
    pl_module: LightningModule,
    *,
    metric_name: str,
):
    if trainer.logger is None:
        return

    epoch = pl_module.current_epoch + 1
    pl_module.log(metric_name, epoch, on_step=False, on_epoch=True)


class LogEpochCallback(Callback):
    def __init__(self, config: LogEpochCallbackConfig):
        super().__init__()

        self.config = config

    @override
    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ):
        if not self.config.train:
            return

        _log_on_step(
            trainer,
            pl_module,
            "num_training_batches",
            metric_name=self.config.metric_name,
        )

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not self.config.train:
            return

        _log_on_epoch(
            trainer,
            pl_module,
            metric_name=self.config.metric_name,
        )

    @override
    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.config.val:
            return

        _log_on_step(
            trainer,
            pl_module,
            "num_val_batches",
            dataloader_idx=dataloader_idx,
            metric_name=self.config.metric_name,
        )

    @override
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not self.config.val:
            return

        _log_on_epoch(
            trainer,
            pl_module,
            metric_name=self.config.metric_name,
        )

    @override
    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.config.test:
            return

        _log_on_step(
            trainer,
            pl_module,
            "num_test_batches",
            dataloader_idx=dataloader_idx,
            metric_name=self.config.metric_name,
        )

    @override
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.config.test:
            return

        _log_on_epoch(
            trainer,
            pl_module,
            metric_name=self.config.metric_name,
        )
