from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback as _LightningCallback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer
from typing_extensions import override

if TYPE_CHECKING:
    from .trainer import Trainer


class NTCallbackBase(_LightningCallback):
    @override
    def setup(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Called when fit, validate, test, predict, or tune begins."""

    @override
    def teardown(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Called when fit, validate, test, predict, or tune ends."""

    @override
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Called when fit begins."""

    @override
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Called when fit ends."""

    @override
    def on_sanity_check_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the validation sanity check starts."""

    @override
    def on_sanity_check_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the validation sanity check ends."""

    @override
    def on_train_batch_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when the train batch begins."""

    @override
    def on_train_batch_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.

        """

    @override
    def on_train_epoch_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the train epoch begins."""

    @override
    def on_train_epoch_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, you can cache step outputs as an attribute of the
        :class:`lightning.pytorch.core.LightningModule` and access them in this hook:

        .. code-block:: python

            class MyLightningModule(L.LightningModule):
                 @override
                 def __init__(self):
                    super().__init__() # pyright: ignore[reportIncompatibleMethodOverride]
                    self.training_step_outputs = []

                 @override
                 def training_step(self):
                    loss = ... # pyright: ignore[reportIncompatibleMethodOverride]
                    self.training_step_outputs.append(loss)
                    return loss


            class MyCallback(L.Callback):
                 @override
                 def on_train_epoch_end(self, trainer, pl_module):
                    # do something with all training_step outputs, for example: # pyright: ignore[reportIncompatibleMethodOverride]
                    epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
                    pl_module.log("training_epoch_mean", epoch_mean)
                    # free up the memory
                    pl_module.training_step_outputs.clear()

        """

    @override
    def on_validation_epoch_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the val epoch begins."""

    @override
    def on_validation_epoch_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the val epoch ends."""

    @override
    def on_test_epoch_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the test epoch begins."""

    @override
    def on_test_epoch_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the test epoch ends."""

    @override
    def on_predict_epoch_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the predict epoch begins."""

    @override
    def on_predict_epoch_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the predict epoch ends."""

    @override
    def on_validation_batch_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch begins."""

    @override
    def on_validation_batch_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch ends."""

    @override
    def on_test_batch_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch begins."""

    @override
    def on_test_batch_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""

    @override
    def on_predict_batch_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the predict batch begins."""

    @override
    def on_predict_batch_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the predict batch ends."""

    @override
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Called when the train begins."""

    @override
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Called when the train ends."""

    @override
    def on_validation_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the validation loop begins."""

    @override
    def on_validation_end(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the validation loop ends."""

    @override
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Called when the test begins."""

    @override
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Called when the test ends."""

    @override
    def on_predict_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called when the predict begins."""

    @override
    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Called when predict ends."""

    @override
    def on_exception(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        """Called when any trainer execution is interrupted by an exception."""

    @override
    def state_dict(self) -> dict[str, Any]:
        """Called when saving a checkpoint, implement to generate callback's ``state_dict``.

        Returns:
            A dictionary containing callback state.

        """
        return {}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload callback state given callback's ``state_dict``.

        Args:
            state_dict: the callback state returned by ``state_dict``.

        """
        pass

    @override
    def on_save_checkpoint(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        r"""Called when saving a checkpoint to give you a chance to store anything else you might want to save.

        Args:
            trainer: the current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: the current :class:`~lightning.pytorch.core.LightningModule` instance.
            checkpoint: the checkpoint dictionary that will be saved.

        """

    @override
    def on_load_checkpoint(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        r"""Called when loading a model checkpoint, use to reload state.

        Args:
            trainer: the current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: the current :class:`~lightning.pytorch.core.LightningModule` instance.
            checkpoint: the full checkpoint dictionary that got loaded by the Trainer.

        """

    @override
    def on_before_backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule, loss: torch.Tensor
    ) -> None:
        """Called before ``loss.backward()``."""

    @override
    def on_after_backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Called after ``loss.backward()`` and before optimizers are stepped."""

    @override
    def on_before_optimizer_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
    ) -> None:
        """Called before ``optimizer.step()``."""

    @override
    def on_before_zero_grad(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
    ) -> None:
        """Called before ``optimizer.zero_grad()``."""

    # =================================================================
    # Our own new callbacks
    # =================================================================
    def on_checkpoint_saved(
        self,
        ckpt_path: Path,
        metadata_path: Path | None,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Called after a checkpoint is saved."""
        pass


@override
def _call_on_checkpoint_saved(
    trainer: Trainer,
    ckpt_path: str | Path,
    metadata_path: str | Path | None,
):
    ckpt_path = Path(ckpt_path)
    metadata_path = Path(metadata_path) if metadata_path else None

    for callback in trainer.callbacks:
        if not isinstance(callback, NTCallbackBase):
            continue

        callback.on_checkpoint_saved(
            ckpt_path,
            metadata_path,
            trainer,
            trainer.lightning_module,
        )
