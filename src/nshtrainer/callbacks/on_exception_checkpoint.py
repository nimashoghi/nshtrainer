import datetime
import logging
import os
from pathlib import Path
from typing import Any, Literal

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import OnExceptionCheckpoint as _OnExceptionCheckpoint
from typing_extensions import override

from .base import CallbackConfigBase

log = logging.getLogger(__name__)


class OnExceptionCheckpointCallbackConfig(CallbackConfigBase):
    kind: Literal["on_exception_checkpoint"] = "on_exception_checkpoint"

    dirpath: str | Path | None = None
    """Directory path to save the checkpoint file."""

    filename: str | None = None
    """Checkpoint filename. This must not include the extension. If `None`, `on_exception_{id}_{timestamp}` is used."""

    @override
    def create_callbacks(self, root_config):
        from ..callbacks.on_exception_checkpoint import OnExceptionCheckpoint

        dirpath = self.dirpath or root_config.directory.resolve_subdirectory(
            root_config.id, "checkpoint"
        )

        if not (filename := self.filename):
            filename = f"on_exception_{root_config.id}"
        yield OnExceptionCheckpoint(self, dirpath=Path(dirpath), filename=filename)


class OnExceptionCheckpoint(_OnExceptionCheckpoint):
    @override
    def __init__(
        self,
        config: OnExceptionCheckpointCallbackConfig,
        dirpath: Path,
        filename: str,
    ):
        self.config = config
        del config

        super().__init__(dirpath, filename)

    @property
    @override
    def ckpt_path(self) -> str:
        ckpt_path = super().ckpt_path

        # Remve the extension and add the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ckpt_path, ext = os.path.splitext(ckpt_path)
        return f"{ckpt_path}_{timestamp}{ext}"

    @override
    def on_exception(self, trainer: Trainer, *_: Any, **__: Any) -> None:
        # We override this to checkpoint the model manually,
        # without calling the dist barrier.

        # trainer.save_checkpoint(self.ckpt_path)

        if trainer.model is None:
            raise AttributeError(
                "Saving a checkpoint is only possible if a model is attached to the Trainer. Did you call"
                " `Trainer.save_checkpoint()` before calling `Trainer.{fit,validate,test,predict}`?"
            )
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
        trainer.strategy.save_checkpoint(
            checkpoint, self.ckpt_path, storage_options=None
        )
        # self.strategy.barrier("Trainer.save_checkpoint") # <-- This is disabled
