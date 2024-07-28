import logging
from pathlib import Path
from typing import Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Checkpoint
from typing_extensions import override

from .base import CallbackConfigBase

log = logging.getLogger(__name__)


class LatestEpochCheckpointCallbackConfig(CallbackConfigBase):
    name: Literal["latest_epoch_checkpoint"] = "latest_epoch_checkpoint"

    dirpath: str | Path | None = None
    """Directory path to save the checkpoint file."""

    filename: str = "latest_epoch{epoch:02d}_step{step:04d}.ckpt"
    """Checkpoint filename. This must not include the extension."""

    save_weights_only: bool = False
    """Whether to save only the model's weights or the entire model object."""

    latest_symlink_filename: str | None = "latest.ckpt"
    """Filename for the latest symlink. If None, no symlink will be created."""

    @override
    def create_callbacks(self, root_config):
        dirpath = self.dirpath or root_config.directory.resolve_subdirectory(
            root_config.id, "checkpoint"
        )
        dirpath = Path(dirpath)

        yield LatestEpochCheckpoint(self, dirpath)


class LatestEpochCheckpoint(Checkpoint):
    def __init__(self, config: LatestEpochCheckpointCallbackConfig, dirpath: Path):
        super().__init__()

        self.config = config
        self.dirpath = dirpath

        # Also, we hold a reference to the last checkpoint path
        # to be able to remove it when a new checkpoint is saved.
        self._last_ckpt_path: Path | None = None

    def _ckpt_path(self, trainer: Trainer):
        return self.dirpath / self.config.filename.format(
            epoch=trainer.current_epoch, step=trainer.global_step
        )

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # Remove the last checkpoint if it exists
        if self._last_ckpt_path is not None:
            trainer.strategy.remove_checkpoint(self._last_ckpt_path)

        # Save the new checkpoint
        filepath = self._ckpt_path(trainer)
        trainer.save_checkpoint(filepath, self.config.save_weights_only)
        self._last_ckpt_path = filepath

        # Create the latest symlink
        if (symlink_filename := self.config.latest_symlink_filename) is not None:
            symlink_path = self.dirpath / symlink_filename
            if symlink_path.exists():
                symlink_path.unlink()
            symlink_path.symlink_to(filepath.name)
            log.info(f"Created latest symlink: {symlink_path}")

    def latest_checkpoint(self):
        if (symlink_filename := self.config.latest_symlink_filename) is None:
            return None

        if not (symlink_path := self.dirpath / symlink_filename).exists():
            return None

        return symlink_path
