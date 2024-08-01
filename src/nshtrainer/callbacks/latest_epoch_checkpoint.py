import logging
from pathlib import Path
from typing import Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Checkpoint
from typing_extensions import override

from .._checkpoint.metadata import _sort_ckpts_by_metadata
from .._checkpoint.saver import _link_checkpoint, _remove_checkpoint
from .base import CallbackConfigBase

log = logging.getLogger(__name__)


class LatestEpochCheckpointCallbackConfig(CallbackConfigBase):
    name: Literal["latest_epoch_checkpoint"] = "latest_epoch_checkpoint"

    dirpath: str | Path | None = None
    """Directory path to save the checkpoint file."""

    filename: str = "epoch{epoch:02d}_step{step:04d}"
    """Checkpoint filename. This must not include the extension."""

    save_weights_only: bool = False
    """Whether to save only the model's weights or the entire model object."""

    latest_symlink_filename: str | None = "latest"
    """Filename for the latest symlink. If None, no symlink will be created."""

    latest_k: int | Literal["all"] = 1
    """Number of latest checkpoints to keep. If "all", all checkpoints are kept."""

    @override
    def create_callbacks(self, root_config):
        dirpath = self.dirpath or root_config.directory.resolve_subdirectory(
            root_config.id, "checkpoint"
        )
        dirpath = Path(dirpath)

        yield LatestEpochCheckpoint(self, dirpath)


class LatestEpochCheckpoint(Checkpoint):
    PREFIX = "latest_"
    EXTENSION = ".ckpt"

    def __init__(self, config: LatestEpochCheckpointCallbackConfig, dirpath: Path):
        super().__init__()

        self.config = config
        self.dirpath = dirpath

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self._save_new_checkpoint(trainer)

    def _latest_symlink_filename(self):
        if (filename := self.config.latest_symlink_filename) is None:
            return None
        return f"{filename}{self.EXTENSION}"

    def _ckpt_path(self, trainer: Trainer):
        filename = self.config.filename.format(
            epoch=trainer.current_epoch, step=trainer.global_step
        )
        filename = f"{self.PREFIX}{filename}.{self.EXTENSION}"
        return self.dirpath / filename

    def _remove_checkpoints(self, trainer: Trainer, ckpt_paths: list[Path]):
        for ckpt_path in ckpt_paths:
            _remove_checkpoint(trainer, ckpt_path, metadata=True, barrier=False)

    def _remove_old_checkpoints(self, trainer: Trainer):
        if (latest_k := self.config.latest_k) == "all":
            return

        # Get all configs, ignoring the latest symlink
        ckpt_paths = list(self.dirpath.glob(f"{self.PREFIX}*{self.EXTENSION}"))
        # Ignore the latest symlink
        if (latest_symlink_filename := self._latest_symlink_filename()) is not None:
            ckpt_paths = [p for p in ckpt_paths if p.name != latest_symlink_filename]

        # Sort by epoch, then step, then last modified
        ckpt_paths = _sort_ckpts_by_metadata(
            ckpt_paths,
            key=lambda meta, p: (meta.epoch, meta.global_step, p.stat().st_mtime),
            fallback_key=lambda p: p.stat().st_mtime,
            # ^ Called if metadata is not found on all checkpoints
        )

        # Remove all but the latest k checkpoints
        ckpts_to_remove = ckpt_paths[:-latest_k]
        self._remove_checkpoints(trainer, ckpts_to_remove)

    def _save_new_checkpoint(self, trainer: Trainer):
        # Remove old checkpoints
        self._remove_old_checkpoints(trainer)

        # Save the new checkpoint
        filepath = self._ckpt_path(trainer)
        trainer.save_checkpoint(filepath, self.config.save_weights_only)

        # Create the latest symlink
        if (symlink_filename := self._latest_symlink_filename()) is not None:
            symlink_path = self.dirpath / symlink_filename
            _link_checkpoint(
                trainer,
                filepath,
                symlink_path,
                barrier=True,
                metadata=True,
            )
            log.info(f"Created latest symlink: {symlink_path}")
