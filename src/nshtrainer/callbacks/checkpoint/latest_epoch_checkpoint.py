import logging
from pathlib import Path
from typing import Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Checkpoint
from typing_extensions import override

from ..._checkpoint.metadata import _sort_ckpts_by_metadata
from ..._checkpoint.saver import _link_checkpoint, _remove_checkpoint
from ..base import CallbackConfigBase

log = logging.getLogger(__name__)


class LatestEpochCheckpointCallbackConfig(CallbackConfigBase):
    name: Literal["latest_epoch_checkpoint"] = "latest_epoch_checkpoint"

    dirpath: str | Path | None = None
    """Directory path to save the checkpoint file."""

    filename: str = "epoch{epoch:03d}_step{step:07d}"
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

        self._last_global_step_saved = 0

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
        filename = f"{self.PREFIX}{filename}{self.EXTENSION}"
        return self.dirpath / filename

    def _remove_old_checkpoints(self, trainer: Trainer):
        if (latest_k := self.config.latest_k) == "all":
            return

        # Get all configs, ignoring the latest symlink
        ckpts = list(self.dirpath.glob(f"{self.PREFIX}*{self.EXTENSION}"))
        # Ignore the latest symlink
        if (latest_symlink_filename := self._latest_symlink_filename()) is not None:
            ckpts = [p for p in ckpts if p.name != latest_symlink_filename]

        # Sort by epoch, then step, then last modified
        ckpts = _sort_ckpts_by_metadata(
            ckpts,
            key=lambda meta, p: (meta.epoch, meta.global_step, p.stat().st_mtime),
            reverse=True,
        )

        # Remove all but the latest k checkpoints
        # NOTE: We add 1 to the latest_k here because
        # we're about to save a new checkpoint.
        for _, ckpt_path in ckpts[latest_k:]:
            _remove_checkpoint(trainer, ckpt_path, metadata=True)

    def _save_new_checkpoint(self, trainer: Trainer):
        if self._should_skip_saving_checkpoint(trainer):
            return

        # Save the new checkpoint
        filepath = self._ckpt_path(trainer)
        trainer.save_checkpoint(filepath, self.config.save_weights_only)

        if trainer.is_global_zero:
            # Remove old checkpoints
            self._remove_old_checkpoints(trainer)

            # Create the latest symlink
            if (symlink_filename := self._latest_symlink_filename()) is not None:
                symlink_path = self.dirpath / symlink_filename
                _link_checkpoint(filepath, symlink_path, metadata=True)
                log.debug(f"Created latest symlink: {symlink_path}")

        # Set the last global step saved
        self._last_global_step_saved = trainer.global_step

        # Barrier to ensure all processes have saved the checkpoint before continuing
        trainer.strategy.barrier()

    def _should_skip_saving_checkpoint(self, trainer: Trainer) -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        return (
            bool(
                getattr(trainer, "fast_dev_run", False)
            )  # disable checkpointing with fast_dev_run
            or trainer.state.fn
            != TrainerFn.FITTING  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
            or self._last_global_step_saved
            == trainer.global_step  # already saved at the last step
        )
