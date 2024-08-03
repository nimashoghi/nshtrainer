import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Checkpoint
from typing_extensions import TypeVar, override

from ..._checkpoint.metadata import CheckpointMetadata
from ..._checkpoint.saver import _link_checkpoint, _remove_checkpoint
from ..base import CallbackConfigBase

if TYPE_CHECKING:
    from ...model.config import BaseConfig

log = logging.getLogger(__name__)


class BaseCheckpointCallbackConfig(CallbackConfigBase, ABC):
    dirpath: str | Path | None = None
    """Directory path to save the checkpoint file."""

    filename: str | None = None
    """Checkpoint filename. This must not include the extension.
    If None, the default filename will be used."""

    save_weights_only: bool = False
    """Whether to save only the model's weights or the entire model object."""

    save_symlink: bool = True
    """Whether to create a symlink to the saved checkpoint."""

    topk: int | Literal["all"] = 1
    """The number of checkpoints to keep."""

    @abstractmethod
    def create_checkpoint(
        self,
        root_config: "BaseConfig",
        dirpath: Path,
    ) -> "CheckpointBase": ...

    @override
    def create_callbacks(self, root_config):
        dirpath = Path(
            self.dirpath
            or root_config.directory.resolve_subdirectory(root_config.id, "checkpoint")
        )

        yield self.create_checkpoint(root_config, dirpath)


TConfig = TypeVar("TConfig", bound=BaseCheckpointCallbackConfig, infer_variance=True)


class CheckpointBase(Checkpoint, ABC, Generic[TConfig]):
    def __init__(self, config: TConfig, dirpath: Path):
        super().__init__()

        self.config = config
        self.dirpath = dirpath / self.name()
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.symlink_dirpath = dirpath

        self._last_global_step_saved = 0

    @abstractmethod
    def default_filename(self) -> str: ...

    @abstractmethod
    def name(self) -> str: ...

    def extension(self) -> str:
        return ".ckpt"

    @abstractmethod
    def topk_sort_key(self, metadata: CheckpointMetadata) -> Any: ...

    @abstractmethod
    def topk_sort_reverse(self) -> bool: ...

    def symlink_path(self):
        if not self.config.save_symlink:
            return None

        return self.symlink_dirpath / f"{self.name()}{self.extension()}"

    def resolve_checkpoint_path(self, current_metrics: dict[str, Any]) -> Path:
        if (filename := self.config.filename) is None:
            filename = self.default_filename()
        filename = filename.format(**current_metrics)
        return self.dirpath / f"{filename}{self.extension()}"

    def remove_old_checkpoints(self, trainer: Trainer):
        if (topk := self.config.topk) == "all":
            return

        # Get all the checkpoint metadata
        metas = [
            CheckpointMetadata.from_file(p)
            for p in self.dirpath.glob(f"*{CheckpointMetadata.PATH_SUFFIX}")
        ]

        # Sort by the topk sort key
        metas = sorted(metas, key=self.topk_sort_key, reverse=self.topk_sort_reverse())

        # Now, the metas are sorted from the best to the worst,
        # so we can remove the worst checkpoints
        for meta in metas[topk:]:
            if not (old_ckpt_path := self.dirpath / meta.checkpoint_filename).exists():
                log.warning(
                    f"Checkpoint file not found: {old_ckpt_path}\n"
                    "Skipping removal of the checkpoint metadata."
                )
                continue

            _remove_checkpoint(trainer, old_ckpt_path, metadata=True)
            log.debug(f"Removed old checkpoint: {old_ckpt_path}")

    def current_metrics(self, trainer: Trainer) -> dict[str, Any]:
        current_metrics: dict[str, Any] = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
        }

        for name, value in trainer.callback_metrics.items():
            match value:
                case torch.Tensor() if value.numel() == 1:
                    value = value.detach().cpu().item()
                case np.ndarray() if value.size == 1:
                    value = value.item()
                case _:
                    pass

            current_metrics[name] = value

        return current_metrics

    def save_checkpoints(self, trainer: Trainer):
        if self._should_skip_saving_checkpoint(trainer):
            return

        # Save the new checkpoint
        filepath = self.resolve_checkpoint_path(self.current_metrics(trainer))
        trainer.save_checkpoint(filepath, self.config.save_weights_only)

        if trainer.is_global_zero:
            # Create the latest symlink
            if (symlink_filename := self.symlink_path()) is not None:
                symlink_path = self.dirpath / symlink_filename
                _link_checkpoint(filepath, symlink_path, metadata=True)
                log.debug(f"Created latest symlink: {symlink_path}")

            # Remove old checkpoints
            self.remove_old_checkpoints(trainer)

        # Barrier to ensure all processes have saved the checkpoint,
        # deleted the old checkpoints, and created the symlink before continuing
        trainer.strategy.barrier()

        # Set the last global step saved
        self._last_global_step_saved = trainer.global_step

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
