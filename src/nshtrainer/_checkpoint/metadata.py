import copy
import datetime
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import nshconfig as C
import numpy as np
import torch

from ..util._environment_info import EnvironmentConfig

if TYPE_CHECKING:
    from ..model import BaseConfig, LightningModuleBase
    from ..trainer.trainer import Trainer

log = logging.getLogger(__name__)


METADATA_PATH_SUFFIX = ".metadata.json"


class CheckpointMetadata(C.Config):
    PATH_SUFFIX: ClassVar[str] = METADATA_PATH_SUFFIX

    checkpoint_path: Path
    checkpoint_filename: str

    run_id: str
    name: str
    project: str | None
    checkpoint_timestamp: datetime.datetime
    start_timestamp: datetime.datetime | None

    epoch: int
    global_step: int
    training_time: datetime.timedelta
    metrics: dict[str, Any]
    environment: EnvironmentConfig

    hparams: Any

    @classmethod
    def from_file(cls, path: Path):
        return cls.model_validate_json(path.read_text())

    @classmethod
    def from_ckpt_path(cls, checkpoint_path: Path):
        if not (metadata_path := checkpoint_path.with_suffix(cls.PATH_SUFFIX)).exists():
            raise FileNotFoundError(
                f"Metadata file not found for checkpoint: {checkpoint_path}"
            )
        return cls.from_file(metadata_path)


def _generate_checkpoint_metadata(
    config: "BaseConfig",
    trainer: "Trainer",
    checkpoint_path: Path,
    metadata_path: Path,
):
    checkpoint_timestamp = datetime.datetime.now()
    start_timestamp = trainer.start_time()
    training_time = trainer.time_elapsed()

    metrics: dict[str, Any] = {}
    for name, metric in copy.deepcopy(trainer.callback_metrics).items():
        match metric:
            case torch.Tensor() | np.ndarray():
                metrics[name] = metric.detach().cpu().item()
            case _:
                metrics[name] = metric

    return CheckpointMetadata(
        # checkpoint_path=checkpoint_path,
        # We should store the path as a relative path
        # to the metadata file to avoid issues with
        # moving the checkpoint directory
        checkpoint_path=checkpoint_path.relative_to(metadata_path.parent),
        checkpoint_filename=checkpoint_path.name,
        run_id=config.id,
        name=config.run_name,
        project=config.project,
        checkpoint_timestamp=checkpoint_timestamp,
        start_timestamp=start_timestamp.datetime
        if start_timestamp is not None
        else None,
        epoch=trainer.current_epoch,
        global_step=trainer.global_step,
        training_time=training_time,
        metrics=metrics,
        environment=config.environment,
        hparams=config.model_dump(),
    )


def _write_checkpoint_metadata(
    trainer: "Trainer",
    model: "LightningModuleBase",
    checkpoint_path: Path,
):
    config = cast("BaseConfig", model.config)
    metadata_path = checkpoint_path.with_suffix(CheckpointMetadata.PATH_SUFFIX)
    metadata = _generate_checkpoint_metadata(
        config, trainer, checkpoint_path, metadata_path
    )

    # Write the metadata to the checkpoint directory
    try:
        metadata_path.write_text(metadata.model_dump_json(indent=4), encoding="utf-8")
    except Exception:
        log.exception(f"Failed to write metadata to {checkpoint_path}")
    else:
        log.debug(f"Checkpoint metadata written to {checkpoint_path}")


def _remove_checkpoint_metadata(checkpoint_path: Path):
    path = checkpoint_path.with_suffix(CheckpointMetadata.PATH_SUFFIX)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        log.exception(f"Failed to remove {path}")
    else:
        log.debug(f"Removed {path}")


def _link_checkpoint_metadata(checkpoint_path: Path, linked_checkpoint_path: Path):
    # First, remove any existing metadata files
    _remove_checkpoint_metadata(linked_checkpoint_path)

    # Link the metadata files to the new checkpoint
    path = checkpoint_path.with_suffix(CheckpointMetadata.PATH_SUFFIX)
    linked_path = linked_checkpoint_path.with_suffix(CheckpointMetadata.PATH_SUFFIX)
    try:
        try:
            # linked_path.symlink_to(path)
            # We should store the path as a relative path
            # to the metadata file to avoid issues with
            # moving the checkpoint directory
            linked_path.symlink_to(path.relative_to(linked_path.parent))
        except OSError:
            # on Windows, special permissions are required to create symbolic links as a regular user
            # fall back to copying the file
            shutil.copy(path, linked_path)
    except Exception:
        log.exception(f"Failed to link {path} to {linked_path}")
    else:
        log.debug(f"Linked {path} to {linked_path}")


def _sort_ckpts_by_metadata(
    checkpoint_paths: list[Path],
    key: Callable[[CheckpointMetadata, Path], Any],
    reverse: bool = False,
):
    return sorted(
        [(CheckpointMetadata.from_ckpt_path(path), path) for path in checkpoint_paths],
        key=lambda args_tuple: key(*args_tuple),
        reverse=reverse,
    )
