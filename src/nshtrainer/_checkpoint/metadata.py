import copy
import datetime
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import nshconfig as C
import numpy as np
import torch

from ..util._environment_info import EnvironmentConfig

if TYPE_CHECKING:
    from ..model import BaseConfig, LightningModuleBase
    from ..trainer.trainer import Trainer

log = logging.getLogger(__name__)


METADATA_PATH_SUFFIX = ".metadata.json"
HPARAMS_PATH_SUFFIX = ".hparams.json"


class CheckpointMetadata(C.Config):
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

    @classmethod
    def from_file(cls, path: Path):
        return cls.model_validate_json(path.read_text())


def _generate_checkpoint_metadata(
    config: "BaseConfig", trainer: "Trainer", checkpoint_path: Path
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
        checkpoint_path=checkpoint_path,
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
    )


def _write_checkpoint_metadata(
    trainer: "Trainer",
    model: "LightningModuleBase",
    checkpoint_path: Path,
):
    config = cast("BaseConfig", model.config)
    metadata = _generate_checkpoint_metadata(config, trainer, checkpoint_path)

    # Write the metadata to the checkpoint directory
    try:
        metadata_path = checkpoint_path.with_suffix(METADATA_PATH_SUFFIX)
        metadata_path.write_text(metadata.model_dump_json(indent=4))
    except Exception as e:
        log.warning(f"Failed to write metadata to {checkpoint_path}: {e}")
    else:
        log.info(f"Checkpoint metadata written to {checkpoint_path}")

    # Write the hparams to the checkpoint directory
    try:
        hparams_path = checkpoint_path.with_suffix(HPARAMS_PATH_SUFFIX)
        hparams_path.write_text(config.model_dump_json(indent=4))
    except Exception as e:
        log.warning(f"Failed to write hparams to {checkpoint_path}: {e}")
    else:
        log.info(f"Checkpoint metadata written to {checkpoint_path}")


def _remove_checkpoint_metadata(checkpoint_path: Path):
    for suffix in (METADATA_PATH_SUFFIX, HPARAMS_PATH_SUFFIX):
        path = checkpoint_path.with_suffix(suffix)
        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            log.warning(f"Failed to remove {path}: {e}")
        else:
            log.info(f"Removed {path}")


def _link_checkpoint_metadata(checkpoint_path: Path, linked_checkpoint_path: Path):
    # First, remove any existing metadata files
    _remove_checkpoint_metadata(linked_checkpoint_path)

    # Link the metadata files to the new checkpoint
    for suffix in (METADATA_PATH_SUFFIX, HPARAMS_PATH_SUFFIX):
        path = checkpoint_path.with_suffix(suffix)
        linked_path = linked_checkpoint_path.with_suffix(suffix)
        try:
            try:
                linked_path.symlink_to(path)
            except OSError:
                # on Windows, special permissions are required to create symbolic links as a regular user
                # fall back to copying the file
                shutil.copy(path, linked_path)
        except Exception as e:
            log.warning(f"Failed to link {path} to {linked_path}: {e}")
        else:
            log.info(f"Linked {path} to {linked_path}")


def _checkpoint_sort_key_fn(key: Callable[[CheckpointMetadata, Path], Any]):
    def sort_key_fn(checkpoint_path: Path):
        if not (p := checkpoint_path.with_suffix(METADATA_PATH_SUFFIX)).exists():
            raise FileNotFoundError(f"Metadata file not found: {p}")

        nonlocal key
        return key(CheckpointMetadata.from_file(p), p)

    return sort_key_fn


def _sort_ckpts_by_metadata(
    checkpoint_paths: list[Path],
    key: Callable[[CheckpointMetadata, Path], Any],
    fallback_key: Callable[[Path], Any],
):
    # First, let's make sure all the metadata files exist.
    # If not, use the fallback function to sort the checkpoints.
    no_metadata_paths: list[Path] = []
    for path in checkpoint_paths:
        if (path.with_suffix(METADATA_PATH_SUFFIX)).exists():
            continue

        no_metadata_paths.append(path)

    if no_metadata_paths:
        log.warning(
            f"Metadata file not found on {len(no_metadata_paths)} checkpoints: {no_metadata_paths}\n"
            "Falling back to sorting by last modified time."
        )
        return sorted(checkpoint_paths, key=fallback_key)

    return sorted(checkpoint_paths, key=_checkpoint_sort_key_fn(key))
