import copy
import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import nshconfig as C
import numpy as np
import torch

from ..model._environment import EnvironmentConfig

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
