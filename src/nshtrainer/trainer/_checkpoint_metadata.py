import copy
import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import nshconfig as C
import numpy as np
import torch

from ..model._environment import EnvironmentConfig

if TYPE_CHECKING:
    from ..model import BaseConfig, LightningModuleBase
    from .trainer import Trainer


class CheckpointMetadata(C.Config):
    id: str
    checkpoint_path: Path
    name: str
    project: str | None
    checkpoint_timestamp: datetime.datetime
    start_timestamp: datetime.datetime | None

    epoch: int
    global_step: int
    training_time: datetime.timedelta
    metrics: dict[str, Any]
    environment: EnvironmentConfig


def _generate_checkpoint_metadata(
    trainer: "Trainer",
    model: "LightningModuleBase",
    checkpoint_path: Path,
):
    config = cast("BaseConfig", model.config)

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
        id=config.id,
        checkpoint_path=checkpoint_path,
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
