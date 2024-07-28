import re
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from lightning.pytorch.callbacks.model_checkpoint import (
    ModelCheckpoint as _ModelCheckpoint,
)
from typing_extensions import override

from ..metrics import MetricConfig
from .base import CallbackConfigBase

if TYPE_CHECKING:
    from ..model.config import BaseConfig

log = getLogger(__name__)


def _convert_string(input_string: str):
    # Find all variables enclosed in curly braces
    variables = re.findall(r"\{(.*?)\}", input_string)

    # Replace each variable with its corresponding key-value pair
    output_string = input_string
    for variable in variables:
        # If the name is something like {variable:format}, we shouldn't process the format.
        key_name = variable
        if ":" in variable:
            key_name, _ = variable.split(":", 1)
            continue

        # Replace '/' with '_' in the key name
        key_name = key_name.replace("/", "_")
        output_string = output_string.replace(
            f"{{{variable}}}", f"{key_name}={{{variable}}}"
        )

    return output_string


class ModelCheckpointCallbackConfig(CallbackConfigBase):
    """Arguments for the ModelCheckpoint callback."""

    name: Literal["model_checkpoint"] = "model_checkpoint"

    dirpath: str | Path | None = None
    """
    Directory path to save the model file. If `None`, we save to the checkpoint directory set in `config.directory`.
    """

    filename: str | None = None
    """
    Checkpoint filename.
        If None, a default template is used (see :attr:`ModelCheckpoint.CHECKPOINT_JOIN_CHAR`).
    """

    metric: MetricConfig | None = None
    """
    Metric to monitor for saving checkpoints.
        If None, the primary metric of the runner will be used, if available.
    """

    verbose: bool = False
    """Verbosity mode. If True, print additional information about checkpoints."""

    save_last: Literal[True, False, "link"] | None = "link"
    """
    Whether to save the last checkpoint.
        If True, saves a copy of the last checkpoint separately.
        If "link", creates a symbolic link to the last checkpoint.
    """

    save_top_k: int = 1
    """
    Number of best models to save.
        If -1, all models are saved.
        If 0, no models are saved.
    """

    save_weights_only: bool = False
    """Whether to save only the model's weights or the entire model object."""

    auto_insert_metric_name: bool = True
    """Whether to automatically insert the metric name in the checkpoint filename."""

    every_n_train_steps: int | None = None
    """
    Number of training steps between checkpoints.
        If None or 0, no checkpoints are saved during training.
    """

    train_time_interval: timedelta | None = None
    """
    Time interval between checkpoints during training.
        If None, no checkpoints are saved during training based on time.
    """

    every_n_epochs: int | None = None
    """
    Number of epochs between checkpoints.
        If None or 0, no checkpoints are saved at the end of epochs.
    """

    save_on_train_epoch_end: bool | None = None
    """
    Whether to run checkpointing at the end of the training epoch.
        If False, checkpointing runs at the end of the validation.
    """

    enable_version_counter: bool = True
    """Whether to append a version to the existing file name."""

    auto_append_metric: bool = True
    """If enabled, this will automatically add "-{monitor}" to the filename."""

    def metric_or_default(self, root_config: "BaseConfig"):
        if self.metric is not None:
            return self.metric
        if root_config.primary_metric is not None:
            return root_config.primary_metric
        raise ValueError("Primary metric must be provided if metric is not specified.")

    def resolve_filename(self, root_config: "BaseConfig"):
        metric = self.metric_or_default(root_config)

        filename = self.filename
        if not filename:
            filename = "{epoch}-{step}"
        if self.auto_append_metric:
            filename = f"{filename}-{{{metric.validation_monitor}}}"

        if self.auto_insert_metric_name and filename:
            new_filename = _convert_string(filename)
            log.critical(
                f"Updated ModelCheckpoint filename: {filename} -> {new_filename}"
            )
            filename = new_filename

        return filename

    @override
    def create_callbacks(self, root_config):
        dirpath = self.dirpath or root_config.directory.resolve_subdirectory(
            root_config.id, "checkpoint"
        )

        metric = self.metric_or_default(root_config)
        filename = self.resolve_filename(root_config)

        yield ModelCheckpoint(
            self,
            dirpath=Path(dirpath),
            filename=filename,
            metric=metric,
        )


class ModelCheckpoint(_ModelCheckpoint):
    @override
    def __init__(
        self,
        config: ModelCheckpointCallbackConfig,
        dirpath: Path,
        filename: str,
        metric: MetricConfig,
    ):
        self.config = config
        del config

        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=metric.validation_monitor,
            mode=metric.mode,
            verbose=self.config.verbose,
            save_last=self.config.save_last,
            save_top_k=self.config.save_top_k,
            save_weights_only=self.config.save_weights_only,
            auto_insert_metric_name=False,
            every_n_train_steps=self.config.every_n_train_steps,
            train_time_interval=self.config.train_time_interval,
            every_n_epochs=self.config.every_n_epochs,
            save_on_train_epoch_end=self.config.save_on_train_epoch_end,
            enable_version_counter=self.config.enable_version_counter,
        )
