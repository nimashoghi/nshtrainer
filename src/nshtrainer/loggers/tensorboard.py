import logging
from typing import Literal

import nshconfig as C
from typing_extensions import override

from ._base import BaseLoggerConfig

log = logging.getLogger(__name__)


def _tensorboard_available():
    try:
        from lightning.fabric.loggers.tensorboard import (
            _TENSORBOARD_AVAILABLE,
            _TENSORBOARDX_AVAILABLE,
        )

        if not _TENSORBOARD_AVAILABLE and not _TENSORBOARDX_AVAILABLE:
            log.warning(
                "TensorBoard/TensorBoardX not found. Disabling TensorBoardLogger. "
                "Please install TensorBoard with `pip install tensorboard` or "
                "TensorBoardX with `pip install tensorboardx` to enable TensorBoard logging."
            )
            return False
        return True
    except ImportError:
        return False


class TensorboardLoggerConfig(BaseLoggerConfig):
    name: Literal["tensorboard"] = "tensorboard"

    enabled: bool = C.Field(default_factory=lambda: _tensorboard_available())
    """Enable TensorBoard logging."""

    priority: int = 2
    """Priority of the logger. Higher priority loggers are created first."""

    log_graph: bool = False
    """
    Adds the computational graph to tensorboard. This requires that
        the user has defined the `self.example_input_array` attribute in their
        model.
    """

    default_hp_metric: bool = True
    """
    Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is
        called without a metric (otherwise calls to log_hyperparams without a metric are ignored).
    """

    prefix: str = ""
    """A string to put at the beginning of metric keys."""

    @override
    def create_logger(self, root_config):
        if not self.enabled:
            return None

        from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

        save_dir = root_config.directory._resolve_log_directory_for_logger(
            root_config.id,
            self,
        )
        return TensorBoardLogger(
            save_dir=save_dir,
            name=root_config.run_name,
            version=root_config.id,
            log_graph=self.log_graph,
            default_hp_metric=self.default_hp_metric,
        )