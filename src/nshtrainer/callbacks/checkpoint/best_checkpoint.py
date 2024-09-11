import logging
from pathlib import Path
from typing import Literal

from lightning.pytorch import LightningModule, Trainer
from typing_extensions import override

from nshtrainer._checkpoint.metadata import CheckpointMetadata

from ...metrics._config import MetricConfig
from ._base import BaseCheckpointCallbackConfig, CheckpointBase

log = logging.getLogger(__name__)


class BestCheckpointCallbackConfig(BaseCheckpointCallbackConfig):
    name: Literal["best_checkpoint"] = "best_checkpoint"

    metric: MetricConfig | None = None
    """Metric to monitor, or `None` to use the default metric."""

    @override
    def create_checkpoint(self, trainer_config, dirpath):
        # Resolve metric
        if (metric := self.metric) is None and (
            metric := trainer_config.primary_metric
        ) is None:
            raise ValueError(
                "No metric provided and no primary metric found in the root config"
            )

        return BestCheckpoint(self, dirpath, metric)


class BestCheckpoint(CheckpointBase[BestCheckpointCallbackConfig]):
    @property
    def _metric_name_normalized(self):
        return self.metric.name.replace("/", "_").replace(" ", "_").replace(".", "_")

    @override
    def __init__(
        self,
        config: BestCheckpointCallbackConfig,
        dirpath: Path,
        metric: MetricConfig,
    ):
        self.metric = metric
        super().__init__(config, dirpath)

    @override
    def name(self):
        return f"best_{self._metric_name_normalized}"

    @override
    def default_filename(self):
        return f"epoch{{epoch}}-step{{step}}-{self._metric_name_normalized}{{{self.metric.validation_monitor}}}"

    @override
    def topk_sort_key(self, metadata: CheckpointMetadata):
        return metadata.metrics.get(
            self.metric.validation_monitor,
            float("-inf" if self.metric.mode == "max" else "inf"),
        )

    @override
    def topk_sort_reverse(self):
        return self.metric.mode == "max"

    # Events
    @override
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.save_checkpoints(trainer)
