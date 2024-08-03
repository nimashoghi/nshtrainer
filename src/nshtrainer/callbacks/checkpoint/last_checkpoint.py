import logging
from typing import Literal

from lightning.pytorch import LightningModule, Trainer
from typing_extensions import final, override

from nshtrainer._checkpoint.metadata import CheckpointMetadata

from ._base import BaseCheckpointCallbackConfig, CheckpointBase

log = logging.getLogger(__name__)


@final
class LastCheckpointCallbackConfig(BaseCheckpointCallbackConfig):
    name: Literal["last_checkpoint"] = "last_checkpoint"

    @override
    def create_checkpoint(self, root_config, dirpath):
        return LastCheckpoint(self, dirpath)


@final
class LastCheckpoint(CheckpointBase[LastCheckpointCallbackConfig]):
    @override
    def name(self):
        return "last"

    @override
    def default_filename(self):
        return "epoch{epoch:03d}-step{step:07d}"

    @override
    def topk_sort_key(self, metadata: CheckpointMetadata):
        return metadata.checkpoint_timestamp

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.save_checkpoints(trainer)