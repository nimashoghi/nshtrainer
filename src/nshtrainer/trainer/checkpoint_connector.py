from __future__ import annotations

import logging
from pathlib import Path

from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector as _LightningCheckpointConnector,
)
from lightning.pytorch.trainer.states import TrainerFn
from typing_extensions import override

from .._checkpoint.loader import CheckpointLoadingConfig, _resolve_checkpoint

log = logging.getLogger(__name__)


class _CheckpointConnector(_LightningCheckpointConnector):
    def __resolve_auto_ckpt_path(
        self,
        ckpt_path: str | Path | None,
        state_fn: TrainerFn,
    ):
        from .trainer import Trainer

        # If this isn't an `nshtrainer` trainer (which I don't know why it wouldn't be),
        # then we just default to the parent class's implementation of `_parse_ckpt_path`.
        trainer = self.trainer
        if not isinstance(trainer, Trainer):
            return None

        # Now, resolve the checkpoint loader config.
        ckpt_loader_config = trainer.hparams.checkpoint_loading
        match ckpt_loader_config:
            case "auto":
                ckpt_loader_config = CheckpointLoadingConfig.auto(ckpt_path, state_fn)
            case "none":
                ckpt_loader_config = CheckpointLoadingConfig.none()
            case _:
                pass
        log.debug(f"Checkpoint loader config: {ckpt_loader_config}")

        # Use the config to resolve the checkpoint.
        if (ckpt_path := _resolve_checkpoint(ckpt_loader_config, trainer)) is None:
            log.info(
                "No checkpoint found for the current trainer state. "
                "Training will start from scratch."
            )

        log.info(f"Loading checkpoint from: {ckpt_path}")
        return ckpt_path

    @override
    def _parse_ckpt_path(
        self,
        state_fn: TrainerFn,
        ckpt_path: str | Path | None,
        model_provided: bool,
        model_connected: bool,
    ):
        if (p := self.__resolve_auto_ckpt_path(ckpt_path, state_fn)) is not None:
            return p

        return super()._parse_ckpt_path(
            state_fn, ckpt_path, model_provided, model_connected
        )

    @override
    def dump_checkpoint(self, weights_only: bool = False):
        checkpoint = super().dump_checkpoint(weights_only)

        # Save the trainer's config.
        _add_trainer_config_to_checkpoint_(checkpoint, self.trainer)

        return checkpoint


def _add_trainer_config_to_checkpoint_(checkpoint: dict, trainer):
    from .trainer import Trainer

    # If this isn't an `nshtrainer` trainer (which I don't know why it wouldn't be),
    # then we just return.
    if isinstance(trainer, Trainer):
        return None

    # Save the trainer's config.
    checkpoint[trainer.CHECKPOINT_HYPER_PARAMS_KEY] = dict(trainer.hparams)
