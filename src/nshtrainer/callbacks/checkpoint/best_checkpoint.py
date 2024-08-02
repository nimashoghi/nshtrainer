import logging
from pathlib import Path
from typing import Any, Literal

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Checkpoint
from typing_extensions import override

from ..._checkpoint.metadata import _sort_ckpts_by_metadata
from ..._checkpoint.saver import _link_checkpoint, _remove_checkpoint
from ...metrics._config import MetricConfig
from ..base import CallbackConfigBase

log = logging.getLogger(__name__)


class BestCheckpointCallbackConfig(CallbackConfigBase):
    name: Literal["best_checkpoint"] = "best_checkpoint"

    dirpath: str | Path | None = None
    """Directory path to save the checkpoint file."""

    filename: str = "epoch{epoch:03d}_step{step:07d}_{metric_name}{metric}"
    """Checkpoint filename. This must not include the extension."""

    save_weights_only: bool = False
    """Whether to save only the model's weights or the entire model object."""

    metric: MetricConfig | None = None
    """Metric to monitor, or `None` to use the default metric."""

    best_symlink_filename: str | None = "best"
    """Filename for the best symlink. If None, no symlink will be created."""

    save_top_k: int | Literal["all"] = 1
    """The number of best checkpoints to keep."""

    @override
    def create_callbacks(self, root_config):
        checkpoint_folder = Path(
            self.dirpath
            or root_config.directory.resolve_subdirectory(root_config.id, "checkpoint")
        )
        dirpath = checkpoint_folder / "best"
        dirpath.mkdir(parents=True, exist_ok=True)

        # Resolve metric
        if (metric := self.metric) is None and (
            metric := root_config.primary_metric
        ) is None:
            raise ValueError(
                "No metric provided and no primary metric found in the root config"
            )

        yield BestCheckpoint(self, metric, dirpath, checkpoint_folder)

    @property
    def _save_top_k_value(self):
        return float("inf" if self.save_top_k == "all" else self.save_top_k)


class BestCheckpoint(Checkpoint):
    PREFIX = "best_"
    EXTENSION = ".ckpt"

    def __init__(
        self,
        config: BestCheckpointCallbackConfig,
        metric: MetricConfig,
        dirpath: Path,
        symlink_dirpath: Path | None = None,
    ):
        super().__init__()
        self.config = config
        self.metric = metric
        self.dirpath = dirpath
        self.symlink_dirpath = symlink_dirpath or dirpath

        self._last_global_step_saved = 0  # no need to save when no steps were taken

    @override
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        self._save_best_checkpoint(trainer)

    def _best_symlink_filename(self):
        if (filename := self.config.best_symlink_filename) is None:
            return None
        return f"{filename}{self.EXTENSION}"

    def _ckpt_path(self, trainer: Trainer, metric_value: float):
        # Escape the metric name:
        # - Replace '/' with '_'
        # - Replace '.' with '_'
        metric_name = self.metric.name.replace("/", "_").replace(".", "_")
        filename = self.config.filename.format(
            epoch=trainer.current_epoch,
            step=trainer.global_step,
            metric_name=metric_name,
            metric=metric_value,
        )
        filename = f"{self.PREFIX}{filename}{self.EXTENSION}"
        return self.dirpath / filename

    def _get_metric_value(self, metrics: dict[str, Any]):
        return metrics.get(
            self.metric.validation_monitor,
            float("-inf" if self.metric.mode == "max" else "inf"),
        )

    def _sorted_ckpts(self):
        """
        Get sorted checkpoints by the metric value.

        Sort order: best -> worst
        """
        ckpt_paths = list(self.dirpath.glob(f"{self.PREFIX}*{self.EXTENSION}"))
        return _sort_ckpts_by_metadata(
            ckpt_paths,
            key=lambda meta, _: self._get_metric_value(meta.metrics),
            reverse=(self.metric.mode == "max"),
        )

    def _create_symlink(self, trainer: Trainer, best_ckpt_path: Path):
        # Resolve the symlink filename
        if (symlink_filename := self._best_symlink_filename()) is None:
            return

        # If the symlink already exists and points to the best checkpoint,
        # then we don't need to create a new symlink.
        symlink_path = self.symlink_dirpath / symlink_filename
        if symlink_path.exists() and symlink_path.resolve() == best_ckpt_path:
            return

        _link_checkpoint(best_ckpt_path, symlink_path, metadata=True)
        log.debug(f"Created best symlink: {symlink_path}")

    def _save_best_checkpoint(self, trainer: Trainer):
        # Skip saving the checkpoint if we're not in the fitting state
        if self._should_skip_saving_checkpoint(trainer):
            return

        # Get the current metric value
        if (current := self._get_metric_value(trainer.callback_metrics)) is None:
            log.warning(
                f"Can't save best model, {self.metric.validation_monitor} not found in metrics"
            )
            return

        # Get sorted checkpoints
        sorted_ckpts = self._sorted_ckpts()

        # If the current model is worse than the worst checkpoint,
        # and we have already saved the maximum number of checkpoints,
        # then don't save the current model.
        if len(
            sorted_ckpts
        ) >= self.config._save_top_k_value and not self.metric.is_better(
            current,
            self._get_metric_value(sorted_ckpts[-1][0].metrics),
        ):
            return

        # Save the current model
        filepath = self._ckpt_path(trainer, current)
        trainer.save_checkpoint(filepath, self.config.save_weights_only)
        log.debug(f"Saved best checkpoint: {filepath}")

        if trainer.is_global_zero:
            # Get the sorted checkpoints again because now we have added a new checkpoint.
            # We could optimize this by adding the new checkpoint to the sorted list,
            # and then sorting it in place, but this is simpler.
            sorted_ckpts = self._sorted_ckpts()

            # Remove worst checkpoint if we've reached save_top_k
            if (topk := self.config.save_top_k) != "all" and len(sorted_ckpts) > topk:
                # NOTE: Sort order is best -> worst. Let's get the worst checkpoints.
                for _, ckpt_path in sorted_ckpts[topk:]:
                    _remove_checkpoint(trainer, ckpt_path, metadata=True)

            # Create symlink to best model
            if sorted_ckpts:
                _, best_ckpt_path = sorted_ckpts[0]
                self._create_symlink(trainer, best_ckpt_path)

        # Update the last global step saved
        self._last_global_step_saved = trainer.global_step

        # Barrier to ensure all processes have saved the checkpoint before continuing
        trainer.strategy.barrier()

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
