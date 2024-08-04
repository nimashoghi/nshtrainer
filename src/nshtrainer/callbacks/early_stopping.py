import logging
import math
from typing import Literal

from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping as _EarlyStopping
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message
from typing_extensions import override

from .base import CallbackConfigBase

log = logging.getLogger(__name__)


class EarlyStoppingConfig(CallbackConfigBase):
    name: Literal["early_stopping"] = "early_stopping"

    monitor: str | None = None
    """
    The metric to monitor for early stopping.
    If None, the primary metric will be used.
    """

    mode: Literal["min", "max"] | None = None
    """
    The mode for the metric to monitor for early stopping.
    If None, the primary metric mode will be used.
    """

    patience: int
    """
    Number of epochs with no improvement after which training will be stopped.
    """

    min_delta: float = 1.0e-8
    """
    Minimum change in the monitored quantity to qualify as an improvement.
    """

    min_lr: float | None = None
    """
    Minimum learning rate. If the learning rate of the model is less than this value,
    the training will be stopped.
    """

    strict: bool = True
    """
    Whether to enforce that the monitored quantity must improve by at least `min_delta`
    to qualify as an improvement.
    """

    @override
    def create_callbacks(self, root_config):
        monitor = self.monitor
        mode = self.mode
        if monitor is None:
            assert mode is None, "If `monitor` is not provided, `mode` must be None."

            primary_metric = root_config.primary_metric
            if primary_metric is None:
                raise ValueError(
                    "No primary metric is set, so `monitor` must be provided in `early_stopping`."
                )
            monitor = primary_metric.validation_monitor
            mode = primary_metric.mode

        if mode is None:
            mode = "min"

        yield EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=self.patience,
            min_delta=self.min_delta,
            min_lr=self.min_lr,
            strict=self.strict,
        )


class EarlyStopping(_EarlyStopping):
    def __init__(
        self,
        monitor: str,
        min_delta: float = 0,
        min_lr: float | None = None,
        patience: int = 3,
        verbose: bool = True,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: float | None = None,
        divergence_threshold: float | None = None,
        check_on_train_epoch_end: bool | None = None,
        log_rank_zero_only: bool = False,
    ):
        super().__init__(
            monitor,
            min_delta,
            patience,
            verbose,
            mode,
            strict,
            check_finite,
            stopping_threshold,
            divergence_threshold,
            check_on_train_epoch_end,
            log_rank_zero_only,
        )

        self.min_lr = min_lr

    @override
    @staticmethod
    def _log_info(
        trainer: Trainer | None, message: str, log_rank_zero_only: bool
    ) -> None:
        rank = _get_rank()
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.critical(message)

    @override
    def _run_early_stopping_check(self, trainer: Trainer):
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        # Disable early_stopping with fast_dev_run
        if getattr(trainer, "fast_dev_run", False):
            return

        should_stop, reason = False, None

        if not should_stop:
            should_stop, reason = self._evaluate_stopping_criteria_min_lr(trainer)

        # If metric present
        if not should_stop and self._validate_condition_metric(logs):
            current = logs[self.monitor].squeeze()
            should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _evaluate_stopping_criteria_min_lr(
        self, trainer: Trainer
    ) -> tuple[bool, str | None]:
        if self.min_lr is None:
            return False, None

        # Get the maximum LR across all param groups in all optimizers
        model_max_lr = max(
            [
                param_group["lr"]
                for optimizer in trainer.optimizers
                for param_group in optimizer.param_groups
            ]
        )
        if not isinstance(model_max_lr, float) or not math.isfinite(model_max_lr):
            return False, None

        # If the maximum LR is less than the minimum LR, stop training
        if model_max_lr >= self.min_lr:
            return False, None

        return True, (
            "Stopping threshold reached: "
            f"The maximum LR of the model across all param groups is {model_max_lr:.2e} "
            f"which is less than the minimum LR {self.min_lr:.2e}"
        )

    def on_early_stopping(self, trainer: Trainer):
        pass
