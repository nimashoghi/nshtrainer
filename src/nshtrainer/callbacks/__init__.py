from typing import Annotated

import nshconfig as C

from .base import CallbackConfigBase as CallbackConfigBase
from .early_stopping import EarlyStopping as EarlyStopping
from .ema import EMA as EMA
from .ema import EMAConfig as EMAConfig
from .finite_checks import FiniteChecksCallback as FiniteChecksCallback
from .finite_checks import FiniteChecksConfig as FiniteChecksConfig
from .gradient_skipping import GradientSkipping as GradientSkipping
from .gradient_skipping import GradientSkippingConfig as GradientSkippingConfig
from .interval import EpochIntervalCallback as EpochIntervalCallback
from .interval import IntervalCallback as IntervalCallback
from .interval import StepIntervalCallback as StepIntervalCallback
from .latest_epoch_checkpoint import LatestEpochCheckpoint as LatestEpochCheckpoint
from .latest_epoch_checkpoint import (
    LatestEpochCheckpointCallbackConfig as LatestEpochCheckpointCallbackConfig,
)
from .log_epoch import LogEpochCallback as LogEpochCallback
from .model_checkpoint import ModelCheckpoint as ModelCheckpoint
from .model_checkpoint import (
    ModelCheckpointCallbackConfig as ModelCheckpointCallbackConfig,
)
from .norm_logging import NormLoggingCallback as NormLoggingCallback
from .norm_logging import NormLoggingConfig as NormLoggingConfig
from .on_exception_checkpoint import OnExceptionCheckpoint as OnExceptionCheckpoint
from .on_exception_checkpoint import (
    OnExceptionCheckpointCallbackConfig as OnExceptionCheckpointCallbackConfig,
)
from .print_table import PrintTableMetricsCallback as PrintTableMetricsCallback
from .print_table import PrintTableMetricsConfig as PrintTableMetricsConfig
from .throughput_monitor import ThroughputMonitorConfig as ThroughputMonitorConfig
from .timer import EpochTimer as EpochTimer
from .timer import EpochTimerConfig as EpochTimerConfig
from .wandb_watch import WandbWatchCallback as WandbWatchCallback
from .wandb_watch import WandbWatchConfig as WandbWatchConfig

CallbackConfig = Annotated[
    ThroughputMonitorConfig
    | EpochTimerConfig
    | PrintTableMetricsConfig
    | FiniteChecksConfig
    | NormLoggingConfig
    | GradientSkippingConfig
    | EMAConfig
    | ModelCheckpointCallbackConfig
    | LatestEpochCheckpointCallbackConfig
    | OnExceptionCheckpointCallbackConfig
    | WandbWatchConfig,
    C.Field(discriminator="name"),
]
