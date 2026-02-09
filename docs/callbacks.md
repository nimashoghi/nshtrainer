# Callbacks

nshtrainer includes a comprehensive set of built-in callbacks, all following the config-driven pattern. Each callback has a paired `Config` class that can be added to your `TrainerConfig`.

## The Callback Registry

Callbacks use a registry system for extensibility:

```python
from nshtrainer.callbacks.base import CallbackConfigBase, callback_registry

@callback_registry.register
class MyCallbackConfig(CallbackConfigBase):
    name: Literal["my_callback"] = "my_callback"
    threshold: float = 0.5

    def create_callbacks(self, trainer_config):
        yield MyCallback(self.threshold)
```

Each config class:
- Subclasses `CallbackConfigBase`
- Has a `name` field used as a discriminator for deserialization
- Implements `create_callbacks(trainer_config)` which yields Lightning `Callback` instances

### Callback Metadata

Configs can specify metadata via `CallbackMetadataConfig`:
- `priority` — Integer determining loading order (higher = loaded first)
- `ignore_if_exists` — Prevents duplicate callbacks of the same class
- `enabled_for_barebones` — Whether the callback runs in barebones mode

## Checkpointing Callbacks

### BestCheckpointCallbackConfig

Saves the model with the best value of a monitored metric.

```python
from nshtrainer import TrainerConfig, MetricConfig

config = TrainerConfig(
    primary_metric=MetricConfig(name="val_loss", mode="min"),
    # Best checkpoint is auto-configured when primary_metric is set
)
```

Key options: `metric`, `topk`, `filename`, `save_weights_only`, `save_symlink`

### LastCheckpointCallbackConfig

Saves the most recent model state. Can also save periodically based on time.

Key options: `topk`, `filename`, `save_on_time_interval` (e.g., save every hour)

### OnExceptionCheckpointCallbackConfig

Emergency checkpoint save when training crashes. Includes deadlock prevention for distributed training.

Key options: `dirpath`, `filename`

## Training Dynamics

### EMACallbackConfig

Maintains Exponential Moving Averages of model weights for smoother evaluation.

```python
from nshtrainer.configs import EMACallbackConfig

config = TrainerConfig(
    callbacks=[
        EMACallbackConfig(decay=0.999, every_n_steps=1),
    ],
)
```

Key options: `decay`, `validate_original_weights`, `cpu_offload`, `every_n_steps`

### GradientSkippingCallbackConfig

Skips optimizer steps when gradient norms exceed a threshold, preventing training instability from anomalous batches.

```python
from nshtrainer.configs import GradientSkippingCallbackConfig

config = TrainerConfig(
    callbacks=[
        GradientSkippingCallbackConfig(threshold=100.0),
    ],
)
```

Key options: `threshold`, `norm_type`, `start_after_n_steps`, `skip_non_finite`

### SharedParametersCallbackConfig

Scales gradients for shared (tied) parameters to avoid gradient over-accumulation. The model must implement a `shared_parameters` property returning the shared parameter groups.

### EarlyStoppingCallbackConfig

Stops training when a monitored metric stops improving. Set at the top level of `TrainerConfig`:

```python
from nshtrainer.configs import EarlyStoppingCallbackConfig

config = TrainerConfig(
    primary_metric=MetricConfig(name="val_loss", mode="min"),
    early_stopping=EarlyStoppingCallbackConfig(patience=10, min_delta=1e-4),
)
```

Key options: `metric`, `patience`, `min_delta`, `min_lr`, `skip_first_n_epochs`

## Monitoring

### NormLoggingCallbackConfig

Logs gradient and parameter norms to your loggers.

```python
from nshtrainer.configs import NormLoggingCallbackConfig

config = TrainerConfig(
    log_norms=NormLoggingCallbackConfig(
        log_grad_norm=True,
        log_param_norm=True,
    ),
)
```

Key options: `log_grad_norm`, `log_param_norm`, `log_grad_norm_per_param`

### LearningRateMonitorConfig

Logs the learning rate of all optimizers. Enabled by default.

Key options: `logging_interval`, `log_momentum`, `log_weight_decay`

### MetricValidationCallbackConfig

Ensures that expected metrics are actually being logged during training.

Key options: `metrics` (list of metric names to check), `error_behavior` (`"raise"` or `"warn"`)

### PrintTableMetricsCallbackConfig

Prints a formatted table of metrics to the console at each validation epoch.

Key options: `metric_patterns` (glob patterns to filter which metrics to display)

### LogEpochCallbackConfig

Logs the current epoch as a metric (supports fractional epochs). Enabled by default.

Key options: `metric_name`, `train`, `val`, `test`

### EpochTimerCallbackConfig

Measures and logs the duration of each training/validation/test epoch.

## Debugging

### FiniteChecksCallbackConfig

Checks for NaN, Inf, or None gradients after the backward pass.

Key options: `nonfinite_grads` (action on non-finite), `none_grads` (action on None gradients)

### DebugFlagCallbackConfig

Automatically sets the debug flag to `True` during fast dev runs and sanity check validation. Enabled by default.

Key options: `enabled`

## Integrations

### WandbWatchCallbackConfig

Calls `wandb.watch()` to log model gradients and topology to Weights & Biases.

Key options: `log` (log type), `log_graph`, `log_freq`

### WandbUploadCodeCallbackConfig

Uploads a snapshot of the current codebase to Weights & Biases.

Key options: `enabled`

### ActSaveConfig

Manages activation saving contexts using the `nshutils.ActSave` library.

Key options: `enabled`, `save_dir`

## Adding Callbacks to TrainerConfig

Callbacks can be configured in several ways:

```python
from nshtrainer import TrainerConfig
from nshtrainer.configs import (
    EMACallbackConfig,
    GradientSkippingCallbackConfig,
    NormLoggingCallbackConfig,
)

config = TrainerConfig(
    # Top-level config fields for common callbacks
    log_norms=NormLoggingCallbackConfig(log_grad_norm=True),
    early_stopping=EarlyStoppingCallbackConfig(patience=10),

    # The callbacks list for everything else
    callbacks=[
        EMACallbackConfig(decay=0.999),
        GradientSkippingCallbackConfig(threshold=100.0),
    ],
)
```

## Custom Callbacks

To add a custom callback:

1. Define a config class and register it:

```python
from typing import Literal
from typing_extensions import final, override
from nshtrainer.callbacks.base import CallbackConfigBase, callback_registry
from lightning.pytorch.callbacks import Callback

@final
@callback_registry.register
class MyCallbackConfig(CallbackConfigBase):
    name: Literal["my_callback"] = "my_callback"
    log_every_n_steps: int = 50

    @override
    def create_callbacks(self, trainer_config):
        yield MyCallback(self.log_every_n_steps)

class MyCallback(Callback):
    def __init__(self, log_every_n_steps: int):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            # Custom logging logic
            pass
```

2. Add it to your config:

```python
config = TrainerConfig(
    callbacks=[MyCallbackConfig(log_every_n_steps=100)],
)
```
