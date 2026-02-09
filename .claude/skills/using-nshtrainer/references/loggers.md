# Loggers

nshtrainer wraps PyTorch Lightning's logging system with a configuration-driven approach. By default, three loggers are enabled: Weights & Biases, CSV, and TensorBoard.

## Default Loggers

When `loggers` is not explicitly set in `TrainerConfig`, the following loggers are enabled:

- **WandB** (`WandbLoggerConfig`) — Weights & Biases experiment tracking
- **CSV** (`CSVLoggerConfig`) — Local CSV file logging
- **TensorBoard** (`TensorboardLoggerConfig`) — TensorBoard logging

```python
# Default behavior — all three loggers active
config = TrainerConfig(max_epochs=10)

# Explicitly specify loggers
from nshtrainer.configs import WandbLoggerConfig, CSVLoggerConfig

config = TrainerConfig(
    loggers=[
        WandbLoggerConfig(),
        CSVLoggerConfig(),
    ],
)
```

## Configuring Loggers

### Weights & Biases

```python
from nshtrainer.configs import WandbLoggerConfig

config = TrainerConfig(
    loggers=[
        WandbLoggerConfig(
            # WandB project name (defaults to TrainerConfig.project)
            # Additional WandB-specific settings are auto-configured
        ),
    ],
)
```

The WandB logger integrates with two optional callback configs on `TrainerConfig`:
- `WandbWatchCallbackConfig` — calls `wandb.watch()` to log gradients
- `WandbUploadCodeCallbackConfig` — uploads a code snapshot to WandB

### TensorBoard

```python
from nshtrainer.configs import TensorboardLoggerConfig

config = TrainerConfig(
    loggers=[
        TensorboardLoggerConfig(
            log_graph=False,        # Whether to log the computation graph
            default_hp_metric=True, # Log default HP metric
        ),
    ],
)
```

### CSV

```python
from nshtrainer.configs import CSVLoggerConfig

config = TrainerConfig(
    loggers=[
        CSVLoggerConfig(
            flush_logs_every_n_steps=100,
        ),
    ],
)
```

## Disabling Loggers

To disable all loggers, either use barebones mode or pass an empty list:

```python
# Barebones mode disables everything
config = TrainerConfig(barebones=True)

# Or explicitly disable loggers
config = TrainerConfig(loggers=[])
```

## ActSave Logger

The `ActSaveLoggerConfig` integrates with [nshutils](https://github.com/nimashoghi/nshutils) for activation logging. It is configured separately from the main loggers list:

```python
from nshtrainer.configs import ActSaveLoggerConfig

config = TrainerConfig(
    actsave_logger=ActSaveLoggerConfig(enabled=True),
)
```

## Logger Registry

Loggers use the same registry pattern as callbacks. The `logger_registry` allows adding custom logger types:

```python
from nshtrainer.loggers.base import LoggerConfigBase, logger_registry

@logger_registry.register
class MyLoggerConfig(LoggerConfigBase):
    name: Literal["my_logger"] = "my_logger"

    def create_logger(self, trainer_config):
        return MyCustomLogger(...)
```

## Log Directory Structure

Each logger writes to a subdirectory under the run's `log/` directory:

```
{project_root}/nshtrainer_logs/{run_id}/log/
├── wandb/
├── tensorboard/
└── csv/
```
