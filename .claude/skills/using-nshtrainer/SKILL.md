---
name: using-nshtrainer
description: Config-driven PyTorch Lightning wrapper with type-safe configs and registries. Use when building training pipelines with nshtrainer, configuring TrainerConfig or callbacks, creating LightningModuleBase subclasses, or setting up optimizers/schedulers/loggers via registry configs.
---

# nshtrainer

Configuration-driven wrapper around PyTorch Lightning. Every component has a paired `Config` class using `nshconfig.Config` (Pydantic-based).

## Import Convention

```python
import nshtrainer
# Core: nshtrainer.Trainer, nshtrainer.TrainerConfig
# Model: nshtrainer.LightningModuleBase
# Data: nshtrainer.LightningDataModuleBase
# Metric: nshtrainer.MetricConfig
```

## Core Pattern

```python
import nshconfig as C
from typing_extensions import override

# 1. Config class for hyperparameters
class MyModelConfig(C.Config):
    hidden_size: int = 64
    lr: float = 1e-3

# 2. Model subclass parameterized by config
class MyModel(nshtrainer.LightningModuleBase[MyModelConfig]):
    @override
    @classmethod
    def hparams_cls(cls):
        return MyModelConfig

    def __init__(self, hparams: MyModelConfig):
        super().__init__(hparams)
        # Access config via self.hparams (type-safe)

# 3. Configure trainer
config = nshtrainer.TrainerConfig(
    max_epochs=10,
    accelerator="gpu",
    primary_metric=nshtrainer.MetricConfig(name="val_loss", mode="min"),
).with_project_root("./outputs")

# 4. Train
trainer = nshtrainer.Trainer(config)
trainer.fit(model, train_dataloaders=..., val_dataloaders=...)
```

## TrainerConfig

Root config composing all sub-configs. Builder methods: `with_*()` returns copy, `*_()` mutates in-place.

Key fields: `max_epochs`, `accelerator`, `strategy`, `primary_metric`, `callbacks` (dict of callback configs), `loggers`, `checkpoint`, `precision`, `gradient_clip_val`.

## Registries

Extensible component registration via `nshconfig.Registry` + discriminated unions:

| Registry | Purpose | Example |
|----------|---------|---------|
| `callback_registry` | Custom callbacks | Subclass `CallbackConfigBase` |
| `optimizer_registry` | Optimizers | Subclass `OptimizerConfigBase` |
| `accelerator_registry` | Accelerators | Subclass config |
| `plugin_registry` | Plugins | Subclass config |

## Built-in Callbacks

EMA, early stopping, model checkpointing, gradient skipping, norm logging, learning rate monitoring, and more. Configure via `TrainerConfig.callbacks` dict.

## Code Style Rules

- `from __future__ import annotations` in every file
- Type hints on all parameters (modern syntax: `X | None`, `list[int]`)
- `ruff format` before committing, `basedpyright` for type checking
- `logging` module only, never `print()`
- Google-style docstrings
- Composition over inheritance

## Detailed Documentation

For in-depth reference on specific topics, see:
- [Getting Started](references/getting-started.md)
- [Configuration](references/configuration.md)
- [Model](references/model.md)
- [Callbacks](references/callbacks.md)
- [Loggers](references/loggers.md)
- [Optimizers & Schedulers](references/optimizers-schedulers.md)
- [Checkpointing](references/checkpointing.md)
- [Distributed Training](references/distributed.md)
- [Neural Network Utilities](references/nn.md)
