# Optimizers & Schedulers

nshtrainer provides a registry-based system for configuring optimizers and LR schedulers through serializable config objects.

## Optimizer System

### Using Optimizer Configs

Optimizer configs live in `nshtrainer.optimizer` and follow the same registry pattern as other components. Each config has a `create_optimizer(params)` method:

```python
from nshtrainer.configs import AdamWConfig

optimizer_config = AdamWConfig(lr=1e-3, weight_decay=0.01)
optimizer = optimizer_config.create_optimizer(model.parameters())
```

Typically, you use these in your model's `configure_optimizers`:

```python
class MyModel(nshtrainer.LightningModuleBase[MyConfig]):
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer.create_optimizer(self.parameters())
        return optimizer
```

### Available Optimizers

| Config Class | Name | Description |
|-------------|------|-------------|
| `AdamWConfig` | `"adamw"` | AdamW (decoupled weight decay) |
| `AdamConfig` | `"adam"` | Standard Adam |
| `SGDConfig` | `"sgd"` | Stochastic Gradient Descent |
| `AdafactorConfig` | `"adafactor"` | Memory-efficient adaptive optimizer |
| `AdadeltaConfig` | `"adadelta"` | Adadelta |
| `AdagradConfig` | `"adagrad"` | Adagrad |
| `AdamaxConfig` | `"adamax"` | Adamax (infinity-norm Adam variant) |
| `ASGDConfig` | `"asgd"` | Averaged SGD |
| `NAdamConfig` | `"nadam"` | NAdam (Adam with Nesterov momentum) |
| `RAdamConfig` | `"radam"` | RAdam (rectified Adam) |
| `RMSpropConfig` | `"rmsprop"` | RMSprop |
| `RpropConfig` | `"rprop"` | Rprop |

### Common Optimizer Options

Using `AdamWConfig` as an example:

```python
from nshtrainer.configs import AdamWConfig

optimizer = AdamWConfig(
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    amsgrad=False,
    fused=False,      # Use fused CUDA implementation
    foreach=None,     # Use foreach implementation
)
```

### Optimizer Registry

The `optimizer_registry` allows registering custom optimizers:

```python
from nshtrainer.optimizer import OptimizerConfigBase, optimizer_registry

@optimizer_registry.register
class MyOptimizerConfig(OptimizerConfigBase):
    name: Literal["my_optimizer"] = "my_optimizer"
    lr: float = 1e-3

    def create_optimizer(self, params):
        return MyOptimizer(params, lr=self.lr)
```

## LR Scheduler System

### Using Scheduler Configs

LR scheduler configs define both the scheduler and its metadata (update interval, monitoring behavior):

```python
from nshtrainer.configs import LinearWarmupCosineDecayLRSchedulerConfig

scheduler_config = LinearWarmupCosineDecayLRSchedulerConfig(
    warmup_duration={"name": "steps", "value": 1000},
    max_duration={"name": "epochs", "value": 100},
)
```

Use in `configure_optimizers`:

```python
class MyModel(nshtrainer.LightningModuleBase[MyConfig]):
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = self.hparams.scheduler.create_scheduler(optimizer, self)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

### Available Schedulers

#### LinearWarmupCosineDecayLRSchedulerConfig

Linear warmup followed by cosine decay. Operates at the **step** level for smooth updates.

```python
from nshtrainer.configs import LinearWarmupCosineDecayLRSchedulerConfig, StepsConfig, EpochsConfig

scheduler = LinearWarmupCosineDecayLRSchedulerConfig(
    warmup_duration=StepsConfig(value=1000),     # or EpochsConfig(value=5)
    max_duration=EpochsConfig(value=100),
    warmup_start_lr_factor=0.0,   # Start from 0
    min_lr_factor=0.0,            # Decay to 0
    annealing=False,              # True = cosine annealing with restarts
)
```

#### ReduceLROnPlateauConfig

Reduces learning rate when a metric stops improving. Operates at the **epoch** level.

```python
from nshtrainer.configs import ReduceLROnPlateauConfig

scheduler = ReduceLROnPlateauConfig(
    patience=10,
    factor=0.1,
    # metric=MetricConfig(...)  # Uses primary_metric if not specified
    cooldown=0,
    min_lr=0.0,
    threshold=1e-4,
    threshold_mode="rel",
)
```

### Scheduler Metadata

Each scheduler config produces `LRSchedulerMetadata` that tells Lightning how to use it:

| Field | Values | Description |
|-------|--------|-------------|
| `interval` | `"step"` or `"epoch"` | When to call `scheduler.step()` |
| `frequency` | `int` | How often to step within the interval |
| `monitor` | `str` | Metric to monitor (for plateau schedulers) |
| `reduce_on_plateau` | `bool` | Whether this is a plateau-style scheduler |

### Duration Config

Schedulers accept durations in steps or epochs via `DurationConfig`:

```python
from nshtrainer.configs import StepsConfig, EpochsConfig

# 1000 training steps
warmup = StepsConfig(value=1000)

# 5 training epochs (automatically converted to steps internally)
warmup = EpochsConfig(value=5)
```

### LR Scheduler Registry

Custom schedulers can be registered:

```python
from nshtrainer.lr_scheduler.base import LRSchedulerConfigBase, lr_scheduler_registry

@lr_scheduler_registry.register
class MySchedulerConfig(LRSchedulerConfigBase):
    name: Literal["my_scheduler"] = "my_scheduler"

    def metadata(self):
        return {"interval": "step"}

    def create_scheduler_impl(self, optimizer, lightning_module):
        return MyScheduler(optimizer)
```
