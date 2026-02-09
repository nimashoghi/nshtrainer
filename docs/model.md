# Model

`nshtrainer` provides `LightningModuleBase` and `LightningDataModuleBase` as base classes that add type-safe hyperparameters, distributed utilities, and debug tools on top of PyTorch Lightning.

## LightningModuleBase

`LightningModuleBase[THparams]` is a generic class parameterized by your hyperparameter config type:

```python
import nshconfig as C
import nshtrainer as nt

class MyConfig(C.Config):
    hidden_size: int = 128
    lr: float = 1e-3

class MyModel(nt.LightningModuleBase[MyConfig]):
    @classmethod
    def hparams_cls(cls):
        return MyConfig

    def __init__(self, hparams: MyConfig):
        super().__init__(hparams)
        self.net = torch.nn.Linear(10, hparams.hidden_size)
```

### Required: `hparams_cls()`

Every subclass must implement this classmethod to return the config class:

```python
@classmethod
def hparams_cls(cls) -> type[MyConfig]:
    return MyConfig
```

This is used for:
- Automatic validation of incoming hyperparameters (dict or Config object)
- Checkpoint loading — reconstructing the config from saved data
- Type inference for `self.hparams`

### Type-Safe `self.hparams`

Unlike vanilla Lightning where `self.hparams` returns an `AttributeDict`, nshtrainer overrides it to return your specific config type. This gives you full IDE autocompletion and type checking:

```python
def training_step(self, batch, batch_idx):
    # self.hparams is typed as MyConfig
    lr = self.hparams.lr  # IDE knows this is a float
    return ...
```

## Distributed Helpers

### `all_gather_object(object)`

Gathers arbitrary Python objects from all processes into a list. Handles non-distributed environments gracefully:

```python
# On each rank, compute local results
local_result = {"accuracy": compute_accuracy(data)}

# Gather from all ranks
all_results = self.all_gather_object(local_result)
# all_results is a list of dicts, one per rank
```

### `reduce(tensor, reduce_op)`

Reduces a tensor across the process group. Supports both `ReduceOp` types and string aliases:

```python
total = self.reduce(local_count, "sum")
average = self.reduce(local_metric, "mean")
maximum = self.reduce(local_score, "max")
```

### `barrier(name=None)`

Synchronization barrier across all processes:

```python
self.barrier("waiting_for_data")
```

## `zero_loss()`

A utility for Distributed Data Parallel (DDP) when some parameters don't contribute to the loss (which would cause DDP errors). Returns a loss term that touches all parameters with zero gradient:

```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    # Ensure all parameters participate in backward pass
    loss = loss + self.zero_loss()
    return loss
```

## Logging Utilities

### `log_context`

A context manager that sets shared logging parameters for all `self.log()` calls within a block:

```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    metrics = self.compute_metrics(batch)

    with self.log_context(prefix="train/", prog_bar=True):
        self.log("loss", loss)          # logged as "train/loss"
        self.log("accuracy", metrics)   # logged as "train/accuracy"

    return loss
```

Contexts are stackable — nested contexts concatenate prefixes and merge settings.

### `logging_enabled`

A property that respects the trainer's `barebones` mode:

```python
if self.logging_enabled:
    self.log("expensive_metric", compute_expensive_metric())
```

## Loading from Checkpoints

### `from_checkpoint()`

The recommended way to load a model from a checkpoint (instead of Lightning's `load_from_checkpoint`):

```python
model = MyModel.from_checkpoint("path/to/checkpoint.ckpt")
```

You can modify hyperparameters during loading:

```python
# Update the config object
model = MyModel.from_checkpoint(
    "checkpoint.ckpt",
    update_hparams=lambda config: config.model_copy(update={"lr": 1e-4}),
)

# Or update the raw dict before validation
model = MyModel.from_checkpoint(
    "checkpoint.ckpt",
    update_hparams_dict=lambda d: {**d, "lr": 1e-4},
)
```

### `hparams_from_checkpoint()`

Extract just the hyperparameters without loading the full model:

```python
config = MyModel.hparams_from_checkpoint("checkpoint.ckpt")
print(config.hidden_size)  # 128
```

## Debug Utilities

### `breakpoint(rank_zero_only=True)`

A distributed-aware breakpoint. With `rank_zero_only=True` (default), it triggers only on rank 0 and places a barrier on other ranks to prevent timeouts:

```python
def training_step(self, batch, batch_idx):
    if batch_idx == 42:
        self.breakpoint()  # Only rank 0 enters the debugger
    return self.compute_loss(batch)
```

### `ensure_finite(tensor, name=None, throw=False)`

Checks a tensor for NaN or Inf values:

```python
logits = self.net(x)
self.ensure_finite(logits, name="logits")  # Logs warning if non-finite
self.ensure_finite(logits, name="logits", throw=True)  # Raises RuntimeError
```

### `debug` property

A boolean flag synced with the trainer's debug mode. Useful for enabling verbose logging or extra checks in debug runs:

```python
if self.debug:
    self.ensure_finite(logits, "logits", throw=True)
```

## Callback Registration

Models and their sub-modules can dynamically register Lightning callbacks using the `register_callback()` method from the `CallbackModuleMixin`:

```python
def __init__(self, hparams):
    super().__init__(hparams)
    # Register a callback that will be added to the trainer
    self.register_callback(MyCustomCallback())
```

Registered callbacks are gathered during `configure_callbacks()`.

## LightningDataModuleBase

`LightningDataModuleBase[THparams]` follows the same pattern as the model base:

```python
class MyDataConfig(C.Config):
    batch_size: int = 32
    num_workers: int = 4

class MyDataModule(nt.LightningDataModuleBase[MyDataConfig]):
    @classmethod
    def hparams_cls(cls):
        return MyDataConfig

    def __init__(self, hparams: MyDataConfig):
        super().__init__(hparams)

    def train_dataloader(self):
        return DataLoader(..., batch_size=self.hparams.batch_size)
```

It provides the same type-safe `self.hparams`, `from_checkpoint()`, `hparams_from_checkpoint()`, debug mixin, and callback registration as the model base.
