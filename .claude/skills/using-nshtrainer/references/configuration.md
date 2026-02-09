# Configuration

`nshtrainer` is configuration-driven: nearly every component has a paired `Config` class built on [nshconfig](https://github.com/nimashoghi/nshconfig) (Pydantic-based). `TrainerConfig` is the root that composes all other configs.

## How nshconfig Works

`nshconfig.Config` extends Pydantic's `BaseModel` with:
- Full type validation and serialization (JSON, YAML, dict)
- Deep validation via `model_deep_validate()`
- Registry support for polymorphic deserialization via discriminated unions

```python
import nshconfig as C

class MyConfig(C.Config):
    lr: float = 1e-3
    hidden_size: int = 128

# Create from dict
config = MyConfig.model_validate({"lr": 1e-4, "hidden_size": 256})

# Serialize to dict/JSON
config.model_dump()
config.model_dump_json()
```

## TrainerConfig Overview

`TrainerConfig` is the central configuration object. Here are its key fields organized by category.

### Run Identity

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | Auto-generated | Unique 8-character alphanumeric run ID |
| `name` | `list[str]` | `[]` | Run name parts (joined with spaces) |
| `project` | `str \| None` | `None` | Project name for organization/logging |
| `tags` | `list[str]` | `[]` | Tags for filtering and grouping |
| `notes` | `list[str]` | `[]` | Human-readable notes |
| `meta` | `dict[str, Any]` | `{}` | Arbitrary metadata dictionary |

### Training Loop

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_epochs` | `int \| None` | `None` | Maximum training epochs (defaults to 1000 if `max_steps` not set) |
| `max_steps` | `int` | `-1` | Maximum training steps (-1 = no limit) |
| `precision` | `str \| None` | `None` | Training precision (e.g., `"bf16-mixed"`, `"16-mixed"`, `"32-true"`) |
| `gradient_clipping` | `GradientClippingConfig \| None` | `None` | Gradient clipping (value or norm) |
| `fast_dev_run` | `int \| bool` | `False` | Quick validation run |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

### Infrastructure

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `accelerator` | `AcceleratorConfig` | `"auto"` | Hardware accelerator (`"auto"`, `"cpu"`, `"cuda"`, `"mps"`, `"xla"`) |
| `devices` | `int \| list[int] \| str \| None` | `None` | Device selection |
| `strategy` | `StrategyConfig` | `"auto"` | Distributed strategy (`"auto"`, `"ddp"`, `"fsdp"`, `"deepspeed"`, etc.) |
| `num_nodes` | `int` | `1` | Number of nodes for distributed training |
| `plugins` | `list[PluginConfig]` | `[]` | Precision, environment, and I/O plugins |

### Metrics and Checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `primary_metric` | `MetricConfig \| None` | `None` | The metric for best-checkpoint, early stopping, and LR scheduling |
| `checkpoint_saving` | `CheckpointSavingConfig \| None` | Auto | Checkpoint callback configuration |
| `ckpt_path` | `str \| Path \| None` | `None` | Path to resume training from |
| `early_stopping` | `EarlyStoppingCallbackConfig \| None` | `None` | Early stopping configuration |

### Logging

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `loggers` | `Sequence[LoggerConfig] \| None` | `None` | Logger configs (defaults to WandB + CSV + TensorBoard) |
| `log_norms` | `NormLoggingCallbackConfig \| None` | `None` | Gradient/parameter norm logging |
| `log_epoch` | `LogEpochCallbackConfig \| None` | Auto | Epoch logging as a metric |
| `lr_monitor` | `LearningRateMonitorConfig \| None` | Auto | Learning rate monitoring |

### Advanced

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `barebones` | `bool` | `False` | Minimal mode for benchmarking (disables loggers, checkpoints, etc.) |
| `debug` | `bool` | `False` | Debug mode |
| `callbacks` | `list[CallbackConfig]` | `[]` | Additional callback configs |
| `set_float32_matmul_precision` | `str \| None` | `None` | Torch matmul precision (e.g., `"high"` for Tensor Cores) |
| `auto_determine_num_nodes` | `bool` | `True` | Auto-detect SLURM/LSF node count |

## Builder-Style API

TrainerConfig provides two patterns for modification:

### `with_*()` — Returns a deep copy (functional)

```python
config = TrainerConfig(max_epochs=10)

# Each call returns a new config, leaving the original unchanged
config_a = config.with_project_root("./exp_a")
config_b = config.with_project_root("./exp_b")
```

### `*_()` — Modifies in place (imperative)

```python
config = TrainerConfig(max_epochs=10)

# Mutates and returns self for chaining
config.project_root_("./experiments").name_("my-run").tags_("v1", "baseline")
```

Available builder methods:

| Functional (`with_*`) | In-place (`*_`) | Purpose |
|------------------------|-----------------|---------|
| `with_project_root(path)` | `project_root_(path)` | Set output directory |
| `with_name(*parts)` | `name_(*parts)` | Set run name |
| `with_id(id)` | `id_(id)` | Set run ID |
| `with_project(name)` | `project_(name)` | Set project name |
| `with_tags(*tags)` | `tags_(*tags)` | Set tags |
| `with_added_tags(*tags)` | `add_tags_(*tags)` | Append tags |
| `with_notes(*notes)` | `notes_(*notes)` | Set notes |
| `with_meta(**kwargs)` | `meta_(**kwargs)` | Update metadata |
| `with_debug(flag)` | `debug_(flag)` | Set debug mode |
| `with_fast_dev_run(n)` | `fast_dev_run_(n)` | Set fast dev run |
| `with_ckpt_path(path)` | `ckpt_path_(path)` | Set checkpoint path |
| `with_barebones(flag)` | `barebones_(flag)` | Set barebones mode |

There is also `reset_run()` which returns a copy with a new ID and cleared metadata while keeping architectural settings intact.

## Directory Layout

nshtrainer organizes artifacts under a predictable path:

```
{project_root}/nshtrainer_logs/{run_id}/
├── checkpoint/     # Model checkpoints
├── log/            # Logger outputs (wandb/, tensorboard/, csv/)
├── stdio/          # Captured console output
└── activation/     # ActSave artifacts
```

The `project_root` is set via `with_project_root()` or `project_root_()`. A `.gitignore` file is automatically created in the `nshtrainer_logs/` directory to exclude training artifacts from version control.

## The `primary_metric` Field

`primary_metric` is a `MetricConfig` that drives several automated behaviors:

```python
config = TrainerConfig(
    primary_metric=nshtrainer.MetricConfig(name="val_loss", mode="min"),
)
```

When set, it automatically configures:
- **Best checkpoint selection** — saves the model with the best metric value
- **Early stopping** — if an `EarlyStoppingCallbackConfig` is provided, it uses this metric
- **LR scheduling** — `ReduceLROnPlateau` schedulers use this metric if no explicit metric is given

`MetricConfig` has two fields:
- `name` — the metric name as logged via `self.log()` (e.g., `"val_loss"`)
- `mode` — `"min"` (lower is better) or `"max"` (higher is better)

## Barebones Mode

Setting `barebones=True` strips away high-overhead features for performance profiling:

- **Disabled**: All loggers, checkpointing, progress bars, model summaries, sanity validation steps, anomaly detection
- **Active**: Core training loop only

```python
config = TrainerConfig(barebones=True, max_epochs=5)
```

## Passing Extra Lightning Arguments

For Lightning `Trainer` parameters not explicitly mapped in `TrainerConfig`:

```python
config = TrainerConfig(
    # Type-safe common parameters
    lightning_kwargs={
        "accumulate_grad_batches": 4,
        "val_check_interval": 0.5,
        "limit_train_batches": 100,
    },
    # Escape hatch for any Lightning parameter
    additional_lightning_kwargs={
        "enable_progress_bar": False,
    },
)
```
