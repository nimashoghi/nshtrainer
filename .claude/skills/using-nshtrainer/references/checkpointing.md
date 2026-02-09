# Checkpointing

nshtrainer extends Lightning's checkpointing with automatic metadata, symlinks, and structured directory management.

## Automatic Checkpoints

When `primary_metric` is set, nshtrainer automatically configures three checkpoint callbacks:

### Best Checkpoint

Saves the model with the best value of the primary metric. A symlink `best_<metric_name>.ckpt` points to the current best:

```
checkpoint/
├── best/
│   ├── epoch=3-step=150.ckpt
│   └── epoch=3-step=150.ckpt.metadata.json
└── best_val_loss.ckpt → best/epoch=3-step=150.ckpt
```

### Last Checkpoint

Saves the most recent model state. A symlink `last.ckpt` always points to it:

```
checkpoint/
├── last/
│   ├── epoch=9-step=500.ckpt
│   └── epoch=9-step=500.ckpt.metadata.json
└── last.ckpt → last/epoch=9-step=500.ckpt
```

The last checkpoint can also save periodically based on wall-clock time via `save_on_time_interval`.

### On-Exception Checkpoint

Saves an emergency checkpoint when training crashes. Includes deadlock prevention for distributed training:

```
checkpoint/
└── on_exception/
    └── 2024-01-15T10-30-00.ckpt
```

## Checkpoint Configuration

Checkpointing is configured via the `checkpoint_saving` field on `TrainerConfig`:

```python
from nshtrainer.configs import CheckpointSavingConfig

config = TrainerConfig(
    primary_metric=MetricConfig(name="val_loss", mode="min"),
    checkpoint_saving=CheckpointSavingConfig(
        # These are configured automatically when primary_metric is set
    ),
)
```

Individual checkpoint callbacks can also be configured:

```python
from nshtrainer.configs import (
    BestCheckpointCallbackConfig,
    LastCheckpointCallbackConfig,
)

config = TrainerConfig(
    callbacks=[
        BestCheckpointCallbackConfig(
            metric=MetricConfig(name="val_loss", mode="min"),
            topk=3,                    # Keep top 3 checkpoints
            save_weights_only=False,   # Save full training state
        ),
        LastCheckpointCallbackConfig(
            topk=1,                    # Keep only the latest
        ),
    ],
)
```

## Symlink System

nshtrainer creates symlinks for convenient access to important checkpoints:

| Symlink | Points to |
|---------|-----------|
| `last.ckpt` | Most recent checkpoint |
| `best_<metric>.ckpt` | Best checkpoint by the named metric |

Metadata files (`.metadata.json`) are also symlinked alongside their checkpoints.

On systems that don't support symlinks (e.g., some Windows configurations), files are copied instead.

## Metadata Files

Every checkpoint is accompanied by a `.metadata.json` file containing:

```json
{
    "run_id": "abc12345",
    "epoch": 5,
    "global_step": 300,
    "training_time_seconds": 1234.5,
    "metrics": {
        "val_loss": 0.123,
        "train_loss": 0.456
    },
    "trainer_config": { ... },
    "environment": {
        "hardware": {
            "cpu_count": 32,
            "gpu_count": 4,
            "gpu_devices": [...]
        },
        "packages": [...],
        "git": {
            "branch": "main",
            "commit": "abc123...",
            "dirty": false
        },
        "slurm": { ... }
    },
    "checkpoint_checksum": "sha256:abcdef..."
}
```

Key sections:
- **`metrics`** — Final metric values at the time of saving
- **`trainer_config`** — Full serialized TrainerConfig for reproducibility
- **`environment`** — Hardware, software, and cluster environment snapshot
- **`checkpoint_checksum`** — SHA256 hash of the checkpoint file for integrity verification

## Resuming Training

### From a Checkpoint Path

```python
config = TrainerConfig(
    max_epochs=100,
    primary_metric=MetricConfig(name="val_loss", mode="min"),
).with_ckpt_path("path/to/checkpoint.ckpt")

trainer = Trainer(config)
trainer.fit(model, train_dataloaders=..., val_dataloaders=...)
```

### Using `"none"` to Skip

If `ckpt_path` is set but you want to skip loading for a run:

```python
config = config.with_ckpt_path("none")
```

## Loading Model Weights

To load a trained model for inference (without resuming training):

```python
# Load with original hyperparameters
model = MyModel.from_checkpoint("path/to/checkpoint.ckpt")

# Modify hyperparameters during loading
model = MyModel.from_checkpoint(
    "path/to/checkpoint.ckpt",
    update_hparams=lambda config: config.model_copy(update={"dropout": 0.0}),
)

# Extract just the config
config = MyModel.hparams_from_checkpoint("path/to/checkpoint.ckpt")
```

## Checkpoint Directory Structure

The full checkpoint directory structure:

```
{project_root}/nshtrainer_logs/{run_id}/checkpoint/
├── best/
│   ├── epoch=3-step=150.ckpt
│   └── epoch=3-step=150.ckpt.metadata.json
├── last/
│   ├── epoch=9-step=500.ckpt
│   └── epoch=9-step=500.ckpt.metadata.json
├── on_exception/
│   └── 2024-01-15T10-30-00.ckpt
├── best_val_loss.ckpt           → symlink
├── best_val_loss.ckpt.metadata.json → symlink
├── last.ckpt                    → symlink
└── last.ckpt.metadata.json      → symlink
```
