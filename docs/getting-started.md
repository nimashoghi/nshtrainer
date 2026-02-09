# Getting Started

This guide walks through a complete training setup with `nshtrainer`.

## Installation

```bash
pip install nshtrainer

# With all optional extras (wandb, tensorboard, huggingface-hub, etc.)
pip install nshtrainer[extra]
```

For development:

```bash
git clone https://github.com/nimashoghi/nshtrainer.git
cd nshtrainer
pip install -e ".[extra]"
```

## Defining Model Hyperparameters

All hyperparameters are defined as `nshconfig.Config` classes — Pydantic models with type validation, serialization, and IDE autocompletion:

```python
import nshconfig as C

class MyModelConfig(C.Config):
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.01
```

## Implementing Your Model

Subclass `LightningModuleBase`, which is generic over your config type. You must implement `hparams_cls()` to tell nshtrainer which config class to use:

```python
import torch
import torch.nn as nn
from typing_extensions import override

import nshtrainer

class MyModel(nshtrainer.LightningModuleBase[MyModelConfig]):
    @override
    @classmethod
    def hparams_cls(cls):
        return MyModelConfig

    def __init__(self, hparams: MyModelConfig):
        super().__init__(hparams)

        layers = []
        in_dim = 10
        for _ in range(hparams.num_layers):
            layers.append(nn.Linear(in_dim, hparams.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(hparams.dropout))
            in_dim = hparams.hidden_size
        layers.append(nn.Linear(hparams.hidden_size, 1))
        self.net = nn.Sequential(*layers)

    @override
    def forward(self, x: torch.Tensor):
        return self.net(x)

    @override
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        self.log("train_loss", loss)
        return loss

    @override
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        self.log("val_loss", loss)

    @override
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
```

Key points:
- `self.hparams` is typed as `MyModelConfig` — full IDE autocompletion
- Hyperparameters are automatically saved and restored from checkpoints
- The standard Lightning methods (`training_step`, `configure_optimizers`, etc.) all work as expected

## Setting Up TrainerConfig

`TrainerConfig` is the root configuration. Key fields to set:

```python
import nshtrainer

trainer_config = nshtrainer.TrainerConfig(
    # Training loop
    max_epochs=100,
    accelerator="cuda",
    precision="bf16-mixed",

    # The metric used for best-checkpoint selection, early stopping, etc.
    primary_metric=nshtrainer.MetricConfig(name="val_loss", mode="min"),
)

# Set the output directory (returns a new config copy)
trainer_config = trainer_config.with_project_root("./experiments")

# Optionally set a run name and tags
trainer_config = trainer_config.with_name("my-experiment")
trainer_config = trainer_config.with_tags("baseline", "v1")
```

See [Configuration](configuration.md) for the full list of options.

## Running Training

```python
trainer = nshtrainer.Trainer(trainer_config)
model = MyModel(MyModelConfig())

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
```

## Directory Structure

After a training run, nshtrainer creates the following directory structure under your project root:

```
./experiments/
└── nshtrainer_logs/
    └── <run_id>/
        ├── checkpoint/
        │   ├── best/
        │   │   └── epoch=5-step=300.ckpt
        │   ├── last/
        │   │   └── epoch=9-step=600.ckpt
        │   ├── best_val_loss.ckpt  → symlink to best checkpoint
        │   └── last.ckpt           → symlink to last checkpoint
        ├── log/
        │   ├── wandb/
        │   ├── tensorboard/
        │   └── csv/
        ├── stdio/
        └── activation/
```

- `checkpoint/` — Model checkpoints with metadata JSON files
- `log/` — Logger-specific outputs (WandB, TensorBoard, CSV)
- `stdio/` — Captured console output
- `activation/` — Activation logging artifacts

A `.gitignore` is automatically created in the `nshtrainer_logs/` directory.

## Resuming from a Checkpoint

To resume training from a checkpoint, set the `ckpt_path` on the config:

```python
# Resume from a specific checkpoint
trainer_config = trainer_config.with_ckpt_path("./experiments/nshtrainer_logs/abc123/checkpoint/last.ckpt")
trainer = nshtrainer.Trainer(trainer_config)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

To load a model for inference without a full trainer:

```python
model = MyModel.from_checkpoint("path/to/checkpoint.ckpt")
```

## Next Steps

- [Configuration](configuration.md) — Deep dive into TrainerConfig
- [Model](model.md) — LightningModuleBase features (logging, distributed helpers, debug tools)
- [Callbacks](callbacks.md) — Built-in callbacks (EMA, gradient skipping, norm logging, etc.)
- [Checkpointing](checkpointing.md) — Checkpoint system and metadata
