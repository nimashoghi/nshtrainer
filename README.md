# nshtrainer

A configuration-driven wrapper around [PyTorch Lightning](https://lightning.ai/) that simplifies deep learning experiment setup. Built on [nshconfig](https://github.com/nimashoghi/nshconfig) (Pydantic-based) for type-safe, serializable configuration of every training aspect.

## Key Features

- **Type-safe configuration** — Every component (callbacks, loggers, optimizers, schedulers) has a paired `Config` class with full IDE autocompletion and validation
- **Automatic checkpointing with metadata** — Best/last/on-exception checkpoints with JSON metadata files containing metrics, environment info, git state, and SHA256 checksums
- **Environment capture** — Automatically records hardware info, installed packages, git state, and cluster details (SLURM/LSF) with every run
- **Registry-based extensibility** — Add custom callbacks, optimizers, schedulers, and loggers by subclassing and registering
- **HPC support** — Automatic node detection on SLURM/LSF clusters, signal handling, and auto-requeue on preemption
- **Builder-style API** — Fluent configuration with `with_*()` (returns copy) and `*_()` (in-place) methods
- **HuggingFace Hub integration** — Optionally push checkpoints to HuggingFace Hub

## Installation

```bash
pip install nshtrainer

# With all optional dependencies (wandb, tensorboard, etc.)
pip install nshtrainer[extra]
```

## Quick Start

```python
import nshconfig as C
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override

import nshtrainer as nt

# 1. Define your hyperparameters as a config class
class MyModelConfig(C.Config):
    hidden_size: int = 64
    lr: float = 1e-3

# 2. Subclass LightningModuleBase with your config
class MyModel(nt.LightningModuleBase[MyModelConfig]):
    @override
    @classmethod
    def hparams_cls(cls):
        return MyModelConfig

    def __init__(self, hparams: MyModelConfig):
        super().__init__(hparams)
        self.net = torch.nn.Linear(10, hparams.hidden_size)
        self.head = torch.nn.Linear(hparams.hidden_size, 1)

    @override
    def forward(self, x: torch.Tensor):
        return self.head(torch.relu(self.net(x)))

    @override
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    @override
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# 3. Configure the trainer
trainer_config = nt.TrainerConfig(
    max_epochs=10,
    accelerator="cpu",
    primary_metric=nt.MetricConfig(name="train_loss", mode="min"),
).with_project_root("./outputs")

# 4. Train
trainer = nt.Trainer(trainer_config)
model = MyModel(MyModelConfig())

dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
trainer.fit(model, train_dataloaders=DataLoader(dataset, batch_size=16))
```

## Documentation

- [Getting Started](docs/getting-started.md) — End-to-end tutorial
- [Configuration](docs/configuration.md) — TrainerConfig in depth
- [Model](docs/model.md) — LightningModuleBase and LightningDataModuleBase
- [Callbacks](docs/callbacks.md) — Built-in callbacks reference
- [Loggers](docs/loggers.md) — Logger configuration
- [Optimizers & Schedulers](docs/optimizers-schedulers.md) — Registry-based optimizer and scheduler system
- [Checkpointing](docs/checkpointing.md) — Checkpoint system and metadata
- [Distributed Training](docs/distributed.md) — Strategies, accelerators, and HPC support
- [Neural Network Utilities](docs/nn.md) — MLP, typed containers, nonlinearities

## License

See [LICENSE](LICENSE) for details.
