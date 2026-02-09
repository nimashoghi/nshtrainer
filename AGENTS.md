# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Package Management with uv

Use uv exclusively for Python package management in this project.

### Package Management Commands

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync --all-extras --all-groups`

### Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools like Pytest with `uv run pytest` or `uv run ruff`
- Launch a Python repl with `uv run python`

## Project Overview

`nshtrainer` is a configuration-driven wrapper around PyTorch Lightning that simplifies deep learning experiment setup. It uses `nshconfig` (Pydantic-based) for type-safe, serializable configuration of every training aspect—model hyperparameters, logging, callbacks, hardware strategies.

## Development Commands

```bash
# Install (editable, with all extras)
uv sync --all-extras --all-groups

# Format
uv run ruff format src/ tests/

# Lint
uv run ruff check src/ tests/
uv run basedpyright src/

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/path/to/test_file.py

# Run a single test function
uv run pytest tests/path/to/test_file.py::test_function_name

# Build
uv build
```

## Architecture

### Config-Driven Design

Nearly every component has a paired `Config` class (e.g., `EMACallback` / `EMACallbackConfig`). Configs extend `nshconfig.Config` (Pydantic-based) and are registered in `nshconfig.Registry` objects, enabling instantiation from dicts/JSON/YAML. The `TrainerConfig` is the root configuration that composes all other configs.

### Registry Pattern

Components use registries for extensibility. For example, `callback_registry` allows adding new callback types by subclassing `CallbackConfigBase`. Registries are used with `typing.Annotated` and discriminated unions for type-safe deserialization. Key registries: `callback_registry`, `optimizer_registry`, `accelerator_registry`, `plugin_registry`.

### Core Classes

- **`Trainer`** (`src/nshtrainer/trainer/trainer.py`): Extends `lightning.pytorch.Trainer` with automatic directory setup, metadata saving, signal handling, and config-driven callback/logger instantiation.
- **`TrainerConfig`** (`src/nshtrainer/trainer/_config.py`): Root config composing all sub-configs (callbacks, loggers, checkpoints, strategies, etc.).
- **`LightningModuleBase`** (`src/nshtrainer/model/base.py`): Generic base class parameterized by `THparams` (a `nshconfig.Config` subclass). Provides type-safe `hparams`, profiling access, and distributed helpers (`all_gather_object`, `reduce`). Uses mixins for callback, debug, and logger functionality.
- **`LightningDataModuleBase`** (`src/nshtrainer/data/`): Base data module with transform and batch sampling utilities.

### Package Layout (`src/nshtrainer/`)

- `trainer/` — Trainer, TrainerConfig, strategy/accelerator/plugin configs
- `model/` — LightningModuleBase and mixins (callback, debug, logger)
- `callbacks/` — Custom Lightning callbacks (EMA, early stopping, checkpointing, gradient skipping, norm logging, etc.)
- `loggers/` — Logger wrappers (Wandb, Tensorboard, CSV, ActSave)
- `optimizer.py` — Registry-based optimizer config for factory-creating PyTorch optimizers
- `lr_scheduler/` — Registry-based LR scheduler configs
- `nn/` — Utility modules (MLP, typed ModuleList/ModuleDict)
- `configs/` — Central hub re-exporting all config classes
- `_checkpoint/` — Checkpoint saving/loading utilities
- `metrics/` — Metric configuration
- `profiler/` — Profiler configuration
- `util/` — Path management, environment detection (Slurm/LSF), hardware info

## Code Style

- **Formatter**: `ruff format` (mandatory before committing)
- **Linter/Type Checker**: `basedpyright` with `standard` type checking mode; `ruff check` for additional linting
- **`from __future__ import annotations`**: Required in every file (enforced by ruff `FA100`/`FA102` rules)
- **Type hints**: Mandatory on all parameters. Use modern syntax (`X | None`, `list[int]`, `collections.abc.Iterable`)
- **Return types**: Omit for implicit `None` returns and simple obvious types; include for complex/non-obvious types
- **Composition over inheritance**: Prefer functions and composition; avoid deep class hierarchies
- **Array typing**: Use `nshutils.typecheck` (jaxtyping-based) for tensor shape/dtype annotations (e.g., `tc.Float[torch.Tensor, "batch seq dim"]`)
- **Logging**: Use `logging` module, never `print()` for diagnostics
- **Docstrings**: Google style

## Claude Code Skill

Install the `using-nshtrainer` skill for Claude Code:

```bash
# Project-local
nshtrainer skill install

# Global (all projects)
nshtrainer skill install --global
```

## Documentation

User-facing documentation lives in:

- `README.md` — Project overview, installation, and quick start
- `docs/getting-started.md` — End-to-end tutorial
- `docs/configuration.md` — TrainerConfig in depth
- `docs/model.md` — LightningModuleBase and LightningDataModuleBase
- `docs/callbacks.md` — Built-in callbacks reference
- `docs/loggers.md` — Logger configuration
- `docs/optimizers-schedulers.md` — Optimizer and scheduler system
- `docs/checkpointing.md` — Checkpoint system and metadata
- `docs/distributed.md` — Distributed training and HPC support
- `docs/nn.md` — Neural network utilities (MLP, typed containers, nonlinearities)
