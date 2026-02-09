# Distributed Training & HPC

nshtrainer provides configuration-driven support for distributed training strategies, hardware accelerators, and HPC cluster integration.

## Strategy Configuration

Strategies control how training is distributed across devices and nodes. Set via `TrainerConfig.strategy`:

```python
config = TrainerConfig(
    strategy="ddp",          # String literal
    devices=4,
    num_nodes=1,
)
```

### Available Strategy Literals

| Strategy | Description |
|----------|-------------|
| `"auto"` | Automatic selection based on hardware |
| `"ddp"` | Distributed Data Parallel |
| `"ddp_find_unused_parameters_true"` | DDP with unused parameter detection |
| `"ddp_spawn"` | DDP using process spawning |
| `"fsdp"` | Fully Sharded Data Parallel |
| `"fsdp_native"` | Native FSDP implementation |
| `"deepspeed"` | DeepSpeed ZeRO Stage 1 |
| `"deepspeed_stage_2"` | DeepSpeed ZeRO Stage 2 |
| `"deepspeed_stage_3"` | DeepSpeed ZeRO Stage 3 |
| `"deepspeed_stage_3_offload"` | DeepSpeed Stage 3 with CPU offloading |
| `"single_device"` | Single device (no distribution) |

### Custom Strategy Configs

For advanced configuration, use a `StrategyConfigBase` subclass instead of a string literal. These provide full control over strategy parameters.

## Accelerator Configuration

Accelerators specify the hardware backend. Set via `TrainerConfig.accelerator`:

```python
config = TrainerConfig(accelerator="cuda")
```

### Available Accelerators

| Config Class | Name | Description |
|-------------|------|-------------|
| `CPUAcceleratorConfig` | `"cpu"` | CPU training |
| `CUDAAcceleratorConfig` | `"cuda"` / `"gpu"` | NVIDIA GPU training |
| `MPSAcceleratorConfig` | `"mps"` | Apple Silicon GPU |
| `XLAAcceleratorConfig` | `"xla"` / `"tpu"` | Google TPU via XLA |

Use `"auto"` (default) to let Lightning choose the best available accelerator.

## Plugin System

Plugins extend trainer behavior for precision, cluster environments, and checkpoint I/O:

```python
from nshtrainer.configs import (
    MixedPrecisionPluginConfig,
    SLURMEnvironmentPlugin,
    AsyncCheckpointIOPlugin,
)

config = TrainerConfig(
    plugins=[
        MixedPrecisionPluginConfig(),
        SLURMEnvironmentPlugin(auto_requeue=True),
        AsyncCheckpointIOPlugin(),
    ],
)
```

### Precision Plugins

| Plugin | Description |
|--------|-------------|
| `MixedPrecisionPluginConfig` | Standard AMP mixed precision |
| `HalfPrecisionPluginConfig` | FP16 precision |
| `DoublePrecisionPluginConfig` | FP64 precision |
| `FSDPPrecisionPluginConfig` | FSDP-specific precision |
| `DeepSpeedPluginConfig` | DeepSpeed precision |
| `BitsandbytesPluginConfig` | Bitsandbytes quantization |
| `TransformerEnginePluginConfig` | NVIDIA Transformer Engine |
| `XLAPluginConfig` | XLA/TPU precision |

### Environment Plugins

| Plugin | Description |
|--------|-------------|
| `SLURMEnvironmentPlugin` | SLURM cluster support with auto-requeue |
| `LSFEnvironmentPlugin` | IBM LSF cluster support |
| `TorchElasticEnvironmentPlugin` | Fault-tolerant elastic training |
| `KubeflowEnvironmentPlugin` | Kubeflow PyTorchJob |
| `LightningEnvironmentPlugin` | Default Lightning environment |
| `MPIEnvironmentPlugin` | MPI-based distributed |
| `XLAEnvironmentPlugin` | XLA/TPU Pod environment |

### I/O Plugins

| Plugin | Description |
|--------|-------------|
| `TorchCheckpointIOPlugin` | Standard PyTorch checkpoint I/O |
| `AsyncCheckpointIOPlugin` | Asynchronous checkpoint saving |
| `XLACheckpointIOPlugin` | XLA-specific checkpoint I/O |

### Layer Sync

| Plugin | Description |
|--------|-------------|
| `TorchSyncBatchNormPlugin` | Synchronized BatchNorm across devices |

## Automatic Node Detection

nshtrainer can automatically detect the number of nodes on SLURM and LSF clusters:

```python
config = TrainerConfig(
    auto_determine_num_nodes=True,  # Default
)
```

When enabled:
- **SLURM**: Reads `SLURM_NNODES` environment variable
- **LSF**: Parses `LSB_HOSTS` to count unique hosts

This eliminates the need to manually set `num_nodes` in cluster job scripts.

## Signal Handling and Auto-Requeue

nshtrainer extends Lightning's signal handling for graceful preemption on HPC clusters.

### SLURM

When a SLURM job receives a preemption signal (e.g., `SIGUSR1`):

1. Training is gracefully stopped
2. An HPC checkpoint is saved
3. Loggers are finalized
4. The job is requeued via `scontrol requeue`

Configure via the SLURM environment plugin:

```python
from nshtrainer.configs import SLURMEnvironmentPlugin

config = TrainerConfig(
    plugins=[
        SLURMEnvironmentPlugin(
            auto_requeue=True,
            requeue_signal="SIGUSR1",  # Optional custom signal
        ),
    ],
)
```

### LSF

Similar flow for LSF clusters, using `brequeue` for requeuing.

### nshrunner Integration

When used with [nshrunner](https://github.com/nimashoghi/nshrunner), nshtrainer can handle custom termination signals and write requeue scripts for cluster-specific workflows.

## Distributed Prediction

For multi-GPU inference with result aggregation:

```python
trainer = Trainer(config)

# Runs prediction across all devices and gathers results
predictions = trainer.distributed_predict(
    model,
    dataloaders=predict_loader,
)
```

This uses `DistributedPredictionWriter` internally to save per-rank predictions and aggregate them.

## Environment Scrubbing

When launching nested jobs (e.g., hyperparameter sweeps from within a SLURM job), nshtrainer provides utilities to clean up inherited environment variables:

```python
from nshtrainer.util.environment import (
    remove_slurm_environment_variables,
    remove_lsf_environment_variables,
)

with remove_slurm_environment_variables():
    # Launch sub-jobs without inheriting SLURM_* variables
    pass
```

## BFloat16 Support Detection

Check for hardware BFloat16 support (Ampere+ GPUs or ROCm):

```python
from nshtrainer.util.bf16 import is_bf16_supported_no_emulation

if is_bf16_supported_no_emulation():
    config = TrainerConfig(precision="bf16-mixed")
```
