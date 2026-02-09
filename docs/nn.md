# Neural Network Utilities

The `nshtrainer.nn` module provides configuration-driven neural network building blocks and typed container classes.

## MLP

A flexible Multi-Layer Perceptron factory with config-driven construction.

### Basic Usage

```python
from nshtrainer.nn import MLP

# Create a 3-layer MLP: 128 → 64 → 32 → 1
mlp = MLP(
    dims=[128, 64, 32, 1],
    activation="relu",
)
```

### Configuration-Driven

```python
from nshtrainer.configs import MLPConfig

config = MLPConfig(
    dims=[128, 64, 32, 1],
    activation={"name": "gelu"},
    dropout=0.1,
    bias=True,
    residual=True,
    layer_norm=True,
    pre_norm=False,
    seed=42,  # Deterministic initialization
)

mlp = config.create_module()
```

### MLP Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `dims` | `list[int]` | Required | Layer dimensions (including input and output) |
| `activation` | `NonlinearityConfig` | Required | Activation function |
| `dropout` | `float` | `0.0` | Dropout rate between layers |
| `bias` | `bool` | `True` | Include bias in linear layers |
| `residual` | `bool` | `False` | Add residual connections (requires matching dims) |
| `layer_norm` | `bool` | `False` | Apply LayerNorm |
| `pre_norm` | `bool` | `False` | LayerNorm before activation (vs. after) |
| `seed` | `int \| None` | `None` | Random seed for deterministic initialization |

When `residual=True`, the MLP uses `ResidualSequential` which adds skip connections: `output = x + f(x)`.

When `seed` is provided, initialization uses `rng_context` to fork the RNG state, ensuring deterministic weights regardless of the global random state.

## Nonlinearity Registry

Activation functions are registered in the `nonlinearity_registry` and can be specified by name in configs:

```python
from nshtrainer.configs import GELUNonlinearityConfig

# Create as a module
activation = GELUNonlinearityConfig(approximate="tanh").create_module()

# Or use directly as a function
output = GELUNonlinearityConfig()(input_tensor)
```

### Available Nonlinearities

| Config | Name | Notes |
|--------|------|-------|
| `ReLUNonlinearityConfig` | `"relu"` | Standard ReLU |
| `GELUNonlinearityConfig` | `"gelu"` | GELU with optional `approximate` (`"tanh"` or `"none"`) |
| `SiLUNonlinearityConfig` | `"silu"` | SiLU (Swish) |
| `SwishNonlinearityConfig` | `"swish"` | Alias for SiLU |
| `MishNonlinearityConfig` | `"mish"` | Mish activation |
| `SwiGLUNonlinearityConfig` | `"swiglu"` | SwiGLU (splits input, applies SiLU gate) |
| `SigmoidNonlinearityConfig` | `"sigmoid"` | Sigmoid |
| `TanhNonlinearityConfig` | `"tanh"` | Tanh |
| `SoftmaxNonlinearityConfig` | `"softmax"` | Softmax with configurable `dim` |
| `SoftplusNonlinearityConfig` | `"softplus"` | Softplus with `beta` and `threshold` |
| `SoftsignNonlinearityConfig` | `"softsign"` | Softsign |
| `ELUNonlinearityConfig` | `"elu"` | ELU with configurable `alpha` |
| `LeakyReLUNonlinearityConfig` | `"leaky_relu"` | Leaky ReLU with `negative_slope` |
| `PReLUConfig` | `"prelu"` | Parametric ReLU (learnable, module-only) |

Each config class can both `create_module()` (returns `nn.Module`) and be called directly as a function `config(tensor)`.

## TypedModuleList

A generic wrapper around `nn.ModuleList` that preserves type information:

```python
from nshtrainer.nn import TypedModuleList

layers: TypedModuleList[nn.Linear] = TypedModuleList([
    nn.Linear(64, 64),
    nn.Linear(64, 64),
])

# Full type safety — IDE knows each element is nn.Linear
for layer in layers:
    x = layer(x)  # No type errors

# Indexing returns the correct type
first: nn.Linear = layers[0]
```

## TypedModuleDict

A generic wrapper around `nn.ModuleDict` with automatic key prefixing to avoid name collisions with `nn.Module` attributes:

```python
from nshtrainer.nn import TypedModuleDict

heads: TypedModuleDict[nn.Linear] = TypedModuleDict({
    "classification": nn.Linear(64, 10),
    "regression": nn.Linear(64, 1),
})

# Access by key — returns the correct type
cls_head: nn.Linear = heads["classification"]

# Iteration
for name, module in heads.items():
    output = module(features)
```

The key prefixing is transparent — you use normal string keys while the internal `nn.ModuleDict` uses prefixed keys to avoid collisions with built-in `nn.Module` attributes like `training`.

## RNG Context Manager

For deterministic module initialization:

```python
from nshtrainer.nn import RNGConfig, rng_context

config = RNGConfig(seed=42)

with rng_context(config):
    # All random operations in this block use the forked seed
    layer = nn.Linear(64, 64)
    # Global RNG state is restored after the block
```

This uses `torch.random.fork_rng` internally, forking across all CUDA devices. Passing `None` for the config makes the context manager a no-op.
