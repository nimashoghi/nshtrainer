from __future__ import annotations

import pytest
import torch

from nshtrainer.nn.mlp import MLP


def test_mlp_seed_reproducibility():
    """Test that the seed parameter in MLP ensures reproducible weights."""

    # Test dimensions
    dims = [10, 20, 5]

    # Create two MLPs with the same seed
    seed1 = 42
    mlp1 = MLP(dims, activation=torch.nn.ReLU(), seed=seed1)
    mlp2 = MLP(dims, activation=torch.nn.ReLU(), seed=seed1)

    # Create an MLP with a different seed
    seed2 = 123
    mlp3 = MLP(dims, activation=torch.nn.ReLU(), seed=seed2)

    # Check first layer weights
    layer1_weights1 = mlp1[0].weight
    layer1_weights2 = mlp2[0].weight
    layer1_weights3 = mlp3[0].weight

    # Same seed should produce identical weights
    assert torch.allclose(layer1_weights1, layer1_weights2)

    # Different seeds should produce different weights
    assert not torch.allclose(layer1_weights1, layer1_weights3)

    # Check second layer weights
    layer2_weights1 = mlp1[2].weight
    layer2_weights2 = mlp2[2].weight
    layer2_weights3 = mlp3[2].weight

    # Same seed should produce identical weights for all layers
    assert torch.allclose(layer2_weights1, layer2_weights2)

    # Different seeds should produce different weights for all layers
    assert not torch.allclose(layer2_weights1, layer2_weights3)

    # Test that not providing a seed gives different results each time
    mlp4 = MLP(dims, activation=torch.nn.ReLU(), seed=None)
    mlp5 = MLP(dims, activation=torch.nn.ReLU(), seed=None)

    # Without seeds, weights should be different
    assert not torch.allclose(mlp4[0].weight, mlp5[0].weight)
