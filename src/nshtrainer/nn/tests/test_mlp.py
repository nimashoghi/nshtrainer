from __future__ import annotations

from typing import cast

import pytest
import torch

from nshtrainer.nn.mlp import MLP, custom_seed_context


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
    layer1_weights1 = cast(torch.Tensor, mlp1[0].weight)
    layer1_weights2 = cast(torch.Tensor, mlp2[0].weight)
    layer1_weights3 = cast(torch.Tensor, mlp3[0].weight)

    # Same seed should produce identical weights
    assert torch.allclose(layer1_weights1, layer1_weights2)

    # Different seeds should produce different weights
    assert not torch.allclose(layer1_weights1, layer1_weights3)

    # Check second layer weights
    layer2_weights1 = cast(torch.Tensor, mlp1[2].weight)
    layer2_weights2 = cast(torch.Tensor, mlp2[2].weight)
    layer2_weights3 = cast(torch.Tensor, mlp3[2].weight)

    # Same seed should produce identical weights for all layers
    assert torch.allclose(layer2_weights1, layer2_weights2)

    # Different seeds should produce different weights for all layers
    assert not torch.allclose(layer2_weights1, layer2_weights3)

    # Test that not providing a seed gives different results each time
    mlp4 = MLP(dims, activation=torch.nn.ReLU(), seed=None)
    mlp5 = MLP(dims, activation=torch.nn.ReLU(), seed=None)

    # Without seeds, weights should be different
    assert not torch.allclose(
        cast(torch.Tensor, mlp4[0].weight), cast(torch.Tensor, mlp5[0].weight)
    )


def test_custom_seed_context():
    """Test that custom_seed_context properly controls random number generation."""

    # Test that the same seed produces the same random numbers
    with custom_seed_context(42):
        tensor1 = torch.randn(10)

    with custom_seed_context(42):
        tensor2 = torch.randn(10)

    # Same seed should produce identical random tensors
    assert torch.allclose(tensor1, tensor2)

    # Test that different seeds produce different random numbers
    with custom_seed_context(123):
        tensor3 = torch.randn(10)

    # Different seeds should produce different random tensors
    assert not torch.allclose(tensor1, tensor3)


def test_custom_seed_context_preserves_state():
    """Test that custom_seed_context preserves the original random state."""

    # Set a known seed for the test
    original_seed = 789
    torch.manual_seed(original_seed)

    # Generate a tensor with the original seed
    original_tensor = torch.randn(10)

    # Use a different seed in the context
    with custom_seed_context(42):
        # This should use the temporary seed
        context_tensor = torch.randn(10)

    # After exiting the context, we should be back to the original seed state
    # Reset the generator to get the same sequence again
    torch.manual_seed(original_seed)
    expected_tensor = torch.randn(10)

    # The tensor generated after the context should match what we would get
    # if we had just set the original seed again
    assert torch.allclose(original_tensor, expected_tensor)

    # And it should be different from the tensor generated inside the context
    assert not torch.allclose(original_tensor, context_tensor)


def test_custom_seed_context_with_none():
    """Test that custom_seed_context with None seed doesn't affect randomization."""

    # Set a known seed
    torch.manual_seed(555)
    expected_tensor = torch.randn(10)

    # Reset and use None seed in context
    torch.manual_seed(555)
    with custom_seed_context(None):
        actual_tensor = torch.randn(10)

    # With None seed, the context should not affect the random state
    assert torch.allclose(expected_tensor, actual_tensor)
