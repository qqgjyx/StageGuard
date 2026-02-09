"""Shared test fixtures."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from stageguard.config import ModalityConfig


@pytest.fixture
def dummy_config():
    """Minimal 3-class config for testing."""
    return ModalityConfig(
        stage_names=["Wake", "NREM", "REM"],
        num_classes=3,
        rare_transitions=[(0, 2), (2, 0)],
        lambda_trans=1.0,
        d_max=10,
        epsilon=5.0,
        gamma=2.0,
        k=5,
        d_min=[2, 2, 2],
        sqi_method="spectral_entropy",
        sqi_threshold=0.5,
        epoch_sec=30.0,
    )


@pytest.fixture
def binary_config():
    """Minimal 2-class config (no rare transitions)."""
    return ModalityConfig(
        stage_names=["Wake", "Sleep"],
        num_classes=2,
        rare_transitions=[],
        lambda_trans=0.5,
        d_max=10,
        epsilon=5.0,
        gamma=2.0,
        k=5,
        d_min=[2, 2],
        sqi_method="acceleration_variance",
        sqi_threshold=0.3,
        epoch_sec=30.0,
    )


@pytest.fixture
def random_logits():
    """Random (B=2, T=20, C=3) logits tensor."""
    torch.manual_seed(42)
    return torch.randn(2, 20, 3)


@pytest.fixture
def random_targets():
    """Random (B=2, T=20) integer targets."""
    torch.manual_seed(123)
    return torch.randint(0, 3, (2, 20))


@pytest.fixture
def random_log_probs():
    """Random (B=2, T=20, C=3) log-probabilities (NumPy)."""
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((2, 20, 3))
    # Log-softmax
    max_val = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_val
    log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - log_sum_exp


class TinyBackbone(nn.Module):
    """Minimal backbone for testing: just a linear layer."""

    def __init__(self, in_features=8, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (B, T, in_features)
        B, T, F = x.shape
        return self.linear(x.reshape(-1, F)).reshape(B, T, self.num_classes)


@pytest.fixture
def tiny_backbone():
    """Tiny backbone for wrapper tests."""
    return TinyBackbone(in_features=8, num_classes=3)
