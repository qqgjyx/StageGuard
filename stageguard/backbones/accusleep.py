"""AccuSleep backbone: compact 2-layer CNN for mouse EEG/EMG."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BackboneBase


class AccuSleepBackbone(BackboneBase):
    """Simplified AccuSleep-style 2-layer CNN.

    Expects input of shape (B, T, 1, epoch_samples) — single-channel EEG
    epochs. Outputs (B, T, num_classes) logits.
    """

    def __init__(
        self, num_classes: int = 3, in_channels: int = 1, hidden: int = 64
    ) -> None:
        super().__init__(num_classes)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, in_channels, epoch_samples) raw signal epochs.

        Returns:
            logits: (B, T, num_classes).
        """
        B, T = x.shape[:2]
        # Merge batch and time
        x = x.reshape(B * T, *x.shape[2:])  # (B*T, C_in, samples)
        x = self.features(x)  # (B*T, hidden, 1)
        x = x.squeeze(-1)  # (B*T, hidden)
        logits = self.classifier(x)  # (B*T, num_classes)
        return logits.reshape(B, T, self.num_classes)
